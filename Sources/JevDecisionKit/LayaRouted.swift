import CoreML
import Foundation

/// Mehrere laya-Checkpoints hinter einer Schnittstelle, mit der Wahl aus `LayaRouter`.
///
/// Entspricht `laya.router.Router.predict`: erst entscheiden, dann rechnen, und die Entscheidung
/// mit ausliefern. Geladen wird erst beim ersten Gebrauch, damit ein Prozess, der nur englische
/// Anfragen sieht, nicht die 615 MiB des mehrsprachigen Checkpoints mitschleppt.
public actor LayaRouted {
    public struct Source: Sendable {
        public let modelURL: URL
        public let tokenizerURL: URL

        public init(modelURL: URL, tokenizerURL: URL) {
            self.modelURL = modelURL
            self.tokenizerURL = tokenizerURL
        }
    }

    public struct Routed: Sendable {
        public let checkpoint: String
        public let reason: String
        /// Was die Schrifterkennung gesehen hat; nil, wenn ein ausdrücklicher Hinweis entschied.
        public let detection: LayaRouter.Detection?
        public let workflow: String?
        public let answers: [(id: String, answer: LayaAnswer)]
    }

    private let sources: [LayaRouter.Checkpoint: Source]
    private let configuration: LayaRuntime.Configuration
    private let fallback: LayaRouter.Checkpoint
    private let autoTaskDetection: Bool
    private var loaded: [LayaRouter.Checkpoint: LayaSystemOne] = [:]
    /// Vorgaben für Anfragen ohne eigenen Hinweis, etwa aus `--checkpoint` und `--lang`.
    private var defaultModel: String?
    private var defaultLanguage: String?

    /// Welche Checkpoints überhaupt bereitstehen.
    nonisolated public var available: [LayaRouter.Checkpoint] { Array(sources.keys) }

    public init(sources: [LayaRouter.Checkpoint: Source],
                configuration: LayaRuntime.Configuration = .init(),
                fallback: LayaRouter.Checkpoint = .english,
                autoTaskDetection: Bool = false) throws {
        guard !sources.isEmpty else {
            throw JevError.invalidRoute("kein einziger Checkpoint angegeben")
        }
        self.sources = sources
        self.configuration = configuration
        // Feste Reihenfolge statt sources.keys.first: Swift mischt die Hash-Reihenfolge je
        // Prozess neu, und dann hinge es vom Neustart ab, welches Modell antwortet.
        let preference: [LayaRouter.Checkpoint] = [fallback, .english, .multilingual, .typedDecisions]
        self.fallback = preference.first { sources[$0] != nil }!
        self.autoTaskDetection = autoTaskDetection
    }

    /// Findet die Modelle nach Namensschema in einem Verzeichnis. Was fehlt, fehlt eben.
    ///
    /// Neben `.mlpackage` auch `.mlmodelc`: Xcode übersetzt ein Paket, das im App-Target liegt,
    /// beim Bauen und legt nur die übersetzte Fassung ins Bundle. Die wird bevorzugt, denn sie
    /// muss beim Start nicht erst übersetzt werden.
    public static func discovering(in directory: URL,
                                   configuration: LayaRuntime.Configuration = .init(),
                                   autoTaskDetection: Bool = false) throws -> LayaRouted {
        let layout: [(LayaRouter.Checkpoint, String, String)] = [
            (.english, "Laya-EN-L512-K512-fp16", "laya-tokenizer-english.json"),
            (.multilingual, "Laya-ML-L1024-K1024-fp16", "laya-tokenizer-multilingual.json"),
            (.typedDecisions, "Laya-TD-L1024-K1024-fp16", "laya-tokenizer-english.json"),
        ]
        var sources: [LayaRouter.Checkpoint: Source] = [:]
        for (checkpoint, model, tokenizer) in layout {
            let tokenizerURL = directory.appendingPathComponent(tokenizer)
            guard FileManager.default.fileExists(atPath: tokenizerURL.path) else { continue }
            for suffix in ["mlmodelc", "mlpackage"] {
                let modelURL = directory.appendingPathComponent(model).appendingPathExtension(suffix)
                if FileManager.default.fileExists(atPath: modelURL.path) {
                    sources[checkpoint] = Source(modelURL: modelURL, tokenizerURL: tokenizerURL)
                    break
                }
            }
        }
        guard !sources.isEmpty else {
            throw JevError.modelFile("in \(directory.path) steht kein laya-Modell")
        }
        return try LayaRouted(sources: sources, configuration: configuration,
                              autoTaskDetection: autoTaskDetection)
    }

    /// Die Modelle aus dem Bundle einer App: Pakete und Tokenizer als Ressourcen im Target.
    public static func bundled(_ bundle: Bundle = .main,
                               configuration: LayaRuntime.Configuration = .init(),
                               autoTaskDetection: Bool = false) throws -> LayaRouted {
        guard let resources = bundle.resourceURL else {
            throw JevError.modelFile("\(bundle.bundlePath) hat kein Ressourcenverzeichnis")
        }
        return try discovering(in: resources, configuration: configuration,
                               autoTaskDetection: autoTaskDetection)
    }

    /// Woher ein Checkpoint geladen wird, falls er bereitsteht.
    nonisolated public func source(of checkpoint: LayaRouter.Checkpoint) -> Source? {
        sources[checkpoint]
    }

    /// Legt fest, was gilt, wenn eine Anfrage weder Modell noch Sprache nennt.
    public func setDefaults(model: String?, language: String?) throws {
        if let model, LayaRouter.Checkpoint.named(model) == nil {
            throw JevError.invalidRoute("unbekanntes Modell \(model)")
        }
        defaultModel = model
        defaultLanguage = language
    }

    public func engine(_ checkpoint: LayaRouter.Checkpoint) throws -> LayaSystemOne {
        if let ready = loaded[checkpoint] { return ready }
        guard let source = sources[checkpoint] else {
            throw JevError.modelFile("Checkpoint \(checkpoint.name) steht nicht bereit")
        }
        let system = try LayaSystemOne(modelURL: source.modelURL,
                                       tokenizerURL: source.tokenizerURL,
                                       configuration: configuration)
        // Das Paket muss der Checkpoint sein, unter dem es geführt wird. Sonst meldet der
        // Router "english", während das mehrsprachige Modell gerechnet hat.
        let actual = system.runtime.metadata.checkpoint
        if actual != "unbekannt", actual != checkpoint.name {
            throw JevError.modelFile(
                "\(source.modelURL.lastPathComponent) ist laut Metadaten \(actual), nicht \(checkpoint.name)")
        }
        loaded[checkpoint] = system
        return system
    }

    /// Entscheidet und rechnet. Ist der gewählte Checkpoint nicht da, wird das gesagt und auf
    /// den vorhandenen ausgewichen, statt stillschweigend das falsche Modell zu nehmen.
    public func answer(state: JevValue,
                       questions: [(id: String, question: JevQuestion)],
                       model: String? = nil,
                       task: String? = nil,
                       language: String? = nil) async throws -> Routed {
        let decision = try route(state: state, questionIDs: questions.map(\.id),
                                 model: model, task: task, language: language)
        let system = try engine(decision.checkpoint)
        return Routed(checkpoint: decision.checkpoint.name, reason: decision.reason,
                      detection: decision.detection, workflow: decision.workflow,
                      answers: try await system.ask(questions, about: state))
    }

    /// Die Wahl des Checkpoints samt Begründung, ohne zu rechnen. Hinweise der Anfrage gehen
    /// den Vorgaben aus `setDefaults` vor; eine Anfrage mit eigener Sprache erbt kein
    /// vorgegebenes Modell, sonst bliebe ihr Hinweis wirkungslos.
    func route(state: JevValue, questionIDs: [String], model: String? = nil,
               task: String? = nil, language: String? = nil) throws -> LayaRouter.Decision {
        let hinted = model != nil || task != nil || language != nil
        let decision = try LayaRouter.decide(state: state, questionIDs: questionIDs,
                                             model: hinted ? model : defaultModel,
                                             task: task,
                                             language: hinted ? language : defaultLanguage,
                                             autoTaskDetection: autoTaskDetection,
                                             fallback: fallback)
        guard sources[decision.checkpoint] == nil else { return decision }
        return LayaRouter.Decision(
            checkpoint: fallback,
            reason: decision.reason + "; \(decision.checkpoint.name) steht hier nicht bereit, also \(fallback.name)",
            detection: decision.detection, workflow: decision.workflow)
    }

    /// Die bereitstehenden Checkpoints in fester Reihenfolge.
    var availableInOrder: [LayaRouter.Checkpoint] {
        LayaRouter.Checkpoint.allCases.filter { sources[$0] != nil }
    }

    func modelName(of checkpoint: LayaRouter.Checkpoint) -> String {
        sources[checkpoint]?.modelURL.deletingPathExtension().lastPathComponent ?? checkpoint.name
    }
}
