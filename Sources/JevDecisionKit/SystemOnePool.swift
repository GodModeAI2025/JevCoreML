import CoreML
import Foundation

/// Mehrere Exporte desselben Modells mit verschiedenen Shape-Budgets, und die Anfrage geht an den
/// kleinsten, in den sie passt.
///
/// Core ML braucht feste Formen, also wird jede Anfrage auf die exportierte Länge gepolstert.
/// Gemessen auf kevs Entwicklungssuite: der Median liegt bei 115 Token, das 90. Perzentil bei 370.
/// Ein einzelner Export mit L=1024 polstert für die Hälfte des Verkehrs also fast das Neunfache.
///
/// Das Encoding hängt nicht vom Modell ab, sondern nur vom Tokenizer und den Budgets. Deshalb
/// wird einmal kodiert und erst danach entschieden, welcher Export rechnet.
public actor SystemOnePool {
    public struct Member: Sendable {
        public let url: URL
        /// Die größte Länge des Exports; ein Universalpaket nimmt darunter weitere an.
        public let sequenceLength: Int
        public let maxQuestions: Int
        public let maxOptions: Int
        public let contract: JevRuntime.Contract
        public var sequenceLengths: [Int] = []
    }

    private struct Entry {
        let runtime: JevRuntime
        let systemOne: SystemOne
        let url: URL
    }

    private let entries: [Entry]
    private let encoder: JevEncoder

    /// Nach aufsteigendem Budget sortiert: das erste passende Mitglied ist das billigste.
    public nonisolated let members: [Member]

    /// Was ein geladener Export zur Laufzeit an Platte belegt, mit Reserve. Binäre Einheiten:
    /// 16 GiB sind 17,2 GB, so wie `df -H` zählt.
    ///
    /// Gemessen an `Kev06B-Q4-fp16`: 225 gelöschte, aber offene `payload-*.bin` im
    /// Temp-Verzeichnis, zusammen 14,3 GB, bei 1,1 GB Gewichten. Core ML legt sie beim Laden an
    /// und gibt sie erst frei, wenn der Prozess endet; `du` sieht sie nicht. Ein Pool aus vier
    /// Exporten hat so die Platte vollgeschrieben, bis nichts mehr ging, auch keine Protokolle.
    public static let scratchPerMember: Int64 = 16 << 30

    /// - Parameter minimumFreeBytesPerMember: Vor jedem Export wird geprüft, ob im
    ///   Temp-Verzeichnis so viel frei ist. 0 schaltet die Prüfung ab.
    public init(modelURLs: [URL],
                tokenizerURL: URL,
                configuration: JevRuntime.Configuration = .init(),
                minimumFreeBytesPerMember: Int64 = SystemOnePool.scratchPerMember) throws {
        guard !modelURLs.isEmpty else { throw JevError.modelFile("keine Modelle angegeben") }
        let tokenizer = try Qwen3Tokenizer(contentsOf: tokenizerURL)
        self.encoder = try JevEncoder(tokenizer: tokenizer,
                                      maxStateTokens: configuration.maxStateTokens,
                                      maxBranchTokens: configuration.maxBranchTokens,
                                      strictState: configuration.strictState)
        var built: [Entry] = []
        for (position, url) in modelURLs.enumerated() {
            // Platz für alle noch folgenden Exporte, nicht nur für den nächsten. Die Payloads
            // entstehen zum Großteil erst bei den ersten Vorhersagen, also nach dem Laden; eine
            // Prüfung je Export sähe jedes Mal genug Platz und füllte die Platte beim Warmlauf.
            // Sättigend multiplizieren: eine sehr große Reserve je Export darf nicht überlaufen,
            // sonst endet schon die Prüfung in einem Trap statt in einer Ablehnung.
            let (product, overflow) = Int64(modelURLs.count - position)
                .multipliedReportingOverflow(by: minimumFreeBytesPerMember)
            let needed = overflow ? Int64.max : product
            if minimumFreeBytesPerMember > 0, let free = SystemOnePool.freeScratchBytes(), free < needed {
                throw JevError.modelFile(String(
                    format: "zu wenig Platz für %d Export(e): im Temp-Verzeichnis sind %.1f GiB frei, "
                        + "gebraucht werden rund %.0f GiB, denn jeder geladene Export belegt dort bis zu "
                        + "14 GiB, bis der Prozess endet. Weniger Buckets nehmen oder Platz schaffen.",
                    modelURLs.count - position, Double(free) / Double(1 << 30),
                    Double(needed) / Double(1 << 30)))
            }
            let runtime = try JevRuntime(modelURL: url, tokenizer: tokenizer, configuration: configuration)
            built.append(Entry(runtime: runtime, systemOne: SystemOne(runtime: runtime), url: url))
        }
        // Kleinste Sequenz zuerst, bei gleicher Länge das Modell mit mehr Fragen und Optionen.
        // Absteigend, nicht aufsteigend: sonst gewänne bei gleicher Länge der Phase-1-Export mit
        // einer Frage je Durchlauf, und jede Anfrage mit mehreren Fragen kostete mehrere Läufe.
        built.sort { a, b in
            SystemOnePool.cheaperFirst((a.runtime.sequenceLength, a.runtime.maxQuestions, a.runtime.maxOptions),
                                       (b.runtime.sequenceLength, b.runtime.maxQuestions, b.runtime.maxOptions))
        }
        self.entries = built
        self.members = built.map {
            Member(url: $0.url,
                   sequenceLength: $0.runtime.sequenceLength,
                   maxQuestions: $0.runtime.maxQuestions,
                   maxOptions: $0.runtime.maxOptions,
                   contract: $0.runtime.contract,
                   sequenceLengths: $0.runtime.sequenceLengths.count > 1 ? $0.runtime.sequenceLengths : [])
        }
    }

    /// Reihenfolge der Exporte: kürzeste Sequenz zuerst, bei gleicher Länge mehr Fragen und dann
    /// mehr Optionen zuerst. Die Tripel sind (Länge, Fragen, Optionen).
    static func cheaperFirst(_ a: (Int, Int, Int), _ b: (Int, Int, Int)) -> Bool {
        if a.0 != b.0 { return a.0 < b.0 }
        if a.1 != b.1 { return a.1 > b.1 }
        return a.2 > b.2
    }

    /// Freier Platz auf dem Datenträger des Temp-Verzeichnisses, dort landen die Payloads.
    static func freeScratchBytes() -> Int64? {
        let values = try? FileManager.default.temporaryDirectory
            .resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    /// Wärmt jeden Export einmal auf. Ohne das zahlt die erste Anfrage je Budget die
    /// Kernel-Kompilierung, und das fällt bei mehreren Exporten mehrfach an.
    @discardableResult
    public func warmUpEach() async throws -> [Double] {
        var times: [Double] = []
        for entry in entries { times.append(try await entry.runtime.warmUp()) }
        return times
    }

    /// Welches Mitglied würde diese Anfrage bekommen? Für Diagnose und Tests.
    public func route(_ request: JevRequest) throws -> Member {
        let plan = try plan(for: request)
        return members[plan.index]
    }

    public func answer(_ request: JevRequest) async throws -> SystemOneResponse {
        let plan = try plan(for: request)
        return try await entries[plan.index].systemOne.answer(request)
    }

    // MARK: - Auswahl

    private func plan(for request: JevRequest) throws -> (index: Int, tokens: Int) {
        guard !request.questions.isEmpty else { throw JevError.invalidJSON("questions ist leer") }
        let prompts = try request.questions
            .map { try SystemOne.renderQuestion($0.question) }
            .map { JevEncoder.Prompt(instructions: $0.instructions, options: $0.options) }
        let encoding = try encoder.encode(state: request.state.render(), questions: prompts)
        let options = prompts.map(\.options.count).max() ?? 0

        for (index, entry) in entries.enumerated() {
            let fitsLength = encoding.count <= entry.runtime.sequenceLength
            let fitsOptions = options <= entry.runtime.maxOptions
            // Ein Phase-1-Export beantwortet eine Frage je Durchlauf, mehrere gehen nacheinander.
            let fitsQuestions = entry.runtime.contract == .singleQuestion
                ? true
                : request.questions.count <= entry.runtime.maxQuestions
            if fitsLength, fitsOptions, fitsQuestions { return (index, encoding.count) }
        }

        let largest = entries[entries.count - 1].runtime
        if encoding.count > largest.sequenceLength {
            throw JevError.sequenceTooLong(tokens: encoding.count, limit: largest.sequenceLength)
        }
        if options > largest.maxOptions {
            throw JevError.tooManyOptions(count: options, limit: largest.maxOptions)
        }
        throw JevError.tooManyQuestions(count: request.questions.count, limit: largest.maxQuestions)
    }
}

extension SystemOnePool: SystemOneEngine {
    /// Summe über alle Exporte.
    @discardableResult
    public func warmUp() async throws -> Double {
        try await warmUpEach().reduce(0, +)
    }

    /// Alle Exporte teilen sich den Tokenizer, also zählt der erste.
    public func outputTokens(of answers: [(id: String, answer: JevAnswer)]) async -> Int {
        await entries[0].systemOne.outputTokens(of: answers)
    }

    /// Dieselbe Auswahl wie beim Beantworten. Passt die Anfrage in keinen Export, nennt der
    /// Fehler das größte Budget.
    public func check(_ request: JevRequest) async throws {
        _ = try plan(for: request)
    }

    public func members() async -> [EngineMember] {
        members.map {
            EngineMember(file: $0.url.deletingPathExtension().lastPathComponent,
                         contract: $0.contract, sequenceLength: $0.sequenceLength,
                         maxQuestions: $0.maxQuestions, maxOptions: $0.maxOptions,
                         sequenceLengths: $0.sequenceLengths)
        }
    }
}
