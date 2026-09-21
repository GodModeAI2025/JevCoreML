import Foundation
import JevDecisionKit
import Observation

/// Eine Antwort, fertig zum Anzeigen.
struct AnswerRow: Identifiable, Sendable {
    let id: String
    let kind: String
    let headline: String
    let confidence: Double?
    /// Nur bei laya: wie wahrscheinlich eine Handlung angebracht ist.
    let actProbability: Double?
    let bars: [(label: String, value: Double)]
}

struct DemoResult: Sendable {
    let engine: String
    let detail: String
    /// Median aus drei Läufen nach dem ersten.
    var milliseconds: Double
    /// Der erste Lauf, falls er Laden oder das Vorbereiten einer neuen Länge enthielt.
    var firstRunMilliseconds: Double?
    let inputTokens: Int
    let answers: [AnswerRow]
}

/// Findet die Modelle, lädt sie beim ersten Gebrauch und rechnet.
@MainActor
@Observable
final class DemoModel {
    var engine: Engine = .laya
    var questionSet: QuestionSet = .triage {
        didSet { if oldValue != questionSet, let first = questionSet.samples.first { text = first.text } }
    }
    var text = QuestionSet.triage.samples[0].text
    var subject = "Rechnung 4411 doppelt abgebucht"
    var customInstruction = "An welches Team geht das Ticket?"
    var customOptions = "billing: Rechnungen, Zahlungen, Erstattungen\ntech: Fehler, Abstürze, Ausfälle\nsales: Preise, Verträge, Angebote"
    var modelsDirectory: URL?
    var isRunning = false
    var status = ""
    var result: DemoResult?
    var errorMessage: String?

    private var laya: LayaRouted?
    private var kev: SystemOne?
    private var loadedFrom: URL?

    init() {
        modelsDirectory = Self.defaultModelsDirectory()
        status = modelsDirectory == nil ? "Kein Modellordner gefunden" : "Bereit"
    }

    /// Wo die Modelle liegen, in dieser Reihenfolge: zuletzt gewählter Ordner, JEV_MODELS, das
    /// App-Bundle, der Ordner `Models` im Repo, aus dem die App gebaut wurde.
    static func defaultModelsDirectory() -> URL? {
        let fm = FileManager.default
        func hasModels(_ url: URL) -> Bool {
            (try? fm.contentsOfDirectory(atPath: url.path))?.contains { $0.hasPrefix("Laya-") || $0.hasPrefix("Kev06B") } ?? false
        }
        var candidates: [URL] = []
        if let saved = UserDefaults.standard.string(forKey: "modelsPath") {
            candidates.append(URL(fileURLWithPath: saved))
        }
        if let env = ProcessInfo.processInfo.environment["JEV_MODELS"] {
            candidates.append(URL(fileURLWithPath: env))
        }
        if let resources = Bundle.main.resourceURL { candidates.append(resources) }
        // Demo/JevDemo/DemoModel.swift -> Models im Repo
        candidates.append(URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Models"))
        return candidates.first(where: hasModels)
    }

    func choose(directory: URL) {
        modelsDirectory = directory
        UserDefaults.standard.set(directory.path, forKey: "modelsPath")
        laya = nil
        kev = nil
        loadedFrom = nil
        status = "Bereit"
    }

    func run() async {
        guard let directory = modelsDirectory else {
            errorMessage = "Erst einen Modellordner wählen. Die Modelle holt scripts/fetch-models.sh."
            return
        }
        isRunning = true
        errorMessage = nil
        defer { isRunning = false }
        let state = questionSet.state(text: text, subject: subject)
        let questions = questionSet.questions(customInstruction: customInstruction,
                                              customOptions: customOptions)
        do {
            if loadedFrom != directory { laya = nil; kev = nil; loadedFrom = directory }
            // Viermal rechnen: der erste Lauf kann Laden und Vorbereiten enthalten, berichtet wird
            // der Median der drei danach. Jeder Lauf liefert dieselben Antworten.
            var runs: [DemoResult] = []
            for _ in 0 ..< 4 {
                switch engine {
                case .laya: runs.append(try await runLaya(in: directory, state: state, questions: questions))
                case .kev: runs.append(try await runKev(in: directory, state: state, questions: questions))
                }
            }
            var final = runs[runs.count - 1]
            let warm = runs.dropFirst().map(\.milliseconds).sorted()
            final.milliseconds = warm[warm.count / 2]
            final.firstRunMilliseconds = runs[0].milliseconds > 2 * final.milliseconds ? runs[0].milliseconds : nil
            result = final
            status = "Fertig"
        } catch {
            errorMessage = String(describing: error)
            status = "Fehler"
        }
    }

    private func runLaya(in directory: URL, state: JevValue,
                         questions: [(id: String, question: JevQuestion)]) async throws -> DemoResult {
        if laya == nil {
            status = "laya wird geladen …"
            laya = try LayaRouted.discovering(in: directory)
        }
        guard let laya else { throw JevError.modelFile("laya nicht geladen") }
        status = "Rechnet …"
        let clock = ContinuousClock()
        let started = clock.now
        let routed = try await laya.answer(state: state, questions: questions)
        let elapsed = clock.now - started
        return DemoResult(
            engine: "laya, Checkpoint \(routed.checkpoint)",
            detail: routed.reason,
            milliseconds: Self.milliseconds(elapsed),
            inputTokens: routed.answers.reduce(0) { $0 + $1.answer.inputTokens },
            answers: routed.answers.map { id, answer in
                Self.row(id: id, answer: answer.answer, confidence: answer.confidence,
                         actProbability: answer.actProbability)
            })
    }

    private func runKev(in directory: URL, state: JevValue,
                        questions: [(id: String, question: JevQuestion)]) async throws -> DemoResult {
        if kev == nil {
            // Ein geladener kev-Export kann im Temp-Verzeichnis bis zu 14 GB belegen, bis die App
            // endet. Dieselbe Schwelle wie SystemOnePool, damit die Platte nicht vollläuft.
            let free = (try? FileManager.default.temporaryDirectory
                .resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
                .volumeAvailableCapacityForImportantUsage ?? .max
            guard free >= 16 << 30 else {
                throw JevError.modelFile(String(format: "kev braucht beim Laden bis zu 14 GB temporär, frei sind %.1f GB", Double(free) / Double(1 << 30)))
            }
            status = "kev wird geladen …"
            kev = try SystemOne(modelURL: directory.appendingPathComponent("Kev06B-Q4-fp16.mlpackage"),
                                tokenizerURL: directory.appendingPathComponent("tokenizer.json"))
        }
        guard let kev else { throw JevError.modelFile("kev nicht geladen") }
        status = "Rechnet …"
        // Ein Durchlauf fasst bis zu vier Fragen; mehr werden in Gruppen gestellt.
        let perPass = max(1, kev.runtime.maxQuestions)
        var rows: [AnswerRow] = []
        var tokens = 0
        var passes = 0
        let clock = ContinuousClock()
        let started = clock.now
        for start in stride(from: 0, to: questions.count, by: perPass) {
            let group = Array(questions[start ..< min(start + perPass, questions.count)])
            let response = try await kev.answer(JevRequest(state: state, questions: group))
            tokens += response.usage.inputTokens
            passes += response.usage.passes
            rows += response.answers.map { id, answer in
                Self.row(id: id, answer: answer, confidence: nil, actProbability: nil)
            }
        }
        let elapsed = clock.now - started
        return DemoResult(engine: "kev, Kev06B-Q4-fp16",
                          detail: "\(passes) Durchlauf/Durchläufe für \(questions.count) Fragen",
                          milliseconds: Self.milliseconds(elapsed), inputTokens: tokens, answers: rows)
    }

    static func milliseconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) * 1000 + Double(duration.components.attoseconds) / 1e15
    }

    static func row(id: String, answer: JevAnswer, confidence: Double?, actProbability: Double?) -> AnswerRow {
        switch answer {
        case let .choice(choice):
            return AnswerRow(id: id, kind: "Auswahl", headline: choice.choice,
                             confidence: confidence ?? choice.confidence, actProbability: actProbability,
                             bars: choice.probabilities.map { ($0.name, $0.probability) })
        case let .score(score):
            let level = min(max(Int(score.score.rounded()), 0), max(score.legend.count - 1, 0))
            let label = score.legend.indices.contains(level) ? score.legend[level] : ""
            return AnswerRow(id: id, kind: "Stufe",
                             headline: score.score.formatted(.number.precision(.fractionLength(2))) + " · " + label,
                             confidence: confidence ?? score.confidence, actProbability: actProbability,
                             bars: zip(score.legend, score.probabilities).map { ($0, $1) })
        case let .noul(noul):
            return AnswerRow(id: id, kind: "Ja/Nein", headline: noul.noul >= 0.5 ? "ja" : "nein",
                             confidence: confidence ?? max(noul.noul, 1 - noul.noul),
                             actProbability: actProbability,
                             bars: [("nein", 1 - noul.noul), ("ja", noul.noul)])
        }
    }
}
