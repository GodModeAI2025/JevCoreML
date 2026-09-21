import CoreML
import Foundation

// MARK: - Antworten

public struct NoulAnswer: Sendable, Equatable {
    /// Wahrscheinlichkeit für "ja".
    public let noul: Double
    public let probabilities: [Double]
}

public struct ChoiceAnswer: Sendable, Equatable {
    public let choice: String
    /// 0 bei Gleichverteilung, 1 bei voller Sicherheit.
    public let confidence: Double
    public let probabilities: [(name: String, probability: Double)]

    public subscript(name: String) -> Double? {
        probabilities.first { $0.name == name }?.probability
    }

    public static func == (lhs: ChoiceAnswer, rhs: ChoiceAnswer) -> Bool {
        lhs.choice == rhs.choice && lhs.confidence == rhs.confidence
            && lhs.probabilities.count == rhs.probabilities.count
            && zip(lhs.probabilities, rhs.probabilities).allSatisfy { $0.name == $1.name && $0.probability == $1.probability }
    }
}

public struct ScoreAnswer: Sendable, Equatable {
    /// Erwartungswert der Stufe, also eine Zahl zwischen 0 und levels-1.
    public let score: Double
    public let legend: [String]
    public let probabilities: [Double]
    public let confidence: Double
}

public enum JevAnswer: Sendable, Equatable {
    case noul(NoulAnswer)
    case choice(ChoiceAnswer)
    case score(ScoreAnswer)
}

// MARK: - Fragen

public enum JevQuestion: Sendable {
    /// Ja/Nein. Die Kriterien beschreiben optional, was "wahr" und was "falsch" bedeutet.
    case noul(instructions: JevValue, whenTrue: JevValue = .null, whenFalse: JevValue = .null)
    /// Auswahl aus benannten Optionen, Reihenfolge bleibt erhalten.
    case choice(instructions: JevValue, criteria: [(name: String, description: JevValue)])
    /// Geordnete Stufen, die Antwort ist der Erwartungswert.
    case score(instructions: JevValue, levels: [JevValue])
}

public struct JevRequest: Sendable {
    public var state: JevValue
    public var model: String
    public var questions: [(id: String, question: JevQuestion)]
    /// Hinweise für die Wahl des laya-Checkpoints, wie `task` und `lang` in
    /// `laya.router.Router.predict`. kev liest sie nicht.
    public var task: String?
    public var language: String?

    public init(state: JevValue, model: String = "kev-latest",
                questions: [(id: String, question: JevQuestion)],
                task: String? = nil, language: String? = nil) {
        self.state = state
        self.model = model
        self.questions = questions
        self.task = task
        self.language = language
    }
}

/// Verbrauchszahlen einer Anfrage, in der Form, die `kev.serve` und die TypeSafe-API ausweisen.
public struct JevUsage: Sendable, Equatable {
    public let inputTokens: Int
    public let outputTokens: Int
    /// Davon entfallen auf den Zustandsblock. Bei mehreren Fragen zeigt das, was ein
    /// gemeinsamer Durchlauf gegenüber getrennten Aufrufen einspart.
    public let stateTokens: Int
    /// Zahl der Modelldurchläufe. Unter Phase 2 immer 1.
    public let passes: Int
}

public struct SystemOneResponse: Sendable {
    public let model: String
    public let answers: [(id: String, answer: JevAnswer)]
    public let usage: JevUsage
    public let latencyMilliseconds: Double
    /// Fertig gerenderte Antwortobjekte je Frage, für eine Maschine mit eigenem Format. laya
    /// schreibt `action`, eine Konfidenz auch bei noul und vier Nachkommastellen, kev nichts
    /// davon. Leer heißt: das kev-Format aus `KevAnswerJSON`, byte-gleich wie bisher.
    public var renderedAnswers: [String: String] = [:]
    /// Weitere Felder auf oberster Ebene, fertig gerendert, etwa `routing` bei laya.
    public var extraFields: [(key: String, json: String)] = []

    public subscript(id: String) -> JevAnswer? {
        answers.first { $0.id == id }?.answer
    }
}

// MARK: - Schnittstelle

/// Die native Entscheidungsschnittstelle: Zustand hinein, getypte Antwort heraus.
///
/// Keine Textgenerierung, kein Python, kein lokaler Server, keine Cloud. Ein Aufruf ist ein
/// Forward-Pass durch das Core-ML-Modell und ein Softmax über so viele Logits wie es Optionen gibt.
public final class SystemOne: Sendable {
    public let runtime: JevRuntime

    public init(modelURL: URL,
                tokenizerURL: URL,
                configuration: JevRuntime.Configuration = .init()) throws {
        let tokenizer = try Qwen3Tokenizer(contentsOf: tokenizerURL)
        self.runtime = try JevRuntime(modelURL: modelURL, tokenizer: tokenizer, configuration: configuration)
    }

    public init(runtime: JevRuntime) {
        self.runtime = runtime
    }

    // MARK: Bequeme Einstiege

    @discardableResult
    public func noul(state: JevValue,
                     instruction: JevValue,
                     whenTrue: JevValue = .null,
                     whenFalse: JevValue = .null) async throws -> NoulAnswer {
        guard case let .noul(answer) = try await ask(
            .noul(instructions: instruction, whenTrue: whenTrue, whenFalse: whenFalse), about: state)
        else { throw JevError.modelOutput("unerwarteter Antworttyp") }
        return answer
    }

    @discardableResult
    public func choice(state: JevValue,
                       instruction: JevValue,
                       options: [String]) async throws -> ChoiceAnswer {
        try await choice(state: state, instruction: instruction,
                         options: options.map { (name: $0, description: JevValue.null) })
    }

    @discardableResult
    public func choice(state: JevValue,
                       instruction: JevValue,
                       options: [(name: String, description: JevValue)]) async throws -> ChoiceAnswer {
        guard case let .choice(answer) = try await ask(
            .choice(instructions: instruction, criteria: options), about: state)
        else { throw JevError.modelOutput("unerwarteter Antworttyp") }
        return answer
    }

    @discardableResult
    public func score(state: JevValue,
                      instruction: JevValue,
                      levels: [JevValue]) async throws -> ScoreAnswer {
        guard case let .score(answer) = try await ask(
            .score(instructions: instruction, levels: levels), about: state)
        else { throw JevError.modelOutput("unerwarteter Antworttyp") }
        return answer
    }

    // MARK: Kern

    public func ask(_ question: JevQuestion, about state: JevValue) async throws -> JevAnswer {
        let response = try await answer(JevRequest(state: state, questions: [(id: "q", question: question)]))
        return response.answers[0].answer
    }

    /// Beantwortet eine vollständige SystemOne-Anfrage.
    ///
    /// Unter Phase 2 gehen alle Fragen in einem Durchlauf durch den Backbone, so wie Jevs
    /// Speculative Fan-Out sie in einem Aufruf beantwortet. Unter Phase 1 läuft jede Frage
    /// einzeln; das Ergebnis ist dasselbe, weil Kevs block-kausale Maske die Zweige ohnehin
    /// voneinander trennt, es kostet nur je Frage einen weiteren Durchlauf.
    public func answer(_ request: JevRequest) async throws -> SystemOneResponse {
        guard !request.questions.isEmpty else { throw JevError.invalidJSON("questions ist leer") }
        let prompts = try request.questions.map { try SystemOne.renderQuestion($0.question) }
            .map { JevEncoder.Prompt(instructions: $0.instructions, options: $0.options) }

        let result = try await runtime.run(state: request.state.render(), questions: prompts)

        var answers: [(id: String, answer: JevAnswer)] = []
        for (index, entry) in request.questions.enumerated() {
            let p = result.probabilities[index]
            switch entry.question {
            case .noul:
                answers.append((entry.id, .noul(NoulAnswer(noul: p[1], probabilities: p))))
            case let .choice(_, criteria):
                let best = SystemOne.argmax(p)
                answers.append((entry.id, .choice(ChoiceAnswer(
                    choice: criteria[best].name,
                    confidence: SystemOne.choiceConfidence(p),
                    probabilities: zip(criteria.map(\.name), p).map { (name: $0, probability: $1) }))))
            case .score:
                let expected = p.enumerated().reduce(0.0) { $0 + Double($1.offset) * $1.element }
                answers.append((entry.id, .score(ScoreAnswer(score: expected,
                                                             legend: prompts[index].options,
                                                             probabilities: p,
                                                             confidence: SystemOne.scoreConfidence(p)))))
            }
        }

        let outputTokens = await tokenCount(of: answers)
        return SystemOneResponse(
            model: request.model,
            answers: answers,
            usage: JevUsage(inputTokens: result.encoding.count,
                            outputTokens: outputTokens,
                            stateTokens: result.encoding.stateTokens,
                            passes: result.passes),
            latencyMilliseconds: result.latencyMilliseconds)
    }

    /// Billing-artige Zahl wie `kev.api.output_tokens`: Token der serialisierten Antwort.
    /// Es wird nichts generiert, die Zahl beschreibt nur die Größe des Ergebnisses.
    public func outputTokens(of answers: [(id: String, answer: JevAnswer)]) async -> Int {
        await tokenCount(of: answers)
    }

    private func tokenCount(of answers: [(id: String, answer: JevAnswer)]) async -> Int {
        let text = KevAnswerJSON.text(answers, pretty: false)
        return await runtime.encoder.tokenizer.encode(text).count
    }

    /// Anweisung und Optionstexte einer Frage, genau wie `kev.api.to_record` sie erzeugt.
    /// Als eigener Schritt herausgezogen, damit der Paritätstest ihn ohne Modell prüfen kann.
    static func renderQuestion(_ question: JevQuestion) throws -> (instructions: String, options: [String]) {
        switch question {
        case let .noul(instructions, whenTrue, whenFalse):
            return (instructions.render(),
                    [optionText("no", whenFalse), optionText("yes", whenTrue)])
        case let .choice(instructions, criteria):
            guard !criteria.isEmpty else { throw JevError.noOptions }
            return (instructions.render(), criteria.map { optionText($0.name, $0.description) })
        case let .score(instructions, levels):
            guard levels.count >= 2 else { throw JevError.noOptions }
            return (instructions.render(), levels.map { $0.render() })
        }
    }

    // MARK: Hilfsfunktionen, identisch zu kev/api.py

    static func optionText(_ name: String, _ description: JevValue) -> String {
        switch description {
        case .null: return name
        case let .string(s) where s.isEmpty: return name
        default: return "\(name): \(description.render())"
        }
    }

    /// Index des grössten Wertes, bei Gleichstand der erste.
    ///
    /// Swifts `max(by:)` liefert bei Gleichstand den letzten Treffer, Pythons `max` den ersten.
    /// Bei exakt gleichen Wahrscheinlichkeiten wäre das eine andere Antwort als die der Referenz.
    static func argmax(_ p: [Double]) -> Int {
        var best = 0
        for i in p.indices where p[i] > p[best] { best = i }
        return best
    }

    static func choiceConfidence(_ p: [Double]) -> Double {
        let k = Double(p.count)
        guard p.count > 1, let best = p.max() else { return 1.0 }
        return (best - 1 / k) / (1 - 1 / k)
    }

    static func scoreConfidence(_ p: [Double]) -> Double {
        let l = p.count
        guard l > 1 else { return 1.0 }
        let mode = argmax(p)
        let spread = p.enumerated().reduce(0.0) { $0 + $1.element * Double(abs($1.offset - mode)) }
        return 1.0 - spread / Double(l - 1)
    }
}
