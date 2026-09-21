import Foundation

/// Was Server und Kommandozeile von einer kev-Entscheidungsmaschine brauchen.
///
/// Zwei Dinge erfüllen das: ein einzelner Export (`SystemOne`) und mehrere Exporte mit
/// verschiedenen Shape-Budgets (`SystemOnePool`). Solange der Server nur `SystemOne` annahm,
/// ließen sich die Buckets zwar bauen und testen, aber nicht betreiben.
public protocol SystemOneEngine: Sendable {
    func answer(_ request: JevRequest) async throws -> SystemOneResponse
    /// Token der Antwort, wie `usage.output_tokens` sie ausweist.
    func outputTokens(of answers: [(id: String, answer: JevAnswer)]) async -> Int
    /// Prüft vorab, ob die Anfrage in ein Budget passt. Eine zu große Anfrage soll 422 ergeben
    /// und den Grund nennen, nicht erst im Modell scheitern.
    func check(_ request: JevRequest) async throws
    /// Einmal jeden Export durchlaufen, damit die erste echte Anfrage keine Kernel kompiliert.
    @discardableResult
    func warmUp() async throws -> Double
    /// Die Exporte, aus denen die Maschine besteht, kleinstes Budget zuerst.
    func members() async -> [EngineMember]
}

public struct EngineMember: Sendable, Equatable {
    public let file: String
    public let contract: JevRuntime.Contract
    public let sequenceLength: Int
    public let maxQuestions: Int
    public let maxOptions: Int
}

extension SystemOne: SystemOneEngine {
    public func check(_ request: JevRequest) async throws {
        guard !request.questions.isEmpty else { throw JevError.invalidJSON("questions ist leer") }
        // Ein Phase-1-Export beantwortet eine Frage je Durchlauf, legt mehrere aber nacheinander
        // vor. Eine Ablehnung wäre eine Einschränkung, die die Laufzeit gar nicht hat.
        if runtime.contract == .fanOut, request.questions.count > runtime.maxQuestions {
            throw JevError.tooManyQuestions(count: request.questions.count, limit: runtime.maxQuestions)
        }
        for (_, question) in request.questions {
            let rendered = try SystemOne.renderQuestion(question)
            if rendered.options.count > runtime.maxOptions {
                throw JevError.tooManyOptions(count: rendered.options.count, limit: runtime.maxOptions)
            }
        }
    }

    @discardableResult
    public func warmUp() async throws -> Double {
        try await runtime.warmUp()
    }

    public func members() async -> [EngineMember] {
        [EngineMember(file: runtime.modelName, contract: runtime.contract,
                      sequenceLength: runtime.sequenceLength,
                      maxQuestions: runtime.maxQuestions, maxOptions: runtime.maxOptions)]
    }
}
