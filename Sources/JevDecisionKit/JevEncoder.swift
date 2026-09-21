import Foundation

/// Das Ergebnis der Prompt-Kodierung. Eine oder mehrere Fragen auf demselben Zustand.
public struct JevEncoding: Sendable, Equatable {
    /// Ein Fragezweig: wo der Pointer-Head liest und wo die Optionen enden.
    public struct Question: Sendable, Equatable {
        public let decideIndex: Int
        public let optionEnds: [Int]
    }

    /// Token-IDs der Sequenz, ungepolstert.
    public let ids: [Int32]
    /// Positions-IDs. Nach dem Zustand startet jeder Fragezweig wieder bei `stateTokens`.
    public let positionIDs: [Int32]
    /// 0 für den Zustand, k für Frage k. Grundlage der block-kausalen Maske.
    public let segmentIDs: [Int32]
    /// Zahl der Token des Zustandsblocks, einschließlich des Delimiters.
    public let stateTokens: Int
    public let questions: [Question]
    /// Wurde der Zustand gekürzt, weil er das Budget überschritten hat?
    public let stateTruncated: Bool

    public var count: Int { ids.count }

    // Bequemlichkeit für den Fall einer einzigen Frage.
    public var decideIndex: Int { questions[0].decideIndex }
    public var optionEnds: [Int] { questions[0].optionEnds }
}

/// Die Budgets, mit denen kev trainiert wurde, gegen die, mit denen es serviert wird.
public enum KevBudget {
    /// `kev.model.MAX_STATE`: so weit reicht das Training.
    public static let trainedState = 384
    /// `kev.model.MAX_BRANCH`.
    public static let trainedBranch = 1024
    /// `kev.serve.INFER_MAX_STATE` und `INFER_MAX_BRANCH`. Darüber entscheidet die Sequenzlänge.
    public static let state = 8192
    public static let branch = 8192
}

/// Baut aus Zustand, Anweisungen und Optionen die Tokenfolge, die Kev erwartet.
///
/// Spiegelt `kev.model.encode` ohne Optionsisolation. Aufbau:
/// `<state> Zustand` gefolgt von je Frage `<q> Anweisung <opt> Option </opt> ... <decide>`.
/// Die fünf Delimiter sind selten benutzte Qwen-Sondertoken; LoRA hat ihnen die Bedeutung gegeben.
public struct JevEncoder: Sendable {
    public static let delimiterTokens = [
        "<|fim_prefix|>",   // <state>
        "<|fim_middle|>",   // <q>
        "<|box_start|>",    // <opt>
        "<|box_end|>",      // </opt>
        "<|fim_suffix|>",   // <decide>
    ]

    /// Eine Frage, so wie der Encoder sie sieht: fertig gerenderter Text.
    public struct Prompt: Sendable, Equatable {
        public let instructions: String
        public let options: [String]

        public init(instructions: String, options: [String]) {
            self.instructions = instructions
            self.options = options
        }
    }

    public let tokenizer: Qwen3Tokenizer
    public let maxStateTokens: Int
    public let maxBranchTokens: Int
    /// true: ein zu langer Zustand ist ein Fehler. false: er wird gekürzt.
    public let strictState: Bool

    private let stateID: Int32
    private let questionID: Int32
    private let optionID: Int32
    private let optionEndID: Int32
    private let decideID: Int32

    /// Budget beim Servieren. kev trainierte mit 384 Zustands- und 1024 Zweigtoken;
    /// `kev.serve` hebt beides zur Inferenz auf 8192 an, weil die echte Grenze das
    /// Kontextfenster des Modells ist. Diese Werte folgen dem, die harte Grenze setzt
    /// dann die exportierte Sequenzlänge.
    public static let inferenceStateBudget = 8192
    public static let inferenceBranchBudget = 8192

    public init(tokenizer: Qwen3Tokenizer,
                maxStateTokens: Int = KevBudget.state,
                maxBranchTokens: Int = KevBudget.branch,
                strictState: Bool = true) throws {
        self.tokenizer = tokenizer
        self.maxStateTokens = maxStateTokens
        self.maxBranchTokens = maxBranchTokens
        self.strictState = strictState
        var ids: [Int32] = []
        for token in JevEncoder.delimiterTokens {
            guard let id = tokenizer.id(forToken: token) else { throw JevError.missingDelimiter(token) }
            ids.append(id)
        }
        (stateID, questionID, optionID, optionEndID, decideID) = (ids[0], ids[1], ids[2], ids[3], ids[4])
    }

    public func encode(state: String, instructions: String, options: [String]) throws -> JevEncoding {
        try encode(state: state, questions: [Prompt(instructions: instructions, options: options)])
    }

    public func encode(state: String, questions: [Prompt]) throws -> JevEncoding {
        guard !questions.isEmpty else { throw JevError.noOptions }
        for prompt in questions where prompt.options.isEmpty { throw JevError.noOptions }

        var stateTokens = tokenizer.encodeUserText(state)
        let truncated = stateTokens.count + 1 > maxStateTokens
        if truncated {
            if strictState { throw JevError.stateTooLong(tokens: stateTokens.count + 1, limit: maxStateTokens) }
            stateTokens = Array(stateTokens.prefix(maxStateTokens - 1))
        }

        var ids: [Int32] = [stateID]
        ids.append(contentsOf: stateTokens)
        let stateCount = ids.count
        var segments = [Int32](repeating: 0, count: stateCount)
        var positions = (0 ..< stateCount).map(Int32.init)
        var encoded: [JevEncoding.Question] = []

        for (index, prompt) in questions.enumerated() {
            var branch: [Int32] = [questionID]
            branch.append(contentsOf: tokenizer.encodeUserText(prompt.instructions))
            var ends: [Int] = []
            var cursor = branch.count
            for option in prompt.options {
                let span = [optionID] + tokenizer.encodeUserText(option) + [optionEndID]
                branch.append(contentsOf: span)
                cursor += span.count
                ends.append(cursor - 1)
            }
            branch.append(decideID)

            guard branch.count <= maxBranchTokens - stateCount else {
                throw JevError.branchTooLong(tokens: branch.count, limit: maxBranchTokens - stateCount)
            }

            let base = ids.count
            ids.append(contentsOf: branch)
            segments.append(contentsOf: [Int32](repeating: Int32(index + 1), count: branch.count))
            // Die Positionen jedes Zweigs beginnen wieder direkt hinter dem Zustand. Zusammen mit
            // der block-kausalen Maske macht das die Fragen voneinander unabhängig.
            positions.append(contentsOf: (0 ..< branch.count).map { Int32(stateCount + $0) })
            encoded.append(JevEncoding.Question(decideIndex: base + branch.count - 1,
                                                optionEnds: ends.map { base + $0 }))
        }

        return JevEncoding(ids: ids, positionIDs: positions, segmentIDs: segments,
                           stateTokens: stateCount, questions: encoded, stateTruncated: truncated)
    }
}
