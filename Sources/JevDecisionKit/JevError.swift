import Foundation

public enum JevError: Error, CustomStringConvertible {
    case tokenizerFile(String)
    case invalidJSON(String)
    case missingDelimiter(String)
    case stateTooLong(tokens: Int, limit: Int)
    case branchTooLong(tokens: Int, limit: Int)
    case sequenceTooLong(tokens: Int, limit: Int)
    case tooManyOptions(count: Int, limit: Int)
    /// Die Optionen passen nicht in die Sequenz, egal wie groß der Export ist. laya bricht in
    /// diesem Fall ebenso ab.
    case optionsDoNotFit(count: Int, fitting: Int, length: Int)
    case tooManyQuestions(count: Int, limit: Int)
    case noOptions
    case multipleQuestions(Int)
    case modelOutput(String)
    case modelFile(String)
    case invalidRoute(String)

    public var description: String {
        switch self {
        case let .tokenizerFile(m): return "Tokenizer-Datei: \(m)"
        case let .invalidJSON(m): return "JSON: \(m)"
        case let .missingDelimiter(t): return "Delimiter \(t) fehlt im Vokabular"
        case let .stateTooLong(n, l): return "Zustand hat \(n) Token, erlaubt sind \(l)"
        case let .branchTooLong(n, l): return "Frage samt Optionen hat \(n) Token, erlaubt sind \(l)"
        case let .sequenceTooLong(n, l): return "Sequenz hat \(n) Token, das Modell wurde für \(l) exportiert"
        case let .tooManyOptions(n, l): return "\(n) Optionen, das Modell wurde für \(l) exportiert"
        case let .optionsDoNotFit(n, k, l):
            return "\(n) Optionen, davon passen nur \(k) in die Sequenz von \(l) Token"
        case let .tooManyQuestions(n, l): return "\(n) Fragen, das Modell wurde für \(l) exportiert"
        case .noOptions: return "eine Frage braucht mindestens eine Option"
        case let .multipleQuestions(n): return "Phase 1 beantwortet eine Frage pro Inferenz, angefragt waren \(n)"
        case let .modelOutput(m): return "Modellausgabe: \(m)"
        case let .modelFile(m): return "Modelldatei: \(m)"
        case let .invalidRoute(m): return "Routing: \(m)"
        }
    }
}
