import Foundation
import XCTest
@testable import JevDecisionKit

/// Zugriff auf die Referenzdaten des laya-Ports unter JevCoreML/Golden/laya.
enum LayaFixtures {
    static var golden: URL {
        Fixtures.golden.appendingPathComponent("laya").appendingPathComponent(checkpoint)
    }

    /// Welcher Checkpoint geprüft wird. LAYA_CHECKPOINT überschreibt die Vorgabe.
    static var checkpoint: String {
        ProcessInfo.processInfo.environment["LAYA_CHECKPOINT"] ?? "english"
    }

    /// Die beiden ModernBERT-Ableger teilen sich einen Tokenizer, mmBERT hat einen eigenen.
    static var tokenizerURL: URL {
        let name = checkpoint == "multilingual" ? "multilingual" : "english"
        return Fixtures.models.appendingPathComponent("laya-tokenizer-\(name).json")
    }

    static let modelNames = [
        "english": ["Laya-EN-L512-K512-fp16.mlpackage", "Laya-EN-L512-K512-fp32.mlpackage"],
        "multilingual": ["Laya-ML-L1024-K1024-fp16.mlpackage"],
        "typed-decisions": ["Laya-TD-L1024-K1024-fp16.mlpackage"],
    ]

    static func modelURL() -> URL? {
        if let path = ProcessInfo.processInfo.environment["LAYA_MODEL"] {
            return URL(fileURLWithPath: path)
        }
        for name in modelNames[checkpoint] ?? [] {
            let url = Fixtures.models.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }

    static func value(_ name: String) throws -> JevValue {
        try JevValue.parse(json: try Data(contentsOf: golden.appendingPathComponent(name)))
    }

    static func hasGolden() -> Bool {
        FileManager.default.fileExists(atPath: golden.appendingPathComponent("records.json").path)
    }

    static func sharedTokenizer() throws -> any TextTokenizer {
        if let cached = cachedTokenizer, cachedFor == checkpoint { return cached }
        let tokenizer = try LayaRuntime.tokenizer(at: tokenizerURL)
        cachedTokenizer = tokenizer
        cachedFor = checkpoint
        return tokenizer
    }

    nonisolated(unsafe) private static var cachedTokenizer: (any TextTokenizer)?
    nonisolated(unsafe) private static var cachedFor: String?

    struct Runtime {
        let maxLength: Int
        let headMaxLength: Int
        let clsID: Int32
        let sepID: Int32
        let maskID: Int32
        let padID: Int32
        let maskToken: String
        let temperature: [Double]
        let temperatureByOptions: [String: Double]
    }

    static func runtime() throws -> Runtime {
        let root = try value("runtime.json")
        func table(_ key: String) -> [String: Double] {
            guard case let .object(pairs)? = Fixtures.field(root, key) else { return [:] }
            var out: [String: Double] = [:]
            for pair in pairs { out[pair.key] = Fixtures.double(pair.value) }
            return out
        }
        return Runtime(
            maxLength: Fixtures.int(Fixtures.field(root, "max_len")) ?? 512,
            headMaxLength: Fixtures.int(Fixtures.field(root, "head_max_len")) ?? 192,
            clsID: Int32(Fixtures.int(Fixtures.field(root, "cls_id")) ?? 0),
            sepID: Int32(Fixtures.int(Fixtures.field(root, "sep_id")) ?? 0),
            maskID: Int32(Fixtures.int(Fixtures.field(root, "mask_id")) ?? 0),
            padID: Int32(Fixtures.int(Fixtures.field(root, "pad_id")) ?? 0),
            maskToken: Fixtures.string(Fixtures.field(root, "mask_token")) ?? "[MASK]",
            temperature: Fixtures.doubles(Fixtures.field(root, "temperature")),
            temperatureByOptions: table("temperature_by_options"))
    }

    struct Record {
        let questionID: String
        let qtype: Int32
        let ids: [Int32]
        let markers: [Int32]
        let logits: [Double]
        let actLogits: [Double]
        let instructions: String
        let options: [String]
        let state: JevValue
        let question: JevQuestion
        let answer: JevValue
    }

    static func records() throws -> [Record] {
        let root = try value("records.json")
        return try Fixtures.array(Fixtures.field(root, "records")).map { entry in
            let request = Fixtures.field(entry, "request")!
            // .max: laya kennt kevs Grenze von 255 Optionen nicht, typed-decisions nimmt 340.
            let parsed = try JevRequest.parse(request, maxOptions: .max)
            guard let first = parsed.questions.first else {
                throw JevError.noOptions
            }
            let encoding = Fixtures.field(entry, "encoding") ?? .null
            let rendered = Fixtures.field(entry, "rendered") ?? .null
            return Record(
                questionID: Fixtures.string(Fixtures.field(entry, "question_id")) ?? "",
                qtype: Int32(Fixtures.int(Fixtures.field(entry, "qtype")) ?? 0),
                ids: Fixtures.ints(Fixtures.field(encoding, "ids")).map(Int32.init),
                markers: Fixtures.ints(Fixtures.field(encoding, "markers")).map(Int32.init),
                logits: Fixtures.doubles(Fixtures.field(entry, "logits")),
                actLogits: Fixtures.doubles(Fixtures.field(entry, "act_logits")),
                instructions: Fixtures.string(Fixtures.field(rendered, "instructions")) ?? "",
                options: Fixtures.strings(Fixtures.field(rendered, "options")),
                state: parsed.state,
                question: first.question,
                answer: Fixtures.field(entry, "answer") ?? .null)
        }
    }

    /// Die Anfrage, an der laya abbricht, weil nicht alle Optionen vor max_len passen.
    struct Overflow {
        let state: JevValue
        let question: JevQuestion
        let options: Int
        let fitting: Int
    }

    static func overflow() throws -> Overflow? {
        guard let entry = Fixtures.field(try value("records.json"), "overflow") else { return nil }
        let parsed = try JevRequest.parse(Fixtures.field(entry, "request") ?? .null, maxOptions: .max)
        guard let first = parsed.questions.first else { throw JevError.noOptions }
        return Overflow(state: parsed.state, question: first.question,
                        options: Fixtures.int(Fixtures.field(entry, "options")) ?? -1,
                        fitting: Fixtures.int(Fixtures.field(entry, "fitting")) ?? -1)
    }

    struct TokenizerCase {
        let text: String
        let ids: [Int32]
    }

    static func hasFuzz() -> Bool {
        FileManager.default.fileExists(atPath: golden.appendingPathComponent("tokenizer-fuzz.json").path)
    }

    static func fuzzCases() throws -> (traps: Int, cases: [TokenizerCase]) {
        let root = try value("tokenizer-fuzz.json")
        let cases = Fixtures.array(Fixtures.field(root, "cases")).map { entry in
            TokenizerCase(text: Fixtures.string(Fixtures.field(entry, "text")) ?? "",
                          ids: Fixtures.ints(Fixtures.field(entry, "ids")).map(Int32.init))
        }
        return (Fixtures.int(Fixtures.field(root, "vocabulary_traps")) ?? 0, cases)
    }

    static func tokenizerCases() throws -> [TokenizerCase] {
        let root = try value("tokenizer.json")
        return Fixtures.array(Fixtures.field(root, "cases")).map { entry in
            TokenizerCase(text: Fixtures.string(Fixtures.field(entry, "text")) ?? "",
                          ids: Fixtures.ints(Fixtures.field(entry, "ids")).map(Int32.init))
        }
    }
}
