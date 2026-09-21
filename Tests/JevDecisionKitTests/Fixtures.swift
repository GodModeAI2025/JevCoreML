import Foundation
import XCTest
@testable import JevDecisionKit

/// Zugriff auf die aus PyTorch gezogenen Referenzdaten unter JevCoreML/Golden.
enum Fixtures {
    /// Tests/JevDecisionKitTests/Fixtures.swift -> Wurzel des Repos
    static let root: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0 ..< 3 { url = url.deletingLastPathComponent() }
        return url
    }()

    static var golden: URL { root.appendingPathComponent("Golden") }
    static var models: URL { root.appendingPathComponent("Models") }
    static var tokenizerURL: URL { models.appendingPathComponent("tokenizer.json") }

    static func value(_ name: String) throws -> JevValue {
        try JevValue.parse(json: try Data(contentsOf: golden.appendingPathComponent(name)))
    }

    static func field(_ value: JevValue, _ key: String) -> JevValue? {
        guard case let .object(pairs) = value else { return nil }
        return pairs.first { $0.key == key }?.value
    }

    static func string(_ value: JevValue?) -> String? {
        guard case let .string(s)? = value else { return nil }
        return s
    }

    static func int(_ value: JevValue?) -> Int? {
        switch value {
        case let .int(i)?: return i
        case let .double(d)?: return Int(d)
        default: return nil
        }
    }

    static func double(_ value: JevValue?) -> Double? {
        switch value {
        case let .int(i)?: return Double(i)
        case let .double(d)?: return d
        default: return nil
        }
    }

    static func array(_ value: JevValue?) -> [JevValue] {
        guard case let .array(items)? = value else { return [] }
        return items
    }

    static func ints(_ value: JevValue?) -> [Int] {
        array(value).compactMap { int($0) }
    }

    static func doubles(_ value: JevValue?) -> [Double] {
        array(value).compactMap { double($0) }
    }

    static func strings(_ value: JevValue?) -> [String] {
        array(value).compactMap { string($0) }
    }

    struct TokenizerCase {
        let text: String
        let ids: [Int32]
        let userIDs: [Int32]
    }

    static func tokenizerCases() throws -> [TokenizerCase] {
        let root = try value("tokenizer.json")
        return array(field(root, "cases")).map { entry in
            TokenizerCase(text: string(field(entry, "text")) ?? "",
                          ids: ints(field(entry, "ids")).map(Int32.init),
                          userIDs: ints(field(entry, "user_ids")).map(Int32.init))
        }
    }

    struct Record {
        let request: JevValue
        let state: String
        let instructions: String
        let options: [String]
        let ids: [Int32]
        let positionIDs: [Int32]
        let decideIndex: Int
        let optionIndices: [Int]
        let logits: [Double]
        let probabilities: [Double]
        let answer: JevValue
    }

    static func records() throws -> [Record] {
        let root = try value("records.json")
        return array(field(root, "records")).map { entry in
            let rendered = field(entry, "rendered")!
            let encoding = field(entry, "encoding")!
            return Record(
                request: field(entry, "request")!,
                state: string(field(rendered, "state")) ?? "",
                instructions: string(field(rendered, "instructions")) ?? "",
                options: strings(field(rendered, "options")),
                ids: ints(field(encoding, "ids")).map(Int32.init),
                positionIDs: ints(field(encoding, "position_ids")).map(Int32.init),
                decideIndex: int(field(encoding, "decide_index")) ?? -1,
                optionIndices: ints(field(encoding, "option_indices")),
                logits: doubles(field(entry, "logits")),
                probabilities: doubles(field(entry, "probabilities")),
                answer: field(entry, "answer")!)
        }
    }

    struct PackedQuestion {
        let id: String
        let instructions: String
        let options: [String]
        let decideIndex: Int
        let optionIndices: [Int]
        let probabilities: [Double]
    }

    struct PackedRecord {
        let request: JevValue
        let state: String
        let length: Int
        let ids: [Int32]
        let positionIDs: [Int32]
        let segmentIDs: [Int32]
        let questions: [PackedQuestion]
    }

    /// Anfragen, die PyTorch als eine Sequenz gerechnet hat. Referenz für die Frage, ob getrennte
    /// Aufrufe dasselbe liefern wie der gepackte Lauf.
    static func packedRecords() throws -> [PackedRecord] {
        let root = try value("packed.json")
        return array(field(root, "records")).map { entry in
            let encoding = field(entry, "encoding")!
            return PackedRecord(
                request: field(entry, "request")!,
                state: string(field(entry, "state")) ?? "",
                length: int(field(entry, "length")) ?? 0,
                ids: ints(field(encoding, "ids")).map(Int32.init),
                positionIDs: ints(field(encoding, "position_ids")).map(Int32.init),
                segmentIDs: ints(field(encoding, "segment_ids")).map(Int32.init),
                questions: array(field(entry, "questions")).map { q in
                    PackedQuestion(id: string(field(q, "id")) ?? "",
                                   instructions: string(field(q, "instructions")) ?? "",
                                   options: strings(field(q, "options")),
                                   decideIndex: int(field(q, "decide_index")) ?? -1,
                                   optionIndices: ints(field(q, "option_indices")),
                                   probabilities: doubles(field(q, "probabilities")))
                })
        }
    }

    static func runtimeConfig() throws -> (length: Int, options: Int, padID: Int32, maskNeg: Float) {
        let root = try value("runtime.json")
        return (int(field(root, "sequence_length")) ?? 512,
                int(field(root, "max_options")) ?? 8,
                Int32(int(field(root, "pad_id")) ?? 151_643),
                Float(double(field(root, "mask_neg")) ?? -1e4))
    }

    static func sharedTokenizer() throws -> Qwen3Tokenizer {
        if let cached = cachedTokenizer { return cached }
        let tokenizer = try Qwen3Tokenizer(contentsOf: tokenizerURL)
        cachedTokenizer = tokenizer
        return tokenizer
    }

    nonisolated(unsafe) private static var cachedTokenizer: Qwen3Tokenizer?

    /// Welches Modell getestet wird. JEV_MODEL überschreibt die Vorgabe.
    /// Mit JEV_REQUIRE_MODEL=1 ist ein fehlendes Modell ein Fehlschlag statt eines stillen
    /// Durchlaufs. reproduce.sh setzt es, damit ein leeres Models/ nicht als grüner Lauf gilt.
    static var requiresModel: Bool {
        ProcessInfo.processInfo.environment["JEV_REQUIRE_MODEL"] == "1"
    }

    static func modelURL() -> URL? {
        if let path = ProcessInfo.processInfo.environment["JEV_MODEL"] {
            return URL(fileURLWithPath: path)
        }
        for name in ["JevCoreML.mlpackage", "Kev06B-Q4-fp16.mlpackage", "Kev06B-fp16.mlpackage",
                     "Kev06B-Q4-fp32.mlpackage", "Kev06B-fp32.mlpackage"] {
            let url = models.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }
}
