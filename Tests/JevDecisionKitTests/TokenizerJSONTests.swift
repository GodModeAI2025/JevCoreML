import Foundation
import XCTest
@testable import JevDecisionKit

/// Eine kaputte Tokenizer-Datei muss einen Fehler werfen, nicht den Prozess beenden.
///
/// Vorher: `1e400` als ID endete in `Int(inf)`, `3000000000` in `Int32(_:)`, beides eine Falle,
/// und 300.000 öffnende Klammern liefen in einen Stapelüberlauf. Der Server wäre an einer
/// einzigen solchen Datei gestorben.
final class TokenizerJSONTests: XCTestCase {
    func write(_ text: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tok-\(UUID().uuidString).json")
        try Data(text.utf8).write(to: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    func metaspace(vocab: String, added: String = "[]") -> String {
        """
        {"normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"},
         "pre_tokenizer": {"type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always", "split": true},
         "added_tokens": \(added),
         "model": {"type": "BPE", "vocab": \(vocab), "merges": []}}
        """
    }

    func testHugeNumbersThrowInsteadOfTrapping() throws {
        for vocab in [#"{"a": 1e400}"#, #"{"a": 3000000000}"#, #"{"a": -1e300}"#, #"{"a": 1.5}"#] {
            XCTAssertThrowsError(try MetaspaceTokenizer(contentsOf: write(metaspace(vocab: vocab))), vocab)
        }
        let badAdded = #"[{"content": "<x>", "id": 1e400, "special": true}]"#
        XCTAssertThrowsError(try MetaspaceTokenizer(contentsOf: write(metaspace(vocab: #"{"a": 0}"#, added: badAdded))))
    }

    func testDeepNestingThrowsInsteadOfOverflowing() throws {
        let deep = String(repeating: "[", count: 300_000)
        XCTAssertThrowsError(try TokenizerJSON.parse(contentsOf: write(deep)))
        let fine = String(repeating: "[", count: 100) + String(repeating: "]", count: 100)
        XCTAssertNoThrow(try TokenizerJSON.parse(contentsOf: write(fine)))
    }

    func testUnsupportedAddedTokenFlagsAreRefused() throws {
        for flag in ["rstrip", "single_word"] {
            let added = #"[{"content": "<x>", "id": 1, "special": true, "\#(flag)": true}]"#
            XCTAssertThrowsError(try MetaspaceTokenizer(contentsOf: write(metaspace(vocab: #"{"a": 0}"#, added: added))), flag)
        }
    }
}
