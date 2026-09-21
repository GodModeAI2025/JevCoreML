import Foundation
import XCTest
@testable import JevDecisionKit

/// Unicode-Verhalten des Tokenizers gegen HuggingFace, Codepunkt für Codepunkt.
///
/// Die Satzkorpora haben zwei ganze Fehlerklassen übersehen, weil kein Satz sie traf: ICU kennt
/// Unicode 17, die Referenz beim Regex Unicode 16 und beim NFC Unicode 9, und Apple gibt einigen
/// privaten Codepunkten eigene Eigenschaften. Ohne Korrektur schnitt der Pretokenizer an 4693
/// Codepunkten anders. Die Referenz schreibt `Exporter/dump_unicode_reference.py`.
final class UnicodeReferenceTests: XCTestCase {
    static let planes: [ClosedRange<UInt32>] = [0x0 ... 0x3FFFF, 0xE0000 ... 0xE01EF]

    func reference() throws -> [String: Any] {
        let url = Fixtures.golden.appendingPathComponent("unicode-reference.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "Golden/unicode-reference.json fehlt")
        return try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    }

    func testRegexClassesMatchOniguruma() throws {
        let classes = try reference()["regex_classes"] as! [String: [[Int]]]
        let patterns = ["L": HFTokenizer.letter, "N": HFTokenizer.number,
                        "S": "[" + HFTokenizer.whitespaceClass + "]"]
        for (name, pattern) in patterns {
            var expected = Set<UInt32>()
            for range in classes[name]! {
                for value in range[0] ... range[1] { expected.insert(UInt32(value)) }
            }
            let regex = try NSRegularExpression(pattern: "^" + pattern + #"\z"#)
            var wrong: [String] = []
            for value in UInt32(0) ... 0x10FFFF {
                guard let scalar = Unicode.Scalar(value) else { continue }
                let text = String(scalar)
                let hit = regex.firstMatch(in: text, range: NSRange(location: 0, length: (text as NSString).length)) != nil
                if hit != expected.contains(value) { wrong.append(String(format: "%04X", value)) }
            }
            XCTAssertTrue(wrong.isEmpty, "\\\(name): \(wrong.count) Codepunkte weichen ab, etwa \(wrong.prefix(12))")
        }
    }

    func testNFCMatchesReferenceForEveryCodePoint() throws {
        let root = try reference()
        let single = root["nfc_single"] as! [String: [Int]]
        var wrong: [String] = []
        for range in UnicodeReferenceTests.planes {
            for value in range {
                guard let scalar = Unicode.Scalar(value) else { continue }
                let want = single[String(value)].map { $0.map(UInt32.init) } ?? [value]
                let got = HFTokenizer.normalizedNFC(String(scalar)).unicodeScalars.map(\.value)
                if got != want { wrong.append(String(format: "%04X", value)) }
            }
        }
        XCTAssertTrue(wrong.isEmpty, "\(wrong.count) Einzelzeichen weichen ab, etwa \(wrong.prefix(12))")
    }

    func testNFCMatchesReferenceForMarksAndAstralStarters() throws {
        let root = try reference()
        func text(_ values: [Int]) -> String {
            String(String.UnicodeScalarView(values.map { Unicode.Scalar(UInt32($0))! }))
        }
        var wrong: [String] = []
        let lists = (root["nfc_marks"] as! [[[Int]]]) + (root["nfc_extra"] as! [[[Int]]])
            + (root["nfc_precomposed"] as! [[[Int]]])
        XCTAssertGreaterThan(lists.count, 20_000)
        for pair in lists {
            let got = HFTokenizer.normalizedNFC(text(pair[0])).unicodeScalars.map { Int($0.value) }
            if got != pair[1] { wrong.append(pair[0].map { String(format: "%04X", $0) }.joined(separator: " ")) }
        }
        let astral = root["nfc_astral"] as! [String: [Int]]
        for value in UInt32(0x10000) ... 0x3FFFF {
            guard let scalar = Unicode.Scalar(value) else { continue }
            let input = String(scalar) + "\u{0301}"
            let want = astral[String(value)] ?? [Int(value), 0x301]
            let got = HFTokenizer.normalizedNFC(input).unicodeScalars.map { Int($0.value) }
            if got != want { wrong.append(String(format: "%04X 0301", value)) }
        }
        XCTAssertTrue(wrong.isEmpty, "\(wrong.count) Folgen weichen ab, etwa \(wrong.prefix(8))")
    }
}
