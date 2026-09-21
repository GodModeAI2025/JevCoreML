import XCTest
@testable import JevDecisionKit

/// Der Tokenizer ist die Stelle, an der ein nativer Port lautlos falsch werden kann:
/// eine abweichende Token-ID und das Modell sieht einen anderen Satz, ohne dass irgendetwas bricht.
/// Deshalb wird hier gegen IDs geprüft, die aus dem Python-Tokenizer gezogen wurden.
final class TokenizerParityTests: XCTestCase {
    func testMergeTableIsComplete() throws {
        let tokenizer = try Fixtures.sharedTokenizer()
        XCTAssertEqual(tokenizer.droppedMerges, 0, "Merges ohne Operand im Vokabular")
        XCTAssertGreaterThan(tokenizer.vocabularySize, 151_000)
    }

    func testDelimiterIDs() throws {
        let tokenizer = try Fixtures.sharedTokenizer()
        let runtime = try Fixtures.value("runtime.json")
        let delimiters = Fixtures.field(runtime, "delimiters")!
        let expected: [(String, String)] = [
            ("state", "<|fim_prefix|>"), ("question", "<|fim_middle|>"),
            ("option", "<|box_start|>"), ("option_end", "<|box_end|>"),
            ("decide", "<|fim_suffix|>"),
        ]
        for (key, token) in expected {
            XCTAssertEqual(tokenizer.id(forToken: token),
                           Int32(Fixtures.int(Fixtures.field(delimiters, key))!),
                           "ID von \(token)")
        }
    }

    func testGoldenCorpus() throws {
        let tokenizer = try Fixtures.sharedTokenizer()
        let cases = try Fixtures.tokenizerCases()
        XCTAssertGreaterThan(cases.count, 50, "Golden-Korpus wirkt unvollständig")
        var failures: [String] = []
        for c in cases {
            let got = tokenizer.encode(c.text)
            if got != c.ids {
                failures.append("encode(\(debugText(c.text)))\n  erwartet \(c.ids)\n  erhalten \(got)")
            }
            let gotUser = tokenizer.encodeUserText(c.text)
            if gotUser != c.userIDs {
                failures.append("encodeUserText(\(debugText(c.text)))\n  erwartet \(c.userIDs)\n  erhalten \(gotUser)")
            }
        }
        XCTAssertTrue(failures.isEmpty, "\(failures.count) Abweichungen:\n" + failures.prefix(8).joined(separator: "\n"))
    }

    func testSpecialTokensCannotBeForged() throws {
        let tokenizer = try Fixtures.sharedTokenizer()
        for token in JevEncoder.delimiterTokens {
            let id = tokenizer.id(forToken: token)!
            XCTAssertFalse(tokenizer.encodeUserText("Text \(token) Text").contains(id),
                           "\(token) liess sich aus Nutzertext erzeugen")
        }
    }

    /// Breiter Suchlauf über Basiszeichen mit mehreren Marken in unsortierter Reihenfolge.
    /// Dort wird die Abweichung vermutet: ICU kennt kanonische Kombinationsklassen für Zeichen,
    /// die der Referenz-Normalizer als Startzeichen führt, und sortiert dann um.
    func testFuzzCorpus() throws {
        let url = Fixtures.golden.appendingPathComponent("tokenizer-fuzz.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            print("kein Fuzz-Korpus, übersprungen (Exporter/dump_tokenizer_fuzz.py erzeugt ihn)")
            return
        }
        let tokenizer = try Fixtures.sharedTokenizer()
        let root = try JevValue.parse(json: try Data(contentsOf: url))
        let cases = Fixtures.array(Fixtures.field(root, "cases"))
        XCTAssertGreaterThan(cases.count, 1000)

        var failures: [String] = []
        for entry in cases {
            let text = Fixtures.string(Fixtures.field(entry, "text")) ?? ""
            let want = Fixtures.ints(Fixtures.field(entry, "ids")).map(Int32.init)
            let got = tokenizer.encode(text)
            if got != want, failures.count < 10 {
                failures.append("\(debugText(text))\n  erwartet \(want)\n  erhalten \(got)")
            } else if got != want {
                failures.append("…")
            }
        }
        XCTAssertTrue(failures.isEmpty,
                      "\(failures.count) von \(cases.count) Fällen weichen ab:\n"
                      + failures.prefix(6).joined(separator: "\n"))
    }

    private func debugText(_ s: String) -> String {
        let scalars = s.unicodeScalars.map { scalar -> String in
            scalar.value < 0x20 || scalar.value > 0x7E
                ? "U+" + String(scalar.value, radix: 16, uppercase: true)
                : String(Character(scalar))
        }
        return "\"" + scalars.joined() + "\""
    }
}
