import Foundation
import XCTest
@testable import JevDecisionKit

/// `LayaEmail` gegen `laya.email`, über 4043 Mails aus `dump_laya_email.py`.
///
/// Byte für Byte verglichen. Ein bereinigter Text, der an einer Stelle anders ausfällt, ist ein
/// anderer Zustand für das Modell, und das fiele sonst niemandem auf.
final class LayaEmailTests: XCTestCase {
    func reference() throws -> JevValue {
        let url = Fixtures.golden.appendingPathComponent("laya").appendingPathComponent("email.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "Golden/laya/email.json fehlt")
        return try JevValue.parse(json: try Data(contentsOf: url))
    }

    func testCleanBodyMatchesLaya() throws {
        let cases = Fixtures.array(Fixtures.field(try reference(), "cases"))
        XCTAssertGreaterThan(cases.count, 4000, "zu wenige Fälle, um etwas zu belegen")
        var failures: [String] = []
        for entry in cases {
            let body = Fixtures.string(Fixtures.field(entry, "body")) ?? ""
            for (key, limit) in [("clean", 3000), ("clean_200", 200)] {
                let want = Fixtures.string(Fixtures.field(entry, key)) ?? ""
                let got = LayaEmail.cleanBody(body, maxCharacters: limit)
                if got != want || Array(got.unicodeScalars) != Array(want.unicodeScalars) {
                    failures.append("\(key) für \(body.prefix(60).debugDescription):\n"
                        + "  erwartet \(want.prefix(120).debugDescription)\n"
                        + "  erhalten \(got.prefix(120).debugDescription)")
                }
            }
        }
        XCTAssertTrue(failures.isEmpty,
                      "\(failures.count) Abweichungen:\n" + failures.prefix(6).joined(separator: "\n"))
    }

    func testStateMatchesLaya() throws {
        let states = Fixtures.array(Fixtures.field(try reference(), "states"))
        XCTAssertEqual(states.count, 5)
        for entry in states {
            var extra: [(key: String, value: JevValue)] = []
            for pair in Fixtures.array(Fixtures.field(entry, "extra")) {
                let items = Fixtures.array(pair)
                extra.append((Fixtures.string(items[0]) ?? "", items[1]))
            }
            let got = LayaEmail.state(
                subject: Fixtures.string(Fixtures.field(entry, "subject")),
                body: Fixtures.string(Fixtures.field(entry, "body")),
                sender: Fixtures.string(Fixtures.field(entry, "sender")),
                clean: Fixtures.field(entry, "clean") == .bool(true),
                extra: extra)
            // Reihenfolge zählt: der Zustand wird serialisiert und landet so im Tokenizer.
            XCTAssertEqual(got.render(), (Fixtures.field(entry, "state") ?? .null).render())
            XCTAssertEqual(got, Fixtures.field(entry, "state"))
        }
    }
}
