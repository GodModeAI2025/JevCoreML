import XCTest
@testable import JevDecisionKit

/// Der JSON-Leser ist selbst geschrieben, weil `JSONSerialization` die Schlüsselreihenfolge
/// wegwirft. Damit erbt er auch die Pflicht, dieselben Dokumente anzunehmen und abzulehnen
/// wie die Referenz. Diese Tests halten die Stellen fest, an denen er das nicht tat.
final class JSONReaderTests: XCTestCase {
    private func parse(_ text: String) throws -> JevValue {
        try JevValue.parse(json: text)
    }

    func testRejectsMalformedNumbers() {
        // Alles hier ist kein gültiges JSON, ging vorher aber durch.
        for bad in ["007", "+1", "1.", ".5", "1e", "1e+", "-", "01", "1.2.3", "--1"] {
            XCTAssertThrowsError(try parse("{\"a\": \(bad)}"), "\(bad) hätte abgelehnt werden müssen")
        }
    }

    func testAcceptsValidNumbers() throws {
        for good in ["0", "-0", "1", "-1", "1.5", "1e3", "1E-3", "1.5e+10", "0.0"] {
            XCTAssertNoThrow(try parse("{\"a\": \(good)}"), "\(good) ist gültiges JSON")
        }
    }

    /// Python kennt keine Breitenbegrenzung für Ganzzahlen und gibt sie mit allen Ziffern aus.
    func testHugeIntegerKeepsAllDigits() throws {
        let digits = "123456789012345678901234567890"
        let value = try parse("{\"n\": \(digits)}")
        guard case let .object(pairs) = value else { return XCTFail("Objekt erwartet") }
        XCTAssertEqual(pairs[0].value.render(), digits)
        if case .double = pairs[0].value { XCTFail("darf nicht über Double gehen") }
    }

    /// Pythons json nimmt diese drei an, und kev.api.render gibt sie als nan, inf, -inf aus.
    func testAcceptsNaNAndInfinity() throws {
        XCTAssertEqual(try parse("{\"a\": NaN}").render(), "a: nan")
        XCTAssertEqual(try parse("{\"a\": Infinity}").render(), "a: inf")
        XCTAssertEqual(try parse("{\"a\": -Infinity}").render(), "a: -inf")
    }

    /// Python baut aus doppelten Schlüsseln ein dict: letzter Wert, erste Position.
    func testDuplicateKeysKeepLastValueAtFirstPosition() throws {
        guard case let .object(pairs) = try parse(#"{"a": 1, "b": 2, "a": 3}"#) else {
            return XCTFail("Objekt erwartet")
        }
        XCTAssertEqual(pairs.map(\.key), ["a", "b"])
        XCTAssertEqual(pairs[0].value, .int(3))
    }

    /// Eine Schachtelung ohne Grenze beendete den Prozess über einen Stapelüberlauf.
    func testDepthIsCapped() {
        let deep = String(repeating: "[", count: 500) + String(repeating: "]", count: 500)
        XCTAssertThrowsError(try parse(deep))
        let fine = String(repeating: "[", count: 100) + String(repeating: "]", count: 100)
        XCTAssertNoThrow(try parse(fine))
    }

    /// Pythons str.lstrip nimmt zusätzlich die vier Informationstrenner.
    func testLstripMatchesPython() {
        XCTAssertEqual(JevValue.lstrip("\u{1C}\u{1D}\u{1E}\u{1F} x"), "x")
        XCTAssertEqual(JevValue.lstrip("  \u{A0}\u{2028}x"), "x")
        XCTAssertEqual(JevValue.lstrip("x "), "x ")
        XCTAssertEqual(JevValue.lstrip("   "), "")
    }

    /// Pythons round(x, 2) rundet kaufmännisch, nicht von null weg.
    func testRoundingMatchesPython() {
        let cases: [(Double, Double)] = [
            (0.015, 0.01), (0.045, 0.04), (0.125, 0.12), (0.155, 0.15),
            (2.675, 2.67), (0.855, 0.85), (0.345, 0.34), (0.025, 0.03),
        ]
        for (input, want) in cases {
            XCTAssertEqual(JevRequest.roundHalfToEven(input), want, accuracy: 1e-12,
                           "round(\(input), 2) ist in Python \(want)")
        }
    }
}
