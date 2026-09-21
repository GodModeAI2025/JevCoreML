import Foundation
import XCTest
@testable import JevDecisionKit

/// JSON so lesen und schreiben, wie Python es tut. Aus der zweiten Prüfung, die drei Stellen
/// fand, an denen der Port anders serialisierte oder las als die Referenz.
final class SerializationParityTests: XCTestCase {
    /// `json.dumps([float("nan"), float("inf"), float("-inf")])` in Python.
    func testNonFiniteNumbersLikeJSONDumps() {
        let value = JevValue.array([.double(.nan), .double(.infinity), .double(-.infinity)])
        XCTAssertEqual(PythonJSON.dumps(value, ensureASCII: false), "[NaN, Infinity, -Infinity]")
        XCTAssertEqual(PythonJSON.dumps(value, ensureASCII: true), "[NaN, Infinity, -Infinity]")
    }

    /// `json.dumps(chr(0x7f))` maskiert DEL, mit `ensure_ascii=False` bleibt es roh.
    func testDeleteIsEscapedOnlyWithEnsureASCII() {
        let del = JevValue.string(String(Unicode.Scalar(0x7F)))
        XCTAssertEqual(PythonJSON.dumps(del, ensureASCII: true), #""\u007f""#)
        XCTAssertEqual(PythonJSON.dumps(del, ensureASCII: false), "\"\u{7F}\"")
    }

    /// Python behält Schlüssel, die nur kanonisch gleich sind, als zwei Einträge.
    func testCanonicallyEquivalentKeysStaySeparate() throws {
        let composed = "caf\u{E9}", decomposed = "cafe\u{301}"
        let json = "{\"\(composed)\": 1, \"\(decomposed)\": 2, \"\(composed)\": 3}"
        guard case let .object(pairs) = try JevValue.parse(json: json) else {
            return XCTFail("kein Objekt")
        }
        XCTAssertEqual(pairs.count, 2, "zwei verschiedene Schlüssel, der dritte überschreibt den ersten")
        XCTAssertTrue(pairs[0].key.utf8.elementsEqual(composed.utf8))
        XCTAssertEqual(pairs[0].value, .int(3))
        XCTAssertTrue(pairs[1].key.utf8.elementsEqual(decomposed.utf8))
        XCTAssertNotEqual(JevValue.string(composed), JevValue.string(decomposed))
    }
}
