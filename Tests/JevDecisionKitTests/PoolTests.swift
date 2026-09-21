import CoreML
import XCTest
@testable import JevDecisionKit

/// Shape-Buckets: mehrere Exporte, die Anfrage geht an den kleinsten passenden.
/// Läuft nur, wenn mindestens zwei Exporte unter Models/ liegen.
final class PoolTests: XCTestCase {
    private func poolModels() -> [URL] {
        ["Kev06B-L256-Q4-fp16.mlpackage", "Kev06B-Q4-fp16.mlpackage", "Kev06B-L1024-Q4K96-fp16.mlpackage"]
            .map { Fixtures.models.appendingPathComponent($0) }
            .filter { FileManager.default.fileExists(atPath: $0.path) }
    }

    private func makePool() throws -> SystemOnePool? {
        let urls = poolModels()
        guard urls.count >= 2 else {
            if Fixtures.requiresModel {
                XCTFail("JEV_REQUIRE_MODEL=1, aber es liegen weniger als zwei Exporte unter Models/")
            } else {
                print("weniger als zwei Exporte unter Models/, Pool-Tests übersprungen")
            }
            return nil
        }
        // 4 GiB je Export statt der 16 GiB für einen Server: der Testprozess macht nur eine
        // Handvoll Vorhersagen, und die Payload-Dateien entstehen vor allem unter Dauerlast.
        // Gemessen fiel der freie Platz über die ganze Suite um gut 3 GiB. Die volle Reserve
        // prüft testRefusesWhenScratchSpaceIsShort.
        return try SystemOnePool(modelURLs: urls, tokenizerURL: Fixtures.tokenizerURL,
                                 minimumFreeBytesPerMember: 4 << 30)
    }

    /// Jeder geladene Export belegt rund 14 GB Platte im Temp-Verzeichnis. Reicht der Platz
    /// nicht, muss der Pool ablehnen, bevor er den ersten Export lädt, statt die Platte zu füllen.
    func testRefusesWhenScratchSpaceIsShort() throws {
        let urls = poolModels()
        try XCTSkipIf(urls.isEmpty, "kein Export unter Models/")
        XCTAssertThrowsError(try SystemOnePool(modelURLs: urls, tokenizerURL: Fixtures.tokenizerURL,
                                               minimumFreeBytesPerMember: .max)) { error in
            XCTAssertTrue(String(describing: error).contains("zu wenig Platz"), "\(error)")
        }
        XCTAssertNotNil(SystemOnePool.freeScratchBytes(), "freier Platz muss sich ermitteln lassen")
    }

    /// Bei gleicher Länge gewinnt der Fan-Out-Export, nicht der mit einer Frage je Durchlauf.
    /// Aufsteigend sortiert kam der Phase-1-Export zuerst, und jede Anfrage mit mehreren Fragen
    /// kostete mehrere Läufe statt einem.
    func testFanOutWinsAtEqualLength() {
        var members = [(512, 1, 8), (256, 4, 8), (512, 4, 8), (1024, 4, 96), (512, 4, 96)]
        members.sort(by: SystemOnePool.cheaperFirst)
        XCTAssertEqual(members.map { "\($0.0)/\($0.1)/\($0.2)" },
                       ["256/4/8", "512/4/96", "512/4/8", "512/1/8", "1024/4/96"])
    }

    func testMembersAreSortedBySmallestBudget() async throws {
        guard let pool = try makePool() else { return }
        let lengths = pool.members.map(\.sequenceLength)
        XCTAssertEqual(lengths, lengths.sorted(), "Mitglieder müssen aufsteigend sortiert sein")
    }

    /// Eine kurze Anfrage darf nicht im größten Export landen, sonst polstert sie umsonst.
    func testShortRequestTakesTheSmallestBucket() async throws {
        guard let pool = try makePool() else { return }
        let record = try Fixtures.records()[0]          // 26 Token
        let chosen = try await pool.route(try JevRequest.parse(record.request))
        XCTAssertEqual(chosen.sequenceLength, pool.members[0].sequenceLength,
                       "kurze Anfrage gehört in den kleinsten Bucket")
    }

    /// Eine Anfrage mit vielen Optionen passt nur in den weiten Export.
    func testWideRequestSkipsNarrowBuckets() async throws {
        guard let pool = try makePool() else { return }
        guard let widest = pool.members.last, widest.maxOptions >= 20 else { return }
        let criteria = (0 ..< 20).map { (name: "option_\($0)", description: JevValue.null) }
        let request = JevRequest(state: .string("kurzer Zustand"), questions: [
            ("viele", .choice(instructions: "Welche Option?", criteria: criteria)),
        ])
        let chosen = try await pool.route(request)
        XCTAssertGreaterThanOrEqual(chosen.maxOptions, 20)
    }

    /// Dasselbe Ergebnis, egal welcher Bucket rechnet. Das ist die Bedingung dafür, dass
    /// die Aufteilung überhaupt erlaubt ist.
    func testBucketsAgreeWithinPrecision() async throws {
        guard let pool = try makePool() else { return }
        let urls = poolModels()
        let record = try Fixtures.records()[2]          // Skill-Routing, fünf Optionen
        let request = try JevRequest.parse(record.request)
        let viaPool = try await pool.answer(request)

        var reference: [Double] = []
        for url in urls {
            let runtime = try JevRuntime(modelURL: url, tokenizer: try Fixtures.sharedTokenizer())
            let single = SystemOne(runtime: runtime)
            guard case let .choice(a) = try await single.answer(request).answers[0].answer else {
                return XCTFail("Choice erwartet")
            }
            let p = a.probabilities.map(\.probability)
            if reference.isEmpty { reference = p }
            for (x, y) in zip(p, reference) {
                XCTAssertEqual(x, y, accuracy: 3e-2, "Buckets weichen voneinander ab: \(url.lastPathComponent)")
            }
        }
        guard case let .choice(pooled) = viaPool.answers[0].answer else { return XCTFail("Choice erwartet") }
        XCTAssertEqual(pooled.choice, "excel")
    }

    func testTooLongRequestFailsWithTheLargestLimit() async throws {
        guard let pool = try makePool() else { return }
        let widest = pool.members[pool.members.count - 1].sequenceLength
        let request = JevRequest(state: .string(String(repeating: "Wort ", count: 4000)), questions: [
            ("q", .noul(instructions: "lang?")),
        ])
        do {
            _ = try await pool.route(request)
            XCTFail("hätte scheitern müssen")
        } catch let error as JevError {
            guard case let .sequenceTooLong(_, limit) = error else {
                return XCTFail("falscher Fehler: \(error)")
            }
            XCTAssertEqual(limit, widest, "der Fehler muss das größte Budget nennen, nicht das erste")
        }
    }
}
