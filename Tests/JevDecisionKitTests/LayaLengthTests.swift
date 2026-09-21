import Foundation
import XCTest
@testable import JevDecisionKit

/// Mehrere Eingabelängen in einem Paket und das Finden der Modelle in einem App-Bundle.
final class LayaLengthTests: XCTestCase {
    func testShortestFittingLength() {
        let lengths = [128, 256, 512]
        XCTAssertEqual(LayaRuntime.length(for: 1, in: lengths), 128)
        XCTAssertEqual(LayaRuntime.length(for: 50, in: lengths), 128)
        XCTAssertEqual(LayaRuntime.length(for: 128, in: lengths), 128)
        XCTAssertEqual(LayaRuntime.length(for: 129, in: lengths), 256)
        XCTAssertEqual(LayaRuntime.length(for: 512, in: lengths), 512)
        XCTAssertNil(LayaRuntime.length(for: 513, in: lengths))
    }

    /// Xcode legt nur die übersetzte `.mlmodelc` ins Bundle. Die muss gefunden und dem Paket
    /// vorgezogen werden; ein Checkpoint ohne Tokenizer zählt nicht als vorhanden.
    func testDiscoveryPrefersCompiledModels() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("laya-discovery-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        for name in ["Laya-EN-L512-K512-fp16.mlmodelc", "Laya-EN-L512-K512-fp16.mlpackage",
                     "Laya-ML-L1024-K1024-fp16.mlpackage", "Laya-TD-L1024-K1024-fp16.mlmodelc"] {
            try FileManager.default.createDirectory(at: dir.appendingPathComponent(name),
                                                    withIntermediateDirectories: true)
        }
        // Nur der englische Tokenizer: typed-decisions teilt ihn, multilingual fehlt er.
        FileManager.default.createFile(atPath: dir.appendingPathComponent("laya-tokenizer-english.json").path,
                                       contents: Data("{}".utf8))

        let routed = try LayaRouted.discovering(in: dir)
        XCTAssertEqual(Set(routed.available), [.english, .typedDecisions])
        XCTAssertEqual(routed.source(of: .english)?.modelURL.pathExtension, "mlmodelc")
        XCTAssertEqual(routed.source(of: .typedDecisions)?.modelURL.pathExtension, "mlmodelc")
        XCTAssertNil(routed.source(of: .multilingual))
    }

    /// Dieselben Referenzanfragen einmal in der kürzesten passenden Länge und einmal in der
    /// längsten. Die Auffüllung ist ausmaskiert, also müssen beide dasselbe ergeben, bis auf fp16.
    func testShortLengthsMatchTheLongest() async throws {
        try XCTSkipUnless(LayaFixtures.hasGolden(), "Golden/laya fehlt")
        guard let modelURL = LayaFixtures.modelURL() else {
            throw XCTSkip("kein laya-Modell unter Models/")
        }
        let short = try LayaSystemOne(modelURL: modelURL, tokenizerURL: LayaFixtures.tokenizerURL)
        let lengths = short.runtime.metadata.sequenceLengths
        try XCTSkipUnless(lengths.count > 1, "Paket mit nur einer Länge: \(lengths)")
        let long = try LayaSystemOne(modelURL: modelURL, tokenizerURL: LayaFixtures.tokenizerURL,
                                     configuration: .init(sequenceLengths: []))
        XCTAssertEqual(long.runtime.activeLengths, [lengths.last!])

        var worst = 0.0
        var shorter = 0
        for record in try LayaFixtures.records() {
            let encoded = try short.runtime.encoder.encode(state: record.state, question: record.question)
            if LayaRuntime.length(for: encoded.ids.count, in: lengths)! < lengths.last! { shorter += 1 }
            let a = try await short.ask(record.question, about: record.state)
            let b = try await long.ask(record.question, about: record.state)
            for (x, y) in zip(a.probabilities, b.probabilities) { worst = max(worst, abs(x - y)) }
            if case let .choice(ca) = a.answer, case let .choice(cb) = b.answer {
                XCTAssertEqual(ca.choice, cb.choice, "Entscheidung hängt an der Länge: \(record.questionID)")
            }
        }
        XCTAssertGreaterThan(shorter, 0, "keine Referenzanfrage lief in einer kürzeren Länge")
        XCTAssertLessThan(worst, 0.02, "kurze und lange Form weichen ab: \(worst)")
        print("laya kurz gegen lang: \(shorter) Anfragen kürzer gerechnet, max|dp| = \(worst)")
    }
}
