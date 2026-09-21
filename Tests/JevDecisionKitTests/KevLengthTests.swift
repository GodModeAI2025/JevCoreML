import CoreML
import XCTest
@testable import JevDecisionKit

/// Ein kev-Paket mit mehreren Eingabelängen: die Anfrage läuft in der kürzesten, in die sie
/// passt, und das Ergebnis darf nicht davon abhängen.
final class KevLengthTests: XCTestCase {
    func testShortestFittingLength() {
        let lengths = [128, 256, 512, 1024, 2048, 3072]
        XCTAssertEqual(JevRuntime.length(for: 1, in: lengths), 128)
        XCTAssertEqual(JevRuntime.length(for: 128, in: lengths), 128)
        XCTAssertEqual(JevRuntime.length(for: 129, in: lengths), 256)
        XCTAssertEqual(JevRuntime.length(for: 1000, in: lengths), 1024)
        XCTAssertEqual(JevRuntime.length(for: 3072, in: lengths), 3072)
        XCTAssertNil(JevRuntime.length(for: 3073, in: lengths))
        XCTAssertEqual(JevRuntime.length(for: 50, in: [512]), 512)
    }

    private func makeRuntime(lengths: [Int]? = nil, units: MLComputeUnits? = nil,
                             allowNeuralEngine: Bool = false) throws -> JevRuntime? {
        guard let url = Fixtures.modelURL() else {
            if Fixtures.requiresModel { XCTFail("JEV_REQUIRE_MODEL=1, aber kein Modell") }
            return nil
        }
        let config = try Fixtures.runtimeConfig()
        let chosen = units ?? (MLComputeUnitsSelection(
            rawValue: ProcessInfo.processInfo.environment["JEV_UNITS"] ?? "all") ?? .all).value
        return try JevRuntime(
            modelURL: url,
            tokenizer: try Fixtures.sharedTokenizer(),
            configuration: .init(padID: config.padID, maskNegative: config.maskNeg,
                                 computeUnits: chosen, sequenceLengths: lengths,
                                 allowNeuralEngine: allowNeuralEngine))
    }

    /// `.all` landet auf der GPU, weil der Planer von Core ML das Paket mit sechs Längen sonst
    /// auf die CPU legt (394 ms statt 8 je Anfrage). Nur `allowNeuralEngine` lässt `.all` durch.
    func testAllIsPinnedToGPUUnlessAllowed() throws {
        guard let pinned = try makeRuntime(units: .all) else { return }
        XCTAssertEqual(pinned.computeUnits, .cpuAndGPU)
        guard let open = try makeRuntime(units: .all, allowNeuralEngine: true) else { return }
        XCTAssertEqual(open.computeUnits, .all)
        guard let cpu = try makeRuntime(units: .cpuOnly) else { return }
        XCTAssertEqual(cpu.computeUnits, .cpuOnly, "eine ausdrückliche Wahl bleibt bestehen")
    }

    /// `sequenceLengths` schneidet die Längen des Pakets; die größte bleibt immer dabei, sonst
    /// passte nicht jede Anfrage.
    func testConfiguredLengthsKeepTheLargest() throws {
        guard let full = try makeRuntime() else { return }
        let package = full.sequenceLengths
        try XCTSkipUnless(package.count > 1, "Paket mit nur einer Länge: \(package)")
        guard let limited = try makeRuntime(lengths: [128, 512]) else { return }
        var expected = package.filter { [128, 512].contains($0) }
        if expected.last != package.last { expected.append(package.last!) }
        XCTAssertEqual(limited.sequenceLengths, expected)
        XCTAssertEqual(limited.sequenceLength, full.sequenceLength)
    }

    /// Dieselben Referenzanfragen einmal in der kürzesten passenden Länge und einmal nur in der
    /// größten. Die Auffüllung ist ausmaskiert, also müssen beide dasselbe ergeben, bis auf fp16.
    func testShortLengthsMatchTheLongest() async throws {
        guard let short = try makeRuntime() else { return }
        let lengths = short.sequenceLengths
        try XCTSkipUnless(lengths.count > 1, "Paket mit nur einer Länge: \(lengths)")
        guard let long = try makeRuntime(lengths: []) else { return }
        XCTAssertEqual(long.sequenceLengths, [lengths.last!])

        var worst = 0.0
        var shorter = 0
        for record in try Fixtures.records() {
            let encoding = try await short.encode(state: record.state,
                                                  instructions: record.instructions,
                                                  options: record.options)
            let a = JevRuntime.softmax(try await short.logits(for: encoding, optionCounts: [record.options.count])[0])
            let b = JevRuntime.softmax(try await long.logits(for: encoding, optionCounts: [record.options.count])[0])
            let usedShort = await short.lastLength
            let usedLong = await long.lastLength
            XCTAssertEqual(usedShort, JevRuntime.length(for: encoding.count, in: lengths))
            XCTAssertEqual(usedLong, lengths.last!)
            if usedShort < usedLong { shorter += 1 }
            for (x, y) in zip(a, b) { worst = max(worst, abs(x - y)) }
            let pickA = a.indices.max { a[$0] < a[$1] }
            let pickB = b.indices.max { b[$0] < b[$1] }
            XCTAssertEqual(pickA, pickB, "Entscheidung hängt an der Länge: \(record.instructions)")
        }
        XCTAssertGreaterThan(shorter, 0, "keine Referenzanfrage lief in einer kürzeren Länge")
        XCTAssertLessThan(worst, 0.02, "kurze und lange Form weichen ab: \(worst)")
        print("kev kurz gegen lang: \(shorter) Anfragen kürzer gerechnet, max|dp| = \(worst)")
    }

    /// Der Warmlauf legt für jede aktive Länge die Instanz an; danach wartet keine Anfrage mehr
    /// auf die Kernel-Kompilierung, egal in welcher Länge sie landet.
    func testWarmUpCoversEveryLength() async throws {
        guard let runtime = try makeRuntime() else { return }
        let lengths = runtime.sequenceLengths
        try XCTSkipUnless(lengths.count > 1, "Paket mit nur einer Länge: \(lengths)")
        let warm = try await runtime.warmUp()
        XCTAssertGreaterThan(warm, 0)
        // Nach dem Warmlauf hat die letzte Länge gerechnet, also die größte.
        let last = await runtime.lastLength
        XCTAssertEqual(last, lengths.last!)
        // Eine kurze Anfrage landet wieder in der kleinsten Länge.
        _ = try await runtime.probabilities(state: "kurz", instructions: "kurz?", options: ["a", "b"])
        let used = await runtime.lastLength
        XCTAssertEqual(used, lengths.first!)
    }
}
