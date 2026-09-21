import CoreML
import XCTest
@testable import JevDecisionKit

/// Der eigentliche Beweis: dieselben Zahlen wie PyTorch, aber aus Swift und Core ML.
///
/// Läuft nur, wenn unter Models/ ein .mlpackage liegt. JEV_MODEL wählt ein anderes aus,
/// JEV_UNITS die Recheneinheiten (cpuOnly, cpuAndGPU, cpuAndNeuralEngine, all).
final class RuntimeParityTests: XCTestCase {
    private func makeRuntime(_ model: URL? = nil) throws -> (JevRuntime, URL)? {
        guard let url = model ?? Fixtures.modelURL() else {
            if Fixtures.requiresModel {
                XCTFail("JEV_REQUIRE_MODEL=1, aber unter \(Fixtures.models.path) liegt kein .mlpackage")
            } else {
                print("kein Modell unter Models/, Laufzeittests übersprungen")
            }
            return nil
        }
        let config = try Fixtures.runtimeConfig()
        let units = MLComputeUnitsSelection(
            rawValue: ProcessInfo.processInfo.environment["JEV_UNITS"] ?? "all") ?? .all
        let runtime = try JevRuntime(
            modelURL: url,
            tokenizer: try Fixtures.sharedTokenizer(),
            configuration: .init(padID: config.padID, maskNegative: config.maskNeg,
                                 computeUnits: units.value))
        return (runtime, url)
    }

    /// fp16 rechnet intern in halber Genauigkeit; die Grenze folgt dem, was die Golden-Prüfung
    /// in Python gemessen hat, nicht einem Wunschwert.
    private func tolerance(for model: URL) -> Double {
        if let raw = ProcessInfo.processInfo.environment["JEV_TOLERANCE"], let v = Double(raw) { return v }
        return model.lastPathComponent.contains("fp16") ? 2e-2 : 1e-4
    }

    /// Der Vertrag wird am Modell abgelesen, nicht konfiguriert. Geprüft wird, dass die
    /// Erkennung zum Dateinamen passt und dass das Budget für die Golden-Daten reicht.
    func testModelContract() throws {
        guard let (runtime, model) = try makeRuntime() else { return }
        let name = model.lastPathComponent
        let expectedFanOut = name.contains("-Q")
        XCTAssertEqual(runtime.contract, expectedFanOut ? .fanOut : .singleQuestion)
        XCTAssertEqual(runtime.maxQuestions, expectedFanOut ? 4 : 1)

        // Nicht "mindestens so gross wie der Referenzexport": ein kleinerer Bucket ist erlaubt.
        // Verlangt ist, dass das Modell die laengste Golden-Sequenz und die meisten Optionen fasst.
        let longest = try Fixtures.records().map(\.ids.count).max() ?? 0
        let widest = try Fixtures.records().map(\.options.count).max() ?? 0
        XCTAssertGreaterThanOrEqual(runtime.sequenceLength, longest,
                                    "Modell fasst die laengste Golden-Sequenz nicht")
        XCTAssertGreaterThanOrEqual(runtime.maxOptions, widest)
        // Die Zahlen im Dateinamen müssen zum Modell passen, sonst liegt das falsche Budget an.
        if let range = name.range(of: #"L(\d+)"#, options: .regularExpression) {
            XCTAssertEqual(Int(name[range].dropFirst()), runtime.sequenceLength)
        } else {
            XCTAssertEqual(runtime.sequenceLength, 512)
        }
        if let range = name.range(of: #"K(\d+)"#, options: .regularExpression) {
            XCTAssertEqual(Int(name[range].dropFirst()), runtime.maxOptions)
        } else {
            XCTAssertEqual(runtime.maxOptions, 8)
        }
    }

    func testProbabilitiesMatchPyTorch() async throws {
        guard let (runtime, model) = try makeRuntime() else { return }
        let limit = tolerance(for: model)
        var worstProbability = 0.0
        var worstLogit = 0.0
        var flips = 0

        for record in try Fixtures.records() {
            let encoding = try await runtime.encode(state: record.state,
                                                    instructions: record.instructions,
                                                    options: record.options)
            XCTAssertEqual(encoding.ids, record.ids)
            XCTAssertEqual(encoding.segmentIDs.filter { $0 == 0 }.count, encoding.stateTokens)
            let logits = try await runtime.logits(for: encoding,
                                                  optionCounts: [record.options.count])[0]
            let probabilities = JevRuntime.softmax(logits)

            for (got, want) in zip(logits, record.logits) { worstLogit = max(worstLogit, abs(got - want)) }
            for (got, want) in zip(probabilities, record.probabilities) {
                worstProbability = max(worstProbability, abs(got - want))
            }
            let a = probabilities.indices.max { probabilities[$0] < probabilities[$1] }
            let b = record.probabilities.indices.max { record.probabilities[$0] < record.probabilities[$1] }
            if a != b { flips += 1 }
        }

        print(String(format: "%@: max|dlogit| %.3e  max|dp| %.3e  Argmax-Flips %d",
                     model.lastPathComponent, worstLogit, worstProbability, flips))
        XCTAssertEqual(flips, 0, "Argmax weicht von PyTorch ab")
        XCTAssertLessThan(worstProbability, limit, "Wahrscheinlichkeiten weichen zu stark ab")
    }

    func testAnswersMatchKevAPI() async throws {
        guard let (runtime, model) = try makeRuntime() else { return }
        let limit = tolerance(for: model)
        let systemOne = SystemOne(runtime: runtime)

        for record in try Fixtures.records() {
            let request = try JevRequest.parse(record.request)
            let response = try await systemOne.answer(request)
            XCTAssertEqual(response.answers.count, 1)
            XCTAssertEqual(response.usage.inputTokens, record.ids.count, "input_tokens")
            XCTAssertGreaterThan(response.usage.outputTokens, 0)
            XCTAssertEqual(response.usage.passes, 1)
            let (id, answer) = response.answers[0]
            guard let expected = Fixtures.field(record.answer, id) else {
                return XCTFail("Golden-Antwort für \(id) fehlt")
            }

            switch answer {
            case let .noul(a):
                XCTAssertEqual(Fixtures.string(Fixtures.field(expected, "type")), "noul")
                XCTAssertEqual(a.noul, Fixtures.double(Fixtures.field(expected, "noul"))!,
                               accuracy: limit + 0.005, "noul \(id)")
            case let .choice(a):
                XCTAssertEqual(a.choice, Fixtures.string(Fixtures.field(expected, "choice")), "choice \(id)")
                let dist = Fixtures.field(expected, "probabilities")!
                for entry in a.probabilities {
                    XCTAssertEqual(entry.probability, Fixtures.double(Fixtures.field(dist, entry.name))!,
                                   accuracy: limit + 0.005, "p(\(entry.name))")
                }
            case let .score(a):
                XCTAssertEqual(a.score, Fixtures.double(Fixtures.field(expected, "score"))!,
                               accuracy: 4 * limit + 0.005, "score \(id)")
            }
        }
    }

    /// Das Beispiel aus der Zielbeschreibung: Skill-Routing ohne Textgenerierung.
    func testSkillRouting() async throws {
        guard let (runtime, _) = try makeRuntime() else { return }
        let systemOne = SystemOne(runtime: runtime)
        let answer = try await systemOne.choice(
            state: .string("Erstelle mir bitte eine Tabelle mit den Umsaetzen der letzten zwoelf "
                           + "Monate inklusive Quartalssummen."),
            instruction: "Welcher Skill soll diese Aufgabe uebernehmen?",
            options: [
                (name: "research", description: .string("Quellen recherchieren und zusammenfassen")),
                (name: "presentation", description: .string("Folien und Praesentationen bauen")),
                (name: "excel", description: .string("Tabellen, Kennzahlen und Berechnungen")),
                (name: "coding", description: .string("Software schreiben oder aendern")),
                (name: "none", description: .string("keiner dieser Skills passt")),
            ])
        XCTAssertEqual(answer.choice, "excel")
        XCTAssertGreaterThan(answer["excel"] ?? 0, 0.5)
        XCTAssertEqual(answer.probabilities.reduce(0) { $0 + $1.probability }, 1.0, accuracy: 1e-9)
    }

    /// Kev packt im Original mehrere Fragen in eine Sequenz. Hier laufen sie unter Phase 1 getrennt
    /// und unter Phase 2 gemeinsam. Dass beides dasselbe ergibt, folgt aus der block-kausalen Maske
    /// und den je Zweig neu startenden Positionen, aber hergeleitet ist nicht gemessen.
    func testFanOutMatchesPackedRun() async throws {
        guard let (runtime, model) = try makeRuntime() else { return }
        let limit = tolerance(for: model)
        let systemOne = SystemOne(runtime: runtime)
        let fanOut = runtime.contract == .fanOut
        var worst = 0.0

        var compared = 0
        for record in try Fixtures.packedRecords() {
            // Unter Phase 1 nicht überspringen: JevRuntime.run löst mehrere Fragen nacheinander
            // auf, und genau das soll hier geprüft werden. Vorher warf diese Zeile unter Phase 1
            // beide Datensätze weg, der Schleifenrumpf lief nie, und der Test meldete Erfolg,
            // ohne etwas gemessen zu haben.
            if fanOut, record.questions.count > runtime.maxQuestions { continue }
            compared += 1
            let request = try JevRequest.parse(record.request)
            XCTAssertEqual(request.questions.count, record.questions.count)
            let response = try await systemOne.answer(request)
            XCTAssertEqual(response.usage.passes, fanOut ? 1 : record.questions.count)
            XCTAssertEqual(response.usage.inputTokens, record.length, "input_tokens der gepackten Sequenz")

            for (expected, got) in zip(record.questions, response.answers) {
                XCTAssertEqual(got.id, expected.id, "Reihenfolge der Fragen")
                let probabilities: [Double] = switch got.answer {
                case let .noul(a): a.probabilities
                case let .choice(a): a.probabilities.map(\.probability)
                case let .score(a): a.probabilities
                }
                XCTAssertEqual(probabilities.count, expected.probabilities.count)
                for (a, b) in zip(probabilities, expected.probabilities) { worst = max(worst, abs(a - b)) }
                let x = probabilities.indices.max { probabilities[$0] < probabilities[$1] }
                let y = expected.probabilities.indices.max { expected.probabilities[$0] < expected.probabilities[$1] }
                XCTAssertEqual(x, y, "Argmax bei \(expected.id)")
            }
        }
        print(String(format: "%@ gegen gepackten PyTorch-Lauf: %d Datensätze, max|dp| %.3e",
                     fanOut ? "Fan-Out" : "getrennte Aufrufe", compared, worst))
        XCTAssertGreaterThan(compared, 0, "kein einziger Datensatz verglichen")
        XCTAssertLessThan(worst, limit, "weicht vom gepackten Lauf ab")
    }

    func testPackedEncodingMatchesPyTorch() async throws {
        let encoder = try JevEncoder(tokenizer: try Fixtures.sharedTokenizer())
        for record in try Fixtures.packedRecords() {
            let encoding = try encoder.encode(
                state: record.state,
                questions: record.questions.map {
                    JevEncoder.Prompt(instructions: $0.instructions, options: $0.options)
                })
            XCTAssertEqual(encoding.ids, record.ids, "Token-IDs der gepackten Sequenz")
            XCTAssertEqual(encoding.positionIDs, record.positionIDs, "Positions-IDs")
            XCTAssertEqual(encoding.segmentIDs, record.segmentIDs, "Segment-IDs")
            XCTAssertEqual(encoding.questions.map(\.decideIndex), record.questions.map(\.decideIndex))
            XCTAssertEqual(encoding.questions.map(\.optionEnds), record.questions.map(\.optionIndices))
        }
    }

    /// Der Router über einem echten Modell: zählt er die Fallbacks, und greift die Schwelle?
    func testRouterOverRealModel() async throws {
        guard let (runtime, _) = try makeRuntime() else { return }
        let systemOne = SystemOne(runtime: runtime)
        let record = try Fixtures.records()[2]   // Skill-Routing, klare Verteilung
        let request = try JevRequest.parse(record.request)

        let confident = DecisionRouter(systemOne: systemOne, policy: .init(confidenceFloor: 0.5))
        let accepted = try await confident.answer(request)
        XCTAssertEqual(accepted.count, 1)
        XCTAssertFalse(accepted[0].routed.isEscalated, "klare Verteilung sollte durchgehen")

        // Eine Schwelle oberhalb jeder erreichbaren Konfidenz muss alles eskalieren.
        let strict = DecisionRouter(systemOne: systemOne, policy: .init(confidenceFloor: 1.01))
        let escalated = try await strict.answer(request)
        XCTAssertTrue(escalated[0].routed.isEscalated)
        let stats = await strict.statistics
        XCTAssertEqual(stats.decisions, 1)
        XCTAssertEqual(stats.escalations, 1)
        XCTAssertEqual(stats.fallbackRate, 1.0, accuracy: 1e-12)

        // Die Eskalation lässt sich auflösen, ohne dass der Aufrufer den Typ auspacken muss.
        let resolved = try await escalated[0].routed.resolve { answer, _ in answer }
        if case .choice = resolved {} else { XCTFail("Choice erwartet") }
    }

    func testRepeatedCallsAreStable() async throws {
        guard let (runtime, _) = try makeRuntime() else { return }
        let record = try Fixtures.records()[0]
        let first = try await runtime.probabilities(state: record.state,
                                                    instructions: record.instructions,
                                                    options: record.options)
        // Zweiter Aufruf mit anderer Länge dazwischen, damit die wiederverwendeten Puffer
        // tatsächlich neu gefüllt werden und nicht zufällig noch passen.
        _ = try await runtime.probabilities(state: record.state + " Zusatz",
                                            instructions: record.instructions,
                                            options: record.options)
        let third = try await runtime.probabilities(state: record.state,
                                                    instructions: record.instructions,
                                                    options: record.options)
        for (a, b) in zip(first, third) {
            XCTAssertEqual(a, b, accuracy: 1e-12, "Ergebnis hängt vom vorherigen Aufruf ab")
        }
    }
}
