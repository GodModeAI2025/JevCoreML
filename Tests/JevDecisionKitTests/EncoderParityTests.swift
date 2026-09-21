import XCTest
@testable import JevDecisionKit

/// Prüft die beiden Schritte vor dem Modell: das Abflachen der Anfrage in Text
/// (`kev.api.render` / `to_record`) und das Packen in die Tokenfolge (`kev.model.encode`).
final class EncoderParityTests: XCTestCase {
    func testRequestRendering() throws {
        for record in try Fixtures.records() {
            let request = try JevRequest.parse(record.request)
            XCTAssertEqual(request.questions.count, 1)
            XCTAssertEqual(request.state.render(), record.state, "gerenderter Zustand")
            let prompt = try SystemOne.renderQuestion(request.questions[0].question)
            XCTAssertEqual(prompt.instructions, record.instructions, "gerenderte Anweisung")
            XCTAssertEqual(prompt.options, record.options, "gerenderte Optionen")
        }
    }

    func testEncodingMatchesPyTorch() throws {
        let encoder = try JevEncoder(tokenizer: try Fixtures.sharedTokenizer())
        for record in try Fixtures.records() {
            let encoding = try encoder.encode(state: record.state,
                                              instructions: record.instructions,
                                              options: record.options)
            XCTAssertEqual(encoding.ids, record.ids, "Token-IDs")
            XCTAssertEqual(encoding.positionIDs, record.positionIDs, "Positions-IDs")
            XCTAssertEqual(encoding.decideIndex, record.decideIndex, "decide_index")
            XCTAssertEqual(encoding.optionEnds, record.optionIndices, "option_indices")
            XCTAssertFalse(encoding.stateTruncated)
        }
    }

    func testPositionsRestartAfterStateOnlyOnce() throws {
        // Phase 1 hat genau einen Fragezweig, die Positionen laufen also durchgehend.
        let encoder = try JevEncoder(tokenizer: try Fixtures.sharedTokenizer())
        let encoding = try encoder.encode(state: "kurz", instructions: "frage", options: ["a", "b"])
        XCTAssertEqual(encoding.positionIDs, (0 ..< encoding.count).map(Int32.init))
        XCTAssertEqual(encoding.decideIndex, encoding.count - 1)
    }

    func testStateBudget() throws {
        let tokenizer = try Fixtures.sharedTokenizer()
        let long = String(repeating: "Wort ", count: 500)

        let strict = try JevEncoder(tokenizer: tokenizer, maxStateTokens: KevBudget.trainedState, strictState: true)
        XCTAssertThrowsError(try strict.encode(state: long, instructions: "frage", options: ["a"]))

        let lenient = try JevEncoder(tokenizer: tokenizer, maxStateTokens: KevBudget.trainedState, strictState: false)
        let encoding = try lenient.encode(state: long, instructions: "frage", options: ["a"])
        XCTAssertTrue(encoding.stateTruncated)
        // Der Zustandsblock selbst muss auf genau 384 Token gekürzt sein, <state> plus 383.
        // prefix(384).count == 384 wäre dagegen wahr, sobald die Sequenz überhaupt so lang ist,
        // und hätte auch bei völlig fehlender Kürzung nichts gemeldet.
        XCTAssertEqual(encoding.stateTokens, KevBudget.trainedState)
        XCTAssertEqual(encoding.segmentIDs.filter { $0 == 0 }.count, KevBudget.trainedState)
        XCTAssertEqual(encoding.positionIDs[KevBudget.trainedState], Int32(KevBudget.trainedState))
    }

    func testOptionTextFollowsKevAPI() {
        XCTAssertEqual(SystemOne.optionText("excel", .null), "excel")
        XCTAssertEqual(SystemOne.optionText("excel", .string("")), "excel")
        XCTAssertEqual(SystemOne.optionText("excel", .string("Tabellen")), "excel: Tabellen")
        XCTAssertEqual(SystemOne.optionText("n", .int(3)), "n: 3")
        XCTAssertEqual(SystemOne.optionText("b", .bool(true)), "b: True")
    }

    func testValueRendering() {
        XCTAssertEqual(JevValue.null.render(), "")
        XCTAssertEqual(JevValue.bool(false).render(), "False")
        XCTAssertEqual(JevValue.double(1.0).render(), "1.0")
        XCTAssertEqual(JevValue.array([.string("a"), .string("b")]).render(), "- a\n- b")
        let nested = JevValue.object([
            (key: "kunde", value: .object([(key: "plan", value: .string("Enterprise"))])),
            (key: "offen", value: .int(2)),
        ])
        XCTAssertEqual(nested.render(), "kunde:\n  plan: Enterprise\noffen: 2")
    }

    func testJSONKeyOrderSurvives() throws {
        let json = #"{"questions":{"q":{"type":"choice","instructions":"x","criteria":{"z":null,"a":null,"m":null}}},"state":"s"}"#
        let request = try JevRequest.parse(json: Data(json.utf8))
        guard case let .choice(_, criteria) = request.questions[0].question else {
            return XCTFail("choice erwartet")
        }
        XCTAssertEqual(criteria.map(\.name), ["z", "a", "m"], "Reihenfolge der Kriterien")
    }
}
