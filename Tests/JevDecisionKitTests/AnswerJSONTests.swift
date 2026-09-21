import XCTest
@testable import JevDecisionKit

/// `usage.output_tokens` ist laut kev die Zahl der Token der serialisierten Antwort. Der Wert
/// hängt damit an jedem Zeichen: an den Trennern, an der Zahl der Nachkommastellen, an der
/// Reihenfolge. Über `JSONSerialization` kam hier rund die Hälfte zu viel heraus, weil eine auf
/// zwei Stellen gerundete 0,66 als 0.66000000000000003 geschrieben wurde.
final class AnswerJSONTests: XCTestCase {
    private func choice(_ pairs: [(String, Double)], confidence: Double) -> JevAnswer {
        .choice(ChoiceAnswer(choice: pairs.max { $0.1 < $1.1 }!.0,
                             confidence: confidence,
                             probabilities: pairs.map { (name: $0.0, probability: $0.1) }))
    }

    /// Wortgleich mit dem, was `json.dumps` in Python für dieselbe Antwort liefert.
    func testMatchesPythonSerialisation() {
        let answers: [(id: String, answer: JevAnswer)] = [
            ("bug", .noul(NoulAnswer(noul: 0.54, probabilities: [0.46, 0.54]))),
            ("team", choice([("tech", 0.68), ("billing", 0.32)], confidence: 0.36)),
        ]
        let text = KevAnswerJSON.text(answers, pretty: false)
        XCTAssertEqual(text,
            #"{"bug": {"type": "noul", "noul": 0.54}, "team": {"type": "choice", "choice": "tech", "confidence": 0.36, "probabilities": {"tech": 0.68, "billing": 0.32}}}"#)
    }

    /// Keine 17-stelligen Doubles, und ganze Werte behalten eine Nachkommastelle wie in Python.
    func testNumbersAreRoundedText() {
        XCTAssertEqual(KevAnswerJSON.number(0.66), "0.66")
        XCTAssertEqual(KevAnswerJSON.number(0.666), "0.67")
        XCTAssertEqual(KevAnswerJSON.number(0.0), "0.0")
        XCTAssertEqual(KevAnswerJSON.number(1.0), "1.0")
        XCTAssertEqual(KevAnswerJSON.number(0.1 + 0.2), "0.3")
        XCTAssertEqual(KevAnswerJSON.number(2.5), "2.5")
        XCTAssertFalse(KevAnswerJSON.number(0.66).contains("0000"))
    }

    func testScoreCarriesLegendAndDistribution() {
        let answer = JevAnswer.score(ScoreAnswer(score: 2.03,
                                                 legend: ["neutral", "gereizt", "eskaliert"],
                                                 probabilities: [0.24, 0.1, 0.66],
                                                 confidence: 0.67))
        let text = KevAnswerJSON.text([("level", answer)], pretty: false)
        XCTAssertTrue(text.contains(#""legend": {"0": "neutral", "1": "gereizt", "2": "eskaliert"}"#))
        XCTAssertTrue(text.contains(#""probabilities": {"0": 0.24, "1": 0.1, "2": 0.66}"#))
        XCTAssertTrue(text.contains(#""score": 2.03"#))
    }

    func testQuotingEscapesControlCharacters() {
        XCTAssertEqual(KevAnswerJSON.quote("a\"b"), #""a\"b""#)
        XCTAssertEqual(KevAnswerJSON.quote("a\nb"), #""a\nb""#)
        XCTAssertEqual(KevAnswerJSON.quote("Grüße"), #""Grüße""#)
    }
}
