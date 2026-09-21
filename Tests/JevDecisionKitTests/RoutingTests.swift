import XCTest
@testable import JevDecisionKit

/// Die Eskalationsregel entscheidet, welcher Teil der Fälle überhaupt automatisch läuft.
/// Sie braucht kein Modell, nur Antworten, deshalb läuft dieser Test immer.
final class RoutingTests: XCTestCase {
    private func choice(_ pairs: [(String, Double)]) -> ChoiceAnswer {
        let best = pairs.max { $0.1 < $1.1 }!.0
        let p = pairs.map(\.1)
        return ChoiceAnswer(choice: best,
                            confidence: SystemOne.choiceConfidence(p),
                            probabilities: pairs.map { (name: $0.0, probability: $0.1) })
    }

    func testConfidenceFloor() {
        let policy = RoutingPolicy(confidenceFloor: 0.5)
        let clear = choice([("a", 0.9), ("b", 0.05), ("c", 0.05)])
        let muddy = choice([("a", 0.4), ("b", 0.35), ("c", 0.25)])
        XCTAssertFalse(policy.route(clear).isEscalated)
        XCTAssertTrue(policy.route(muddy).isEscalated)
        if case let .belowConfidenceFloor(confidence, floor)? = policy.route(muddy).reason {
            XCTAssertLessThan(confidence, floor)
        } else {
            XCTFail("falscher Eskalationsgrund")
        }
    }

    /// Die Schwelle ist keine einzelne Zahl: eine riskante Aktion braucht mehr als eine harmlose.
    func testPerOutcomeThreshold() {
        let policy = RoutingPolicy(confidenceFloor: 0.5, perOutcome: ["approve_transfer": 0.9])
        let answer = choice([("approve_transfer", 0.8), ("check_balance", 0.15), ("support", 0.05)])
        XCTAssertGreaterThan(answer.confidence, 0.5)
        XCTAssertTrue(policy.route(answer).isEscalated)
        let harmless = choice([("check_balance", 0.8), ("approve_transfer", 0.15), ("support", 0.05)])
        XCTAssertFalse(policy.route(harmless).isEscalated)
    }

    /// Noul trägt keine Konfidenz, also wird zweiseitig geschnitten.
    func testNoulBand() {
        let policy = RoutingPolicy(noulYes: 0.7, noulNo: 0.3)
        XCTAssertFalse(policy.route(NoulAnswer(noul: 0.92, probabilities: [0.08, 0.92])).isEscalated)
        XCTAssertFalse(policy.route(NoulAnswer(noul: 0.05, probabilities: [0.95, 0.05])).isEscalated)
        let undecided = policy.route(NoulAnswer(noul: 0.44, probabilities: [0.56, 0.44]))
        XCTAssertTrue(undecided.isEscalated)
        XCTAssertEqual(undecided.reason, .noulUndecided(value: 0.44, no: 0.3, yes: 0.7))
    }

    /// Score hatte keinen einzigen Test: weder die Regel noch der Router wurden je mit einer
    /// Score-Antwort aufgerufen.
    func testScoreBand() {
        let policy = RoutingPolicy(confidenceFloor: 0.5)
        // Eine Spitze auf einer Stufe: Konfidenz 1, also handeln.
        let klar = ScoreAnswer(score: 1.0, legend: ["a", "b", "c"],
                               probabilities: [0.0, 1.0, 0.0],
                               confidence: SystemOne.scoreConfidence([0.0, 1.0, 0.0]))
        XCTAssertEqual(klar.confidence, 1.0, accuracy: 1e-12)
        XCTAssertFalse(policy.route(klar).isEscalated)

        // Zwei gleich hohe Spitzen an den Rändern: Konfidenz 0, also eskalieren.
        let zweigipflig = ScoreAnswer(score: 1.0, legend: ["a", "b", "c"],
                                      probabilities: [0.5, 0.0, 0.5],
                                      confidence: SystemOne.scoreConfidence([0.5, 0.0, 0.5]))
        XCTAssertEqual(zweigipflig.confidence, 0.5, accuracy: 1e-12)
        let routed = policy.route(zweigipflig)
        XCTAssertFalse(routed.isEscalated, "0,5 liegt genau auf dem Boden und gilt als erfüllt")

        let streng = RoutingPolicy(confidenceFloor: 0.6)
        XCTAssertTrue(streng.route(zweigipflig).isEscalated)
        XCTAssertEqual(streng.route(zweigipflig).reason,
                       .belowConfidenceFloor(confidence: 0.5, floor: 0.6))
    }

    /// Eine Auffangoption in Grossschreibung muss genauso zählen.
    func testNoneOptionIsCaseInsensitiveOnBothSides() {
        var policy = RoutingPolicy(confidenceFloor: 0.0, requireNoneOption: true,
                                   noneOptionNames: ["Sonstiges", "NONE"])
        policy.requireNoneOption = true
        let answer = choice([("billing", 0.6), ("sonstiges", 0.4)])
        XCTAssertFalse(policy.route(answer).isEscalated, "sonstiges muss Sonstiges treffen")
    }

    func testNoneOptionRequired() {
        var policy = RoutingPolicy(confidenceFloor: 0.0)
        policy.requireNoneOption = true
        let without = choice([("billing", 0.6), ("shipping", 0.4)])
        XCTAssertEqual(policy.route(without).reason, .missingNoneOption)
        let with = choice([("billing", 0.6), ("shipping", 0.3), ("none", 0.1)])
        XCTAssertFalse(policy.route(with).isEscalated)
    }

    func testResolveRunsEscalation() async throws {
        let policy = RoutingPolicy(confidenceFloor: 0.9)
        let answer = choice([("a", 0.6), ("b", 0.4)])
        let resolved = try await policy.route(answer).resolve { _, _ in
            self.choice([("b", 0.99), ("a", 0.01)])
        }
        XCTAssertEqual(resolved.choice, "b")
    }

    func testStatisticsCountFallbacks() async throws {
        // Ohne Modell lässt sich der Router nicht bauen, also wird die Regel direkt gezählt.
        let policy = RoutingPolicy(confidenceFloor: 0.7)
        let answers = [
            choice([("a", 0.95), ("b", 0.05)]),
            choice([("a", 0.45), ("b", 0.4), ("c", 0.15)]),
            choice([("a", 0.5), ("b", 0.3), ("c", 0.2)]),
        ]
        let escalated = answers.filter { policy.route($0).isEscalated }.count
        XCTAssertEqual(escalated, 2)
        XCTAssertEqual(Double(escalated) / Double(answers.count), 2.0 / 3.0, accuracy: 1e-12)
    }

    func testConfidenceFormulasMatchKevAPI() {
        // choice_confidence: (max - 1/K) / (1 - 1/K)
        XCTAssertEqual(SystemOne.choiceConfidence([0.25, 0.25, 0.25, 0.25]), 0.0, accuracy: 1e-12)
        XCTAssertEqual(SystemOne.choiceConfidence([1.0, 0.0]), 1.0, accuracy: 1e-12)
        XCTAssertEqual(SystemOne.choiceConfidence([0.6, 0.4]), 0.2, accuracy: 1e-12)
        // score_confidence: 1 - E|Stufe - Modus| / (L-1)
        XCTAssertEqual(SystemOne.scoreConfidence([0.0, 1.0, 0.0]), 1.0, accuracy: 1e-12)
        // Gleichstand: der Modus ist die erste der beiden Spitzen, wie in Python.
        XCTAssertEqual(SystemOne.scoreConfidence([0.5, 0.0, 0.5]), 0.5, accuracy: 1e-12)
        XCTAssertEqual(SystemOne.argmax([0.4, 0.4, 0.2]), 0)
    }
}
