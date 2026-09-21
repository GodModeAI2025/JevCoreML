import Foundation
import XCTest
@testable import JevDecisionKit

/// Der Router gegen laya.
///
/// Die Schrifterkennung entscheidet, welcher Checkpoint läuft, und ein falscher Checkpoint
/// kostet mehr als jede Rechenungenauigkeit: auf den eigenen deutschen Referenzanfragen fällt
/// die Konfidenz von 0,95 auf 0,09, wenn der englische statt des mehrsprachigen rechnet.
final class LayaRouterTests: XCTestCase {
    nonisolated(unsafe) static var reference: JevValue?

    func routing() throws -> JevValue {
        if let cached = LayaRouterTests.reference { return cached }
        let url = Fixtures.golden.appendingPathComponent("laya").appendingPathComponent("routing.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "Golden/laya/routing.json fehlt")
        let value = try JevValue.parse(json: try Data(contentsOf: url))
        LayaRouterTests.reference = value
        return value
    }

    func testScriptDetection() throws {
        let entries = Fixtures.array(Fixtures.field(try routing(), "detections"))
        XCTAssertGreaterThan(entries.count, 20)
        for entry in entries {
            let state = Fixtures.field(entry, "state") ?? .null
            let label = (Fixtures.string(Fixtures.field(entry, "text")) ?? "").prefix(40)
            XCTAssertEqual(LayaRouter.stateText(state),
                           Fixtures.string(Fixtures.field(entry, "text")),
                           "Zustandstext weicht ab: \(label)")
            let detection = LayaRouter.analyse(state)
            XCTAssertEqual(detection.script, Fixtures.string(Fixtures.field(entry, "script")),
                           "Schrift weicht ab: \(label)")
            XCTAssertEqual(detection.language, Fixtures.string(Fixtures.field(entry, "language")),
                           "Sprachtipp weicht ab: \(label)")
            XCTAssertEqual(detection.isEnglish,
                           Fixtures.field(entry, "is_english") == .bool(true),
                           "is_english weicht ab: \(label)")
            // Genau, nicht auf Stellen: der laya-Server gibt diese Zahlen aus.
            XCTAssertEqual(detection.nonLatinFraction,
                           Fixtures.double(Fixtures.field(entry, "non_latin_fraction")) ?? -1,
                           "Anteil fremder Schrift weicht ab: \(label)")
            XCTAssertEqual(LayaRouter.detectScript(LayaRouter.stateText(state)),
                           Fixtures.string(Fixtures.field(entry, "detect_script")),
                           "detect_script weicht ab: \(label)")
            let profile = Fixtures.field(entry, "script_profile") ?? .null
            if case let .object(pairs) = profile {
                XCTAssertEqual(detection.scriptProfile.map(\.name), pairs.map(\.key),
                               "Profil hat andere Einträge oder Reihenfolge: \(label)")
                for pair in pairs {
                    let got = detection.scriptProfile.first { $0.name == pair.key }?.fraction
                    XCTAssertEqual(got ?? -1, Fixtures.double(pair.value) ?? -1,
                                   "Anteil \(pair.key) weicht ab: \(label)")
                }
            }
            let decision = try LayaRouter.decide(state: state)
            XCTAssertEqual(decision.checkpoint.name, Fixtures.string(Fixtures.field(entry, "model")),
                           "Checkpoint weicht ab: \(label)")
        }
    }

    func testWorkflowRouting() throws {
        let entries = Fixtures.array(Fixtures.field(try routing(), "routes"))
        XCTAssertGreaterThan(entries.count, 40)
        for entry in entries {
            let state = Fixtures.field(entry, "state") ?? .null
            let ids = Fixtures.strings(Fixtures.field(entry, "question_ids"))
            XCTAssertEqual(LayaRouter.matchTypedDecisionsWorkflow(ids),
                           Fixtures.string(Fixtures.field(entry, "workflow")),
                           "Abgleich des Ablaufs weicht ab bei \(ids)")
            XCTAssertEqual(try LayaRouter.decide(state: state, questionIDs: ids).checkpoint.name,
                           Fixtures.string(Fixtures.field(entry, "model")),
                           "Checkpoint ohne Ablauferkennung weicht ab bei \(ids)")
            XCTAssertEqual(try LayaRouter.decide(state: state, questionIDs: ids,
                                                 autoTaskDetection: true).checkpoint.name,
                           Fixtures.string(Fixtures.field(entry, "model_auto_task")),
                           "Checkpoint mit Ablauferkennung weicht ab bei \(ids)")
        }
    }

    func testOverridePrecedence() throws {
        let entries = Fixtures.array(Fixtures.field(try routing(), "overrides"))
        XCTAssertEqual(entries.count, 8)
        for entry in entries {
            let kwargs = Fixtures.field(entry, "kwargs") ?? .null
            let decision = try LayaRouter.decide(
                state: .string("The customer received two identical charges for the same order."),
                questionIDs: ["urgency"],
                model: Fixtures.string(Fixtures.field(kwargs, "model")),
                task: Fixtures.string(Fixtures.field(kwargs, "task")),
                language: Fixtures.string(Fixtures.field(kwargs, "lang")),
                autoTaskDetection: true)
            XCTAssertEqual(decision.checkpoint.name, Fixtures.string(Fixtures.field(entry, "model")),
                           "Vorrang weicht ab bei \(kwargs.render())")
        }
    }

    /// Der Nutzen des Routers, an den eigenen Referenzanfragen gemessen statt behauptet:
    /// derselbe deutsche Zustand, dieselbe Frage, einmal englischer und einmal mehrsprachiger
    /// Checkpoint. Wenn der Router hier nichts bringt, braucht ihn niemand.
    func testRoutingBeatsEnglishOnGerman() async throws {
        let models = Fixtures.models
        let routed: LayaRouted
        do {
            routed = try LayaRouted.discovering(in: models)
        } catch {
            throw XCTSkip("keine laya-Modelle unter Models/")
        }
        guard await routed.available.contains(.multilingual) else {
            throw XCTSkip("der mehrsprachige Checkpoint fehlt")
        }
        let state = JevValue.string(
            "Erstelle mir bitte eine Tabelle mit den Umsaetzen der letzten zwoelf Monate "
                + "inklusive Quartalssummen.")
        let question = JevQuestion.choice(
            instructions: .string("Welcher Skill soll diese Aufgabe uebernehmen?"),
            criteria: [("research", .string("Quellen recherchieren und zusammenfassen")),
                       ("presentation", .string("Folien und Praesentationen bauen")),
                       ("excel", .string("Tabellen, Kennzahlen und Berechnungen")),
                       ("coding", .string("Software schreiben oder aendern")),
                       ("none", .string("keiner dieser Skills passt"))])

        let result = try await routed.answer(state: state, questions: [("skill", question)])
        XCTAssertEqual(result.checkpoint, "multilingual", "deutscher Zustand muss mehrsprachig laufen")
        let viaRouter = result.answers[0].answer
        let viaEnglish = try await routed.engine(.english).ask(question, about: state)

        guard case let .choice(routedChoice) = viaRouter.answer,
              case let .choice(englishChoice) = viaEnglish.answer else {
            return XCTFail("keine Auswahlantwort")
        }
        XCTAssertGreaterThan(viaRouter.confidence, viaEnglish.confidence,
                             "der Router soll die sicherere Antwort liefern")
        print("Router: \(routedChoice.choice) mit \(viaRouter.confidence), "
              + "englisch: \(englishChoice.choice) mit \(viaEnglish.confidence)")
    }

    func testUnknownNameIsRefused() {
        XCTAssertThrowsError(try LayaRouter.decide(state: .string("x"), model: "gibt-es-nicht"))
    }
}
