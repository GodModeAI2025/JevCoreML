import Foundation
import XCTest
@testable import JevDecisionKit

/// Die Presets gegen laya.
///
/// Zwei Prüfungen: stimmt der Wortlaut, und kommt auf einem typischen Zustand dasselbe heraus.
/// Die erste fängt einen Fehler im Generator, die zweite alles, was ein anderer Wortlaut am
/// Modell ändern würde. Die Referenz schreibt `gen_laya_presets.py` aus laya selbst.
final class LayaPresetTests: XCTestCase {
    struct Preset {
        let name: String
        let state: JevValue
        let questions: [(id: String, question: JevQuestion)]
        let answers: JevValue
    }

    func reference() throws -> [Preset] {
        let url = Fixtures.golden.appendingPathComponent("laya").appendingPathComponent("presets.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "Golden/laya/presets.json fehlt")
        let root = try JevValue.parse(json: try Data(contentsOf: url))
        return try Fixtures.array(Fixtures.field(root, "presets")).map { entry in
            let state = Fixtures.field(entry, "state") ?? .null
            let questions = Fixtures.field(entry, "questions") ?? .null
            let request = try JevRequest.parse(.object([("state", state), ("questions", questions)]))
            return Preset(name: Fixtures.string(Fixtures.field(entry, "name")) ?? "",
                          state: state, questions: request.questions,
                          answers: Fixtures.field(entry, "answers") ?? .null)
        }
    }

    /// Ein Encoder ohne Modell reicht, um zu sehen, was das Modell zu lesen bekäme.
    func encoder() throws -> LayaEncoder {
        let runtime = try LayaFixtures.runtime()
        return LayaEncoder(tokenizer: try LayaFixtures.sharedTokenizer(),
                           maxLength: runtime.maxLength, headMaxLength: runtime.headMaxLength,
                           clsID: runtime.clsID, sepID: runtime.sepID, maskID: runtime.maskID,
                           padID: runtime.padID, maskToken: runtime.maskToken)
    }

    func testPresetsMatchLayaWordForWord() throws {
        let presets = try reference()
        XCTAssertEqual(presets.map(\.name), LayaPresets.all.map(\.name))
        let encoder = try encoder()
        for (preset, ours) in zip(presets, LayaPresets.all) {
            XCTAssertEqual(ours.questions.map(\.id), preset.questions.map(\.id),
                           "Fragen oder Reihenfolge weichen ab in \(preset.name)")
            for (a, b) in zip(ours.questions, preset.questions) {
                // Verglichen wird, was das Modell zu lesen bekommt: Sequenz, Marker, Fragetyp.
                // Gleiche Token heißen gleicher Wortlaut, bis aufs letzte Zeichen.
                let mine = try encoder.encode(state: preset.state, question: a.question)
                let theirs = try encoder.encode(state: preset.state, question: b.question)
                XCTAssertEqual(mine.qtype, theirs.qtype, "\(preset.name).\(a.id)")
                XCTAssertEqual(mine.options, theirs.options, "\(preset.name).\(a.id)")
                XCTAssertEqual(mine.ids, theirs.ids, "\(preset.name).\(a.id)")
            }
        }
    }

    /// `email` nimmt eigene Teams, der Rest bleibt, wie er ist.
    func testEmailTakesOwnCategories() {
        let teams: [(name: String, description: JevValue)] = [("sales", .null), ("support", .null)]
        let questions = LayaPresets.email(categories: teams)
        XCTAssertEqual(questions.map(\.id), LayaPresets.email().map(\.id))
        guard case let .choice(_, criteria) = questions[0].question else {
            return XCTFail("category ist keine Auswahlfrage")
        }
        XCTAssertEqual(criteria.map(\.name), ["sales", "support"])
    }

    /// laya schreibt `categories or {...}`: eine leere Liste heißt Vorgabe. Der Port warf vorher
    /// `noOptions`, weil er die leere Liste durchreichte.
    func testEmailWithEmptyCategoriesFallsBackToDefaults() {
        guard case let .choice(_, criteria) = LayaPresets.email(categories: [])[0].question else {
            return XCTFail("category ist keine Auswahlfrage")
        }
        XCTAssertEqual(criteria.map(\.name), LayaPresets.emailCategories.map(\.name))
    }

    func testPresetsAnswerLikeLaya() async throws {
        let presets = try reference()
        guard let modelURL = LayaFixtures.modelURL(), LayaFixtures.checkpoint == "english" else {
            throw XCTSkip("braucht den englischen laya-Checkpoint")
        }
        let system = try LayaSystemOne(modelURL: modelURL, tokenizerURL: LayaFixtures.tokenizerURL)
        var compared = 0
        for (preset, ours) in zip(presets, LayaPresets.all) {
            let answers = try await system.ask(ours.questions, about: preset.state)
            for (id, answer) in answers {
                guard let want = Fixtures.field(preset.answers, id) else {
                    XCTFail("keine Referenz für \(preset.name).\(id)"); continue
                }
                let label = "\(preset.name).\(id)"
                switch answer.answer {
                case let .choice(choice):
                    XCTAssertEqual(choice.choice, Fixtures.string(Fixtures.field(want, "choice")), label)
                case let .score(score):
                    XCTAssertEqual(score.score, Fixtures.double(Fixtures.field(want, "score")) ?? -1,
                                   accuracy: 0.02, label)
                case let .noul(noul):
                    XCTAssertEqual(noul.noul, Fixtures.double(Fixtures.field(want, "noul")) ?? -1,
                                   accuracy: 0.02, label)
                }
                XCTAssertEqual(answer.confidence, Fixtures.double(Fixtures.field(want, "confidence")) ?? -1,
                               accuracy: 0.02, "Konfidenz \(label)")
                compared += 1
            }
        }
        XCTAssertEqual(compared, 24, "alle 24 Presetfragen müssen verglichen sein")
    }
}
