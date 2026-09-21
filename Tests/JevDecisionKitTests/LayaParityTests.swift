import CoreML
import Foundation
import XCTest
@testable import JevDecisionKit

/// Der laya-Port gegen das Original.
///
/// Drei Ebenen, weil ein Fehler auf jeder einzelnen lautlos bleibt: die Token, die Sequenz und
/// die fertige Antwort. Die Referenzwerte kommen aus `dump_laya_golden.py`, also aus laya selbst.
final class LayaParityTests: XCTestCase {
    func testTokenizerMatchesReference() throws {
        try XCTSkipUnless(LayaFixtures.hasGolden(), "Golden/laya fehlt")
        let tokenizer = try LayaFixtures.sharedTokenizer()
        switch LayaFixtures.checkpoint {
        case "multilingual":
            XCTAssertTrue(tokenizer is MetaspaceTokenizer,
                          "mmBERT liest Metaspace-BPE, nicht Byte-Level")
        default:
            XCTAssertEqual((tokenizer as? HFTokenizer)?.flavor, .byteLevelRegex,
                           "die ModernBERT-Ableger lesen ByteLevel mit eingebautem Regex")
        }
        let cases = try LayaFixtures.tokenizerCases()
        XCTAssertGreaterThan(cases.count, 20, "zu wenige Faelle, um etwas zu belegen")
        for testCase in cases {
            XCTAssertEqual(tokenizer.encode(testCase.text), testCase.ids,
                           "Tokenisierung weicht ab fuer \(testCase.text.debugDescription)")
        }
    }

    /// Die 109 Einrueckungsketten sind eigene Token mit `normalized: true`. Ueber BPE ergaeben
    /// sie andere IDs, und der Fehler faellt nur auf, wenn man ihn genau hier prueft.
    func testNormalizedAddedTokens() throws {
        try XCTSkipUnless(LayaFixtures.hasGolden(), "Golden/laya fehlt")
        try XCTSkipIf(LayaFixtures.checkpoint == "multilingual",
                      "mmBERT fuehrt keine normalisierten Added-Token")
        guard let tokenizer = try LayaFixtures.sharedTokenizer() as? HFTokenizer else {
            throw XCTSkip("kein Byte-Level-Tokenizer")
        }
        let runs = [String(repeating: " ", count: 8), String(repeating: " ", count: 16),
                    String(repeating: " ", count: 24)]
        for run in runs {
            let ids = tokenizer.encode("a" + run + "b")
            XCTAssertEqual(ids.count, 3, "\(run.count) Leerzeichen sollten ein Token sein, nicht \(ids.count - 2)")
        }
    }

    /// Der breite Suchlauf. Die handverlesenen Faelle haben zwei Fehler gefunden, beide in der
    /// Art, wie Swift mit Zeichenketten umgeht statt im Algorithmus. Solche Fehler treffen
    /// einzelne Token des Wortschatzes, deshalb sucht dieser Korpus den Wortschatz selbst ab:
    /// jedes Token mit einem kanonisch gleichwertigen Gegenstueck und jedes, das mit U+FEFF
    /// oder einer kombinierenden Marke beginnt.
    func testFuzzCorpus() throws {
        try XCTSkipUnless(LayaFixtures.hasFuzz(), "tokenizer-fuzz.json fehlt")
        let tokenizer = try LayaFixtures.sharedTokenizer()
        let (traps, cases) = try LayaFixtures.fuzzCases()
        XCTAssertGreaterThan(cases.count, 10_000, "der Korpus ist zu klein, um etwas zu belegen")
        var failures: [String] = []
        for testCase in cases where tokenizer.encode(testCase.text) != testCase.ids {
            if failures.count < 8 {
                failures.append("\(testCase.text.unicodeScalars.map { String(format: "U+%04X", $0.value) }.joined(separator: " ")): "
                    + "erwartet \(testCase.ids), erhalten \(tokenizer.encode(testCase.text))")
            } else {
                failures.append("")
            }
        }
        XCTAssertTrue(failures.isEmpty,
                      "\(failures.count) von \(cases.count) Faellen weichen ab "
                          + "(\(traps) davon aus dem Wortschatz):\n"
                          + failures.prefix(8).joined(separator: "\n"))
    }

    func testEncoderMatchesReference() throws {
        try XCTSkipUnless(LayaFixtures.hasGolden(), "Golden/laya fehlt")
        let runtime = try LayaFixtures.runtime()
        let encoder = LayaEncoder(tokenizer: try LayaFixtures.sharedTokenizer(),
                                  maxLength: runtime.maxLength,
                                  headMaxLength: runtime.headMaxLength,
                                  clsID: runtime.clsID, sepID: runtime.sepID,
                                  maskID: runtime.maskID, padID: runtime.padID,
                                  maskToken: runtime.maskToken)
        let records = try LayaFixtures.records()
        XCTAssertFalse(records.isEmpty)
        for record in records {
            let encoded = try encoder.encode(state: record.state, question: record.question)
            XCTAssertEqual(encoded.qtype, record.qtype, record.questionID)
            XCTAssertEqual(encoded.options, record.options,
                           "Optionstexte weichen ab bei \(record.questionID)")
            XCTAssertEqual(encoded.markers, record.markers,
                           "Markerpositionen weichen ab bei \(record.questionID)")
            XCTAssertEqual(encoded.ids, record.ids,
                           "Sequenz weicht ab bei \(record.questionID)")
        }
    }

    /// Mehr Optionen, als vor max_len passen: laya bricht ab, statt die Frage mit weniger Optionen
    /// zu beantworten. Der Port muss an genau derselben Stelle ablehnen, mit derselben Zahl.
    func testOptionOverflowIsRefusedLikeLaya() throws {
        try XCTSkipUnless(LayaFixtures.hasGolden(), "Golden/laya fehlt")
        guard let overflow = try LayaFixtures.overflow() else {
            throw XCTSkip("records.json ohne Überlaufanfrage, dump_laya_golden.py neu laufen lassen")
        }
        let runtime = try LayaFixtures.runtime()
        let encoder = LayaEncoder(tokenizer: try LayaFixtures.sharedTokenizer(),
                                  maxLength: runtime.maxLength, headMaxLength: runtime.headMaxLength,
                                  clsID: runtime.clsID, sepID: runtime.sepID, maskID: runtime.maskID,
                                  padID: runtime.padID, maskToken: runtime.maskToken)
        XCTAssertThrowsError(try encoder.encode(state: overflow.state, question: overflow.question)) { error in
            guard case let JevError.optionsDoNotFit(count, fitting, length)? = error as? JevError else {
                return XCTFail("falscher Fehler: \(error)")
            }
            XCTAssertEqual(count, overflow.options)
            XCTAssertEqual(fitting, overflow.fitting)
            XCTAssertEqual(length, runtime.maxLength)
        }
    }

    /// laya gibt in `legend` die Stufen so zurück, wie sie kamen: Zahlen bleiben Zahlen. Der Port
    /// führt sie deshalb neben der Textform als Rohwerte mit.
    func testScoreLegendKeepsRawValues() throws {
        try XCTSkipUnless(LayaFixtures.hasGolden(), "Golden/laya fehlt")
        let runtime = try LayaFixtures.runtime()
        let encoder = LayaEncoder(tokenizer: try LayaFixtures.sharedTokenizer(),
                                  maxLength: runtime.maxLength, headMaxLength: runtime.headMaxLength,
                                  clsID: runtime.clsID, sepID: runtime.sepID, maskID: runtime.maskID,
                                  padID: runtime.padID, maskToken: runtime.maskToken)
        let levels: [JevValue] = [.int(1), .double(2.5), .object([("label", .string("hoch"))]), .string("x")]
        let encoded = try encoder.encode(state: .string("s"), question: .score(instructions: "Wie?", levels: levels))
        XCTAssertEqual(encoded.legendValues, levels)
        XCTAssertEqual(encoded.legend, ["1", "2.5", #"{"label": "hoch"}"#, "x"])
    }

    /// Die Temperaturtabelle wird ueber einen Eimer aus Fragetyp und Optionszahl nachgeschlagen.
    /// Greift der falsche Eimer, sind alle Wahrscheinlichkeiten leise daneben.
    /// Das Maskentoken steht im Paket und darf nicht fest verdrahtet sein. `build_sequence`
    /// ersetzt es in Anweisung, Optionen und Zustand durch ein Leerzeichen, und es heisst je
    /// nach Rueckgrat "[MASK]" oder "<mask>". Wer eines davon festschreibt, schleust auf dem
    /// anderen Checkpoint ein Markertoken in den Zustand.
    func testMaskTokenComesFromTheModel() throws {
        try XCTSkipUnless(LayaFixtures.hasGolden(), "Golden/laya fehlt")
        guard let modelURL = LayaFixtures.modelURL() else {
            throw XCTSkip("kein laya-Modell unter Models/")
        }
        let runtime = try LayaRuntime(modelURL: modelURL, tokenizerURL: LayaFixtures.tokenizerURL)
        let expected = try LayaFixtures.runtime().maskToken
        XCTAssertEqual(runtime.metadata.maskToken, expected)
        XCTAssertEqual(runtime.encoder.maskToken, expected)
        XCTAssertEqual(expected, LayaFixtures.checkpoint == "multilingual" ? "<mask>" : "[MASK]")
    }

    func testTemperatureBuckets() throws {
        XCTAssertEqual(LayaSystemOne.temperatureBucket(qtype: 0, options: 2), "choice:2")
        XCTAssertEqual(LayaSystemOne.temperatureBucket(qtype: 0, options: 5), "choice:3-5")
        XCTAssertEqual(LayaSystemOne.temperatureBucket(qtype: 1, options: 6), "score:6-10")
        XCTAssertEqual(LayaSystemOne.temperatureBucket(qtype: 1, options: 10), "score:6-10")
        XCTAssertEqual(LayaSystemOne.temperatureBucket(qtype: 2, options: 11), "noul:11+")
    }

    /// Die Nachverarbeitung allein, mit den Logits aus der Referenz statt aus Core ML. Weicht
    /// hier etwas ab, liegt es nicht am Modell.
    func testPostProcessingMatchesReference() throws {
        try XCTSkipUnless(LayaFixtures.hasGolden(), "Golden/laya fehlt")
        let runtime = try LayaFixtures.runtime()
        let metadata = LayaRuntime.Metadata(
            checkpoint: "english", sequenceLength: runtime.maxLength,
            sequenceLengths: [runtime.maxLength], maxLength: runtime.maxLength,
            maxOptions: runtime.maxLength,
            headMaxLength: runtime.headMaxLength, padID: runtime.padID, clsID: runtime.clsID,
            sepID: runtime.sepID, maskID: runtime.maskID, maskToken: runtime.maskToken,
            temperature: runtime.temperature, temperatureByOptions: runtime.temperatureByOptions)
        let encoder = LayaEncoder(tokenizer: try LayaFixtures.sharedTokenizer(),
                                  maxLength: runtime.maxLength,
                                  headMaxLength: runtime.headMaxLength,
                                  clsID: runtime.clsID, sepID: runtime.sepID,
                                  maskID: runtime.maskID, padID: runtime.padID,
                                  maskToken: runtime.maskToken)
        for record in try LayaFixtures.records() {
            let encoded = try encoder.encode(state: record.state, question: record.question)
            let answer = LayaSystemOne.answer(
                from: Array(record.logits.prefix(encoded.markers.count)),
                act: record.actLogits, encoded: encoded, metadata: metadata)
            try assertMatches(answer, record: record, tolerance: 1e-4)
        }
    }

    func testCoreMLMatchesReference() async throws {
        try XCTSkipUnless(LayaFixtures.hasGolden(), "Golden/laya fehlt")
        guard let modelURL = LayaFixtures.modelURL() else {
            throw XCTSkip("kein laya-Modell unter Models/")
        }
        let system = try LayaSystemOne(modelURL: modelURL, tokenizerURL: LayaFixtures.tokenizerURL)
        let records = try LayaFixtures.records()
        var worstProbability = 0.0
        for record in records {
            let answer = try await system.ask(record.question, about: record.state)
            let reference = try referenceProbabilities(record)
            XCTAssertEqual(answer.probabilities.count, reference.count, record.questionID)
            for (got, want) in zip(answer.probabilities, reference) {
                worstProbability = max(worstProbability, abs(got - want))
            }
            // Die gewaehlte Option darf nicht kippen. Das ist die Zahl, an der eine Anwendung haengt.
            if case let .choice(choice) = answer.answer {
                XCTAssertEqual(choice.choice,
                               Fixtures.string(Fixtures.field(record.answer, "choice")),
                               "Entscheidung gekippt bei \(record.questionID)")
            }
        }
        XCTAssertLessThan(worstProbability, 0.02,
                          "groesster Abstand der Wahrscheinlichkeiten zu laya: \(worstProbability)")
        print("laya Core ML vs PyTorch: max|dp| = \(worstProbability)")
    }

    /// Mehrere Fragen gehen als Stapel an Core ML. Das muss dasselbe ergeben wie jede Frage
    /// einzeln, sonst stimmt die Zuordnung von Ergebnis zu Frage nicht. Der Stapel ist der
    /// einzige Zweig der Laufzeit, den die Einzelvergleiche nicht beruehren.
    func testBatchMatchesSingleQuestions() async throws {
        try XCTSkipUnless(LayaFixtures.hasGolden(), "Golden/laya fehlt")
        guard let modelURL = LayaFixtures.modelURL() else {
            throw XCTSkip("kein laya-Modell unter Models/")
        }
        let system = try LayaSystemOne(modelURL: modelURL, tokenizerURL: LayaFixtures.tokenizerURL)
        let records = try LayaFixtures.records()
        // Alle Referenzfragen auf einem Zustand, gemischte Typen und Optionszahlen, damit eine
        // vertauschte Zuordnung auch an der Laenge der Wahrscheinlichkeiten auffiele.
        let state = records[0].state
        let questions = records.enumerated().map { ("q\($0.offset)", $0.element.question) }
        XCTAssertGreaterThan(questions.count, 3)

        let batched = try await system.ask(questions, about: state)
        XCTAssertEqual(batched.map(\.id), questions.map(\.0))
        for (entry, question) in zip(batched, questions) {
            let single = try await system.ask(question.1, about: state)
            XCTAssertEqual(entry.answer.probabilities.count, single.probabilities.count, entry.id)
            for (a, b) in zip(entry.answer.probabilities, single.probabilities) {
                XCTAssertEqual(a, b, accuracy: 1e-6, "Stapel und Einzellauf weichen ab bei \(entry.id)")
            }
            XCTAssertEqual(entry.answer.actProbability, single.actProbability, accuracy: 1e-6, entry.id)
        }
    }

    /// Die Neural Engine rechnet dieses Modell falsch, deshalb muss sie ohne ausdrueckliches
    /// Zutun gar nicht erst anlaufen.
    func testNeuralEngineIsRefused() throws {
        guard let modelURL = LayaFixtures.modelURL() else {
            throw XCTSkip("kein laya-Modell unter Models/")
        }
        XCTAssertThrowsError(try LayaRuntime(
            modelURL: modelURL, tokenizerURL: LayaFixtures.tokenizerURL,
            configuration: .init(computeUnits: .cpuAndNeuralEngine)))
    }

    // MARK: - Hilfen

    private func referenceProbabilities(_ record: LayaParityTests.RecordAlias) throws -> [Double] {
        let runtime = try LayaFixtures.runtime()
        let k = record.markers.count
        let bucket = LayaSystemOne.temperatureBucket(qtype: record.qtype, options: k)
        let fallback = runtime.temperature.indices.contains(Int(record.qtype))
            ? runtime.temperature[Int(record.qtype)] : 1.0
        return LayaSystemOne.probabilities(Array(record.logits.prefix(k)),
                                           temperature: runtime.temperatureByOptions[bucket] ?? fallback)
    }

    private func assertMatches(_ answer: LayaAnswer, record: LayaParityTests.RecordAlias,
                               tolerance: Double) throws {
        let want = record.answer
        XCTAssertEqual(answer.confidence,
                       Fixtures.double(Fixtures.field(want, "confidence")) ?? -1,
                       accuracy: tolerance, "Konfidenz bei \(record.questionID)")
        let action = Fixtures.field(want, "action") ?? .null
        XCTAssertEqual(answer.actProbability,
                       Fixtures.double(Fixtures.field(action, "act_probability")) ?? -1,
                       accuracy: tolerance, "act_probability bei \(record.questionID)")
        switch answer.answer {
        case let .choice(choice):
            XCTAssertEqual(choice.choice, Fixtures.string(Fixtures.field(want, "choice")),
                           record.questionID)
        case let .score(score):
            XCTAssertEqual(score.score, Fixtures.double(Fixtures.field(want, "score")) ?? -1,
                           accuracy: tolerance, record.questionID)
            // Die Legende sind die Stufen selbst, nicht die daraus gebauten Optionstexte.
            // laya gibt {"0": "neutral"} zurueck und nicht {"0": "level 0: neutral"}.
            if case let .object(pairs)? = Fixtures.field(want, "legend") {
                XCTAssertEqual(score.legend, pairs.map { Fixtures.string($0.value) ?? "" },
                               "Legende weicht ab bei \(record.questionID)")
                XCTAssertEqual(pairs.map(\.key), score.legend.indices.map(String.init),
                               "Legendenschluessel weichen ab bei \(record.questionID)")
            } else {
                XCTFail("Referenz ohne Legende bei \(record.questionID)")
            }
        case let .noul(noul):
            XCTAssertEqual(noul.noul, Fixtures.double(Fixtures.field(want, "noul")) ?? -1,
                           accuracy: tolerance, record.questionID)
        }
    }

    typealias RecordAlias = LayaFixtures.Record
}
