import CoreML
import Foundation

/// Eine laya-Antwort: die getypte Antwort selbst plus das, was laya zusätzlich ausweist.
public struct LayaAnswer: Sendable {
    public let answer: JevAnswer
    /// Normalisierte Entropie: 0 bei Gleichverteilung, 1 bei voller Sicherheit. Bei `noul`
    /// stattdessen `max(p, 1-p)`, so rechnet laya das auch.
    public let confidence: Double
    /// Der zweite Kopf: wie wahrscheinlich laya eine Handlung für angebracht hält.
    public let actProbability: Double
    public let probabilities: [Double]
    public let labels: [String]
    /// Bei `score` die Stufen als Rohwerte, wie laya sie in `legend` zurückgibt; sonst leer.
    public let legend: [JevValue]
    /// Länge der Sequenz, die das Modell gelesen hat. laya weist die Summe in `usage` aus.
    public let inputTokens: Int
}

/// Die Entscheidungsschnittstelle auf einem laya-Modell.
///
/// Dieselben Fragetypen und dieselben Antworttypen wie `SystemOne`, nur ein anderes Modell
/// darunter: ein bidirektionaler Encoder statt eines Decoders mit Pointer-Kopf. Wer beides
/// hat, kann je Frage wählen; was gleich bleiben muss, ist die Form der Antwort.
public final class LayaSystemOne: Sendable {
    public let runtime: LayaRuntime

    public init(modelURL: URL, tokenizerURL: URL,
                configuration: LayaRuntime.Configuration = .init()) throws {
        self.runtime = try LayaRuntime(modelURL: modelURL, tokenizerURL: tokenizerURL,
                                       configuration: configuration)
    }

    public init(runtime: LayaRuntime) {
        self.runtime = runtime
    }

    // MARK: - Bequeme Einstiege

    @discardableResult
    public func choice(state: JevValue, instruction: JevValue,
                       options: [(name: String, description: JevValue)]) async throws -> LayaAnswer {
        try await ask(.choice(instructions: instruction, criteria: options), about: state)
    }

    @discardableResult
    public func score(state: JevValue, instruction: JevValue, levels: [JevValue]) async throws -> LayaAnswer {
        try await ask(.score(instructions: instruction, levels: levels), about: state)
    }

    @discardableResult
    public func noul(state: JevValue, instruction: JevValue,
                     whenTrue: JevValue = .null, whenFalse: JevValue = .null) async throws -> LayaAnswer {
        try await ask(.noul(instructions: instruction, whenTrue: whenTrue, whenFalse: whenFalse),
                      about: state)
    }

    /// Eine Frage, ein Durchlauf.
    public func ask(_ question: JevQuestion, about state: JevValue) async throws -> LayaAnswer {
        let encoded = try runtime.encoder.encode(state: state, question: question)
        let raw = try await runtime.logits(for: encoded)
        return LayaSystemOne.answer(from: raw.logits, act: raw.act, encoded: encoded,
                                    metadata: runtime.metadata)
    }

    /// Mehrere Fragen auf demselben Zustand.
    ///
    /// Jede Frage hat ihre eigene Sequenz, denn Anweisung und Optionen stehen mit im Text. Ein
    /// gemeinsamer Durchlauf wie bei kev geht deshalb nicht. laya bündelt die Sequenzen zu einem
    /// Batch; hier laufen sie nacheinander, warum, steht bei `LayaRuntime.logits(for:)`.
    public func ask(_ questions: [(id: String, question: JevQuestion)],
                    about state: JevValue) async throws -> [(id: String, answer: LayaAnswer)] {
        guard !questions.isEmpty else { return [] }
        let encoded = try questions.map { try runtime.encoder.encode(state: state, question: $0.question) }
        let raw = try await runtime.logits(for: encoded)
        let metadata = runtime.metadata
        return zip(questions, zip(encoded, raw)).map { entry, pair in
            (entry.id, LayaSystemOne.answer(from: pair.1.logits, act: pair.1.act,
                                            encoded: pair.0, metadata: metadata))
        }
    }

    // MARK: - Nachverarbeitung

    /// Der Eimer, über den die Temperatur nachgeschlagen wird. `laya.common.temp_bucket`.
    public static func temperatureBucket(qtype: Int32, options: Int) -> String {
        let size = options <= 2 ? "2" : options <= 5 ? "3-5" : options <= 10 ? "6-10" : "11+"
        return "\(LayaEncoder.typeName(qtype)):\(size)"
    }

    /// Softmax nach Temperatur. Der Boden von 1e-3 steht so in laya und verhindert,
    /// dass eine kaputte Konfiguration durch Null teilt.
    public static func probabilities(_ logits: [Double], temperature: Double) -> [Double] {
        let scale = max(1e-3, temperature)
        return JevRuntime.softmax(logits.map { $0 / scale })
    }

    /// `1 - H(p) / log(k)`, auf [0,1] geklemmt. `laya.common.confidence_from_probs`.
    public static func entropyConfidence(_ p: [Double]) -> Double {
        guard p.count >= 2 else { return 1.0 }
        let entropy = -p.reduce(0.0) { $0 + $1 * Foundation.log(min(max($1, 1e-12), 1.0)) }
        return min(max(1.0 - entropy / Foundation.log(Double(p.count)), 0.0), 1.0)
    }

    static func answer(from logits: [Double], act: [Double],
                       encoded: LayaEncoder.Encoded,
                       metadata: LayaRuntime.Metadata) -> LayaAnswer {
        let bucket = temperatureBucket(qtype: encoded.qtype, options: logits.count)
        let fallback = metadata.temperature.indices.contains(Int(encoded.qtype))
            ? metadata.temperature[Int(encoded.qtype)] : 1.0
        let p = probabilities(logits, temperature: metadata.temperatureByOptions[bucket] ?? fallback)
        let actProbability = JevRuntime.softmax(act).first ?? 0.0

        switch encoded.qtype {
        case 0:
            let confidence = entropyConfidence(p)
            let pairs = zip(encoded.labels, p).map { (name: $0.0, probability: $0.1) }
            let pick = LayaSystemOne.argmax(p)
            return LayaAnswer(
                answer: .choice(ChoiceAnswer(choice: encoded.labels[pick], confidence: confidence,
                                             probabilities: pairs)),
                confidence: confidence, actProbability: actProbability,
                probabilities: p, labels: encoded.labels,
                legend: encoded.qtype == 1 ? encoded.legendValues : [],
                inputTokens: encoded.ids.count)
        case 1:
            let confidence = entropyConfidence(p)
            let expected = p.enumerated().reduce(0.0) { $0 + Double($1.offset) * $1.element }
            return LayaAnswer(
                answer: .score(ScoreAnswer(score: expected, legend: encoded.legend,
                                           probabilities: p, confidence: confidence)),
                confidence: confidence, actProbability: actProbability,
                probabilities: p, labels: encoded.labels,
                legend: encoded.qtype == 1 ? encoded.legendValues : [],
                inputTokens: encoded.ids.count)
        default:
            // laya misst die Sicherheit einer Ja/Nein-Frage nicht über die Entropie, sondern
            // als Abstand zur Mitte. Bei zwei Optionen ist das monoton dasselbe, aber die Zahl
            // ist eine andere, und die steht in der Antwort.
            let yes = p.count > 1 ? p[1] : 0.0
            return LayaAnswer(
                answer: .noul(NoulAnswer(noul: yes, probabilities: p)),
                confidence: max(yes, 1.0 - yes), actProbability: actProbability,
                probabilities: p, labels: encoded.labels,
                legend: encoded.qtype == 1 ? encoded.legendValues : [],
                inputTokens: encoded.ids.count)
        }
    }

    /// Erster Höchstwert, nicht letzter. `numpy.argmax` entscheidet Gleichstände nach links,
    /// Swifts `max(by:)` nach rechts.
    static func argmax(_ values: [Double]) -> Int {
        var best = 0
        for i in values.indices where values[i] > values[best] { best = i }
        return best
    }
}
