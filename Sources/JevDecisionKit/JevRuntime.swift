import CoreML
import Foundation

/// Führt das exportierte Kev-Modell aus: Encoding hinein, Logits heraus.
///
/// Zwei Modellverträge werden unterstützt und am Namen der Eingaben erkannt:
///
/// - `singleQuestion` (Phase 1): additive Maske als Eingabe `[1,1,L,L]`, Ausgabe `[K]`.
/// - `fanOut` (Phase 2): `segment_ids [1,L]`, die Maske entsteht im Graphen, Ausgabe `[Q,K]`.
///
/// Der Actor hält die Eingabepuffer über Aufrufe hinweg. Unter Phase 1 ist die Maske allein
/// eine Million Byte; sie bei jeder Entscheidung neu zu allozieren wäre Verschwendung.
public actor JevRuntime {
    public enum Contract: String, Sendable {
        /// Eine Frage pro Inferenz, Maske kommt von außen.
        case singleQuestion
        /// Mehrere Fragen in einer Sequenz, Maske wird im Graphen gebaut.
        case fanOut
    }

    public struct Configuration: Sendable {
        public var padID: Int32
        /// Additiver Wert für gesperrte Attention-Felder. fp16-sicher gewählt.
        public var maskNegative: Float
        public var computeUnits: MLComputeUnits
        public var maxStateTokens: Int
        public var maxBranchTokens: Int
        public var strictState: Bool

        public init(padID: Int32 = 151_643,
                    maskNegative: Float = -1e4,
                    computeUnits: MLComputeUnits = .all,
                    maxStateTokens: Int = KevBudget.state,
                    maxBranchTokens: Int = KevBudget.branch,
                    strictState: Bool = true) {
            self.padID = padID
            self.maskNegative = maskNegative
            self.computeUnits = computeUnits
            self.maxStateTokens = maxStateTokens
            self.maxBranchTokens = maxBranchTokens
            self.strictState = strictState
        }
    }

    public nonisolated let configuration: Configuration
    public nonisolated let encoder: JevEncoder
    public nonisolated let contract: Contract
    /// Länge der Sequenz, für die das Modell exportiert wurde.
    public nonisolated let sequenceLength: Int
    /// Zahl der Optionen je Frage, für die das Modell exportiert wurde.
    public nonisolated let maxOptions: Int
    /// Zahl der Fragen, die ein Durchlauf beantwortet. Unter Phase 1 immer 1.
    public nonisolated let maxQuestions: Int
    /// Dateiname des Exports ohne Endung, etwa `Kev06B-Q4-fp16`. Für `/api/info` und Protokolle.
    public nonisolated let modelName: String

    private let model: MLModel
    private let inputIDs: MLMultiArray
    private let positionIDs: MLMultiArray
    private let segmentIDs: MLMultiArray?
    private let attentionMask: MLMultiArray?
    private let decideIndex: MLMultiArray
    private let optionIndices: MLMultiArray
    /// Länge, für die die Maske zuletzt gefüllt wurde. Nur Phase 1.
    private var maskLength = -1

    public init(modelURL: URL, tokenizer: Qwen3Tokenizer, configuration: Configuration = .init()) throws {
        self.modelName = modelURL.deletingPathExtension().lastPathComponent
        self.configuration = configuration
        self.encoder = try JevEncoder(tokenizer: tokenizer,
                                      maxStateTokens: configuration.maxStateTokens,
                                      maxBranchTokens: configuration.maxBranchTokens,
                                      strictState: configuration.strictState)

        let mlConfiguration = MLModelConfiguration()
        mlConfiguration.computeUnits = configuration.computeUnits
        let compiled = try CompiledModel.url(for: modelURL)
        self.model = try MLModel(contentsOf: compiled, configuration: mlConfiguration)

        let inputs = model.modelDescription.inputDescriptionsByName
        func shape(_ name: String) throws -> [Int] {
            guard let c = inputs[name]?.multiArrayConstraint else {
                throw JevError.modelOutput("Eingabe \(name) fehlt")
            }
            return c.shape.map(\.intValue)
        }
        guard model.modelDescription.outputDescriptionsByName["logits"] != nil else {
            throw JevError.modelOutput("Ausgabe logits fehlt")
        }

        // Der Vertrag wird am Modell abgelesen, nicht konfiguriert. Ein stillschweigend
        // anderes Shape-Budget würde sich sonst erst als falsche Entscheidung zeigen.
        let ids = try shape("input_ids")
        guard ids.count == 2, ids[0] == 1 else {
            throw JevError.modelOutput("input_ids hat Shape \(ids), erwartet [1, L]")
        }
        let L = ids[1]
        guard try shape("position_ids") == [1, L] else {
            throw JevError.modelOutput("position_ids passt nicht zu input_ids")
        }

        if inputs["segment_ids"] != nil {
            guard try shape("segment_ids") == [1, L] else {
                throw JevError.modelOutput("segment_ids passt nicht zu input_ids")
            }
            let opt = try shape("option_indices")
            guard opt.count == 2, try shape("decide_index") == [opt[0]] else {
                throw JevError.modelOutput("decide_index und option_indices passen nicht zusammen")
            }
            self.contract = .fanOut
            self.maxQuestions = opt[0]
            self.maxOptions = opt[1]
        } else {
            guard try shape("attention_mask") == [1, 1, L, L], try shape("decide_index") == [1] else {
                throw JevError.modelOutput("Phase-1-Vertrag nicht erfüllt")
            }
            let opt = try shape("option_indices")
            guard opt.count == 1 else { throw JevError.modelOutput("option_indices hat Shape \(opt)") }
            self.contract = .singleQuestion
            self.maxQuestions = 1
            self.maxOptions = opt[0]
        }
        self.sequenceLength = L

        self.inputIDs = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .int32)
        self.positionIDs = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .int32)
        switch contract {
        case .fanOut:
            self.segmentIDs = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .int32)
            self.attentionMask = nil
            self.decideIndex = try MLMultiArray(shape: [NSNumber(value: maxQuestions)], dataType: .int32)
            self.optionIndices = try MLMultiArray(
                shape: [NSNumber(value: maxQuestions), NSNumber(value: maxOptions)], dataType: .int32)
        case .singleQuestion:
            self.segmentIDs = nil
            self.attentionMask = try MLMultiArray(
                shape: [1, 1, NSNumber(value: L), NSNumber(value: L)], dataType: .float32)
            self.decideIndex = try MLMultiArray(shape: [1], dataType: .int32)
            self.optionIndices = try MLMultiArray(shape: [NSNumber(value: maxOptions)], dataType: .int32)
        }
    }

    // MARK: - Inferenz

    /// Logits je Frage. Die äußere Liste folgt der Reihenfolge der Fragen im Encoding.
    public func logits(for encoding: JevEncoding, optionCounts: [Int]) throws -> [[Double]] {
        guard encoding.count <= sequenceLength else {
            throw JevError.sequenceTooLong(tokens: encoding.count, limit: sequenceLength)
        }
        guard encoding.questions.count == optionCounts.count else {
            throw JevError.modelOutput("Zahl der Fragen passt nicht zu den Optionszahlen")
        }
        guard encoding.questions.count <= maxQuestions else {
            throw JevError.tooManyQuestions(count: encoding.questions.count, limit: maxQuestions)
        }
        for count in optionCounts where count > maxOptions {
            throw JevError.tooManyOptions(count: count, limit: maxOptions)
        }

        fill(inputIDs, with: encoding.ids, padding: configuration.padID)
        fill(positionIDs, with: encoding.positionIDs, padding: 0)

        var features: [String: MLMultiArray] = [
            "input_ids": inputIDs,
            "position_ids": positionIDs,
            "decide_index": decideIndex,
            "option_indices": optionIndices,
        ]

        switch contract {
        case .fanOut:
            guard let segmentIDs else { throw JevError.modelOutput("segment_ids fehlt") }
            fill(segmentIDs, with: encoding.segmentIDs, padding: -1)
            features["segment_ids"] = segmentIDs
        case .singleQuestion:
            guard let attentionMask else { throw JevError.modelOutput("attention_mask fehlt") }
            fillCausalMask(attentionMask, validTokens: encoding.count)
            features["attention_mask"] = attentionMask
        }

        let questions = encoding.questions
        decideIndex.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, strides in
            let step = strides.last ?? 1
            for q in 0 ..< maxQuestions {
                // Ungenutzte Frageplätze wiederholen Frage 0; ihre Logits werden verworfen.
                buffer[q * step] = Int32(questions[q < questions.count ? q : 0].decideIndex)
            }
        }
        let options = maxOptions
        optionIndices.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, strides in
            let rowStride = strides.count > 1 ? strides[0] : options
            let columnStride = strides.last ?? 1
            for q in 0 ..< maxQuestions {
                let ends = questions[q < questions.count ? q : 0].optionEnds
                for k in 0 ..< options {
                    buffer[q * rowStride + k * columnStride] =
                        Int32(k < ends.count ? ends[k] : ends[ends.count - 1])
                }
            }
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        let output = try model.prediction(from: provider)
        guard let raw = output.featureValue(for: "logits")?.multiArrayValue else {
            throw JevError.modelOutput("logits fehlen in der Antwort")
        }
        lastFlatLogits = try read(raw)
        return optionCounts.enumerated().map { index, count in
            Array(lastFlatLogits[(index * options) ..< (index * options + count)])
        }
    }

    /// Rohe Ausgabe des letzten Aufrufs, [Q*K] beziehungsweise [K]. Für Diagnose und Tests.
    public private(set) var lastFlatLogits: [Double] = []

    /// Kompletter Weg von Text zu Wahrscheinlichkeiten, eine Frage.
    public func probabilities(state: String, instructions: String, options: [String]) throws -> [Double] {
        try probabilities(state: state,
                          questions: [JevEncoder.Prompt(instructions: instructions, options: options)])[0]
    }

    /// Mehrere Fragen auf demselben Zustand. Unter Phase 2 ein einziger Durchlauf.
    public func probabilities(state: String, questions: [JevEncoder.Prompt]) throws -> [[Double]] {
        try run(state: state, questions: questions).probabilities
    }

    /// Wahrscheinlichkeiten samt Verbrauchszahlen, wie sie die TypeSafe-Antwort ausweist.
    public func run(state: String, questions: [JevEncoder.Prompt]) throws
        -> (probabilities: [[Double]], encoding: JevEncoding, passes: Int, latencyMilliseconds: Double) {
        let started = Date()
        let counts = questions.map(\.options.count)

        if contract == .fanOut || questions.count == 1 {
            let encoding = try encoder.encode(state: state, questions: questions)
            guard questions.count <= maxQuestions else {
                throw JevError.tooManyQuestions(count: questions.count, limit: maxQuestions)
            }
            let raw = try logits(for: encoding, optionCounts: counts)
            return (raw.map(JevRuntime.softmax), encoding, 1, Date().timeIntervalSince(started) * 1000)
        }

        // Phase 1 kann nur eine Frage je Durchlauf. Das Ergebnis ist dasselbe, weil die
        // block-kausale Maske die Zweige ohnehin voneinander trennt; es kostet nur mehr Zeit.
        var probabilities: [[Double]] = []
        for question in questions {
            let encoding = try encoder.encode(state: state, questions: [question])
            probabilities.append(JevRuntime.softmax(
                try logits(for: encoding, optionCounts: [question.options.count])[0]))
        }
        let full = try encoder.encode(state: state, questions: questions)
        return (probabilities, full, questions.count, Date().timeIntervalSince(started) * 1000)
    }

    public func encode(state: String, instructions: String, options: [String]) throws -> JevEncoding {
        try encoder.encode(state: state, instructions: instructions, options: options)
    }

    public func encode(state: String, questions: [JevEncoder.Prompt]) throws -> JevEncoding {
        try encoder.encode(state: state, questions: questions)
    }

    /// Zahl der Eingabe-Token einer Anfrage, ohne sie zu rechnen.
    public func inputTokens(state: String, questions: [JevEncoder.Prompt]) throws -> Int {
        try encoder.encode(state: state, questions: questions).count
    }

    /// Ein Leerlauf durch den Graphen, damit der erste echte Aufruf nicht die Kernel-Kompilierung
    /// von GPU oder Neural Engine mitbezahlt. Ohne das kostet die erste Entscheidung ein Vielfaches.
    @discardableResult
    public func warmUp() throws -> Double {
        let started = Date()
        let encoding = try encoder.encode(state: "warm", instructions: "warm", options: ["a", "b"])
        _ = try logits(for: encoding, optionCounts: [2])
        maskLength = -1
        return Date().timeIntervalSince(started) * 1000
    }

    // MARK: - Puffer

    private func fill(_ array: MLMultiArray, with values: [Int32], padding: Int32) {
        let length = sequenceLength
        array.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, strides in
            guard let base = buffer.baseAddress else { return }
            // Core ML darf Zeilen aufgefuellt ablegen; bei [1,L] zaehlt nur der letzte Schritt.
            let step = strides.last ?? 1
            if step == 1 {
                base.update(repeating: padding, count: length)
                values.withUnsafeBufferPointer { source in
                    guard let from = source.baseAddress else { return }
                    base.update(from: from, count: min(values.count, length))
                }
            } else {
                for i in 0 ..< length {
                    base[i * step] = i < values.count ? values[i] : padding
                }
            }
        }
    }

    private func fillCausalMask(_ array: MLMultiArray, validTokens n: Int) {
        guard maskLength != n else { return }
        let L = sequenceLength
        let negative = configuration.maskNegative
        array.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, strides in
            guard let base = buffer.baseAddress else { return }
            let rowStride = strides.count >= 2 ? strides[strides.count - 2] : L
            base.update(repeating: negative, count: rowStride * L)
            for i in 0 ..< n {
                base.advanced(by: i * rowStride).update(repeating: 0, count: i + 1)
            }
            // Padding-Zeilen behalten ihre Diagonale. Eine vollständig maskierte Softmax-Zeile
            // ergibt NaN, und das liefe über die Keys in die echten Token zurück.
            for i in n ..< L {
                base[i * rowStride + i] = 0
            }
        }
        maskLength = n
    }

    /// Liest die Ausgabe in logischer Reihenfolge aus.
    ///
    /// Core ML legt eine Ausgabe der Form [Q,K] nicht zwingend dicht ab: bei Q=4, K=8 kam hier
    /// ein Zeilenabstand von 16 zurueck. Wer die Strides ignoriert, liest Nullen und die Zeilen
    /// verschoben, und zwar ohne Fehlermeldung.
    private func read(_ array: MLMultiArray) throws -> [Double] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        switch array.dataType {
        case .float32:
            return array.withUnsafeBufferPointer(ofType: Float.self) { buffer in
                JevRuntime.gather(buffer, shape: shape, strides: strides) { Double($0) }
            }
        case .float16:
            return array.withUnsafeBufferPointer(ofType: Float16.self) { buffer in
                JevRuntime.gather(buffer, shape: shape, strides: strides) { Double($0) }
            }
        case .double:
            return array.withUnsafeBufferPointer(ofType: Double.self) { buffer in
                JevRuntime.gather(buffer, shape: shape, strides: strides) { $0 }
            }
        default:
            throw JevError.modelOutput("unerwarteter Datentyp der Logits: \(array.dataType.rawValue)")
        }
    }

    nonisolated static func gather<T>(_ buffer: UnsafeBufferPointer<T>,
                                      shape: [Int], strides: [Int],
                                      convert: (T) -> Double) -> [Double] {
        let total = shape.reduce(1, *)
        guard total > 0, shape.count == strides.count else { return [] }
        var out = [Double]()
        out.reserveCapacity(total)
        var index = [Int](repeating: 0, count: shape.count)
        for _ in 0 ..< total {
            var offset = 0
            for d in 0 ..< shape.count { offset += index[d] * strides[d] }
            out.append(convert(buffer[offset]))
            var d = shape.count - 1
            while d >= 0 {
                index[d] += 1
                if index[d] < shape[d] { break }
                index[d] = 0
                d -= 1
            }
        }
        return out
    }

    public nonisolated static func softmax(_ x: [Double]) -> [Double] {
        guard let m = x.max() else { return [] }
        let e = x.map { Foundation.exp($0 - m) }
        let s = e.reduce(0, +)
        return s > 0 ? e.map { $0 / s } : e
    }
}
