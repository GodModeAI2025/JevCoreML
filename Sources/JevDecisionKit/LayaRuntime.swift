import CoreML
import Foundation

/// Laufzeit für ein laya-Modell als Core ML.
///
/// Der Vertrag hat fünf Eingaben und zwei Ausgaben:
///
///     input_ids [1,L]  attention_mask [1,L]  marker_pos [K]  marker_mask [K]  qtype [1]
///     -> logits [K]    act_logits [2]
///
/// Alles, was Gewichte hat, rechnet der Graph. Draußen bleibt nur Konfiguration: die Temperatur
/// je Fragetyp und Optionszahl und die Entropiekonfidenz.
public actor LayaRuntime {
    public struct Configuration: Sendable {
        /// Rechenwerk. Die Voreinstellung ist nicht `.all`, und das ist Absicht.
        ///
        /// Gemessen an `Laya-EN-L512-K512-fp16` mit mehreren Längen, über 15 Referenzanfragen,
        /// Bericht in `Benchmarks/laya-backends-EN.json`: GPU 6,3 ms, max|dLogit| 2,5e-02 und
        /// keine gekippte Entscheidung; reine CPU 86 ms; Neural Engine ebenfalls richtig, aber
        /// 268 ms, also 40-mal langsamer als die GPU. Bei den Exporten mit fester Länge davor
        /// rechnete die Neural Engine falsch: 9 von 15 Entscheidungen gekippt, ohne Fehler und
        /// ohne Warnung. `.all` überlässt die Wahl dem Planer von Core ML, und der hat im Test
        /// zwischen 22 ms und 115 ms geschwankt. Also wird das Rechenwerk hier festgenagelt statt
        /// ausgewürfelt.
        ///
        /// `.cpuAndNeuralEngine` wird abgelehnt, `.all` wird ohne Meldung auf `.cpuAndGPU`
        /// heruntergesetzt: `.all` darf die Neural Engine benutzen, und das soll es nicht.
        public var computeUnits: MLComputeUnits
        /// Lässt `.cpuAndNeuralEngine` zu. Bei festen Exporten rechnete sie falsch, beim Paket mit
        /// mehreren Längen richtig, aber 40-mal langsamer als die GPU.
        public var allowNeuralEngine: Bool
        /// Welche der Eingabelängen des Pakets benutzt werden, nil für alle. Die größte ist immer
        /// dabei, sonst passte nicht jede Frage. Jede Länge kostet eine eigene Instanz, gemessen
        /// 25 bis 50 MB Speicher; siehe `logits(for:)`.
        public var sequenceLengths: [Int]?

        public init(computeUnits: MLComputeUnits = .cpuAndGPU, allowNeuralEngine: Bool = false,
                    sequenceLengths: [Int]? = nil) {
            self.computeUnits = computeUnits
            self.allowNeuralEngine = allowNeuralEngine
            self.sequenceLengths = sequenceLengths
        }
    }

    /// Was im Modellpaket über das Modell steht. Keine Beidatei, keine zweite Quelle.
    public struct Metadata: Sendable {
        public let checkpoint: String
        /// Die größte Eingabelänge des Pakets. Kürzer als `maxLength` ist sie nie.
        public let sequenceLength: Int
        /// Alle Eingabelängen, die das Paket annimmt, aufsteigend. Ältere Pakete haben nur eine.
        public let sequenceLengths: [Int]
        /// Länge, auf die laya kürzt (`cfg["max_len"]`). Höchstens `sequenceLength`; fehlt sie in
        /// älteren Paketen, gilt `sequenceLength`.
        public let maxLength: Int
        public let maxOptions: Int
        public let headMaxLength: Int
        public let padID: Int32
        public let clsID: Int32
        public let sepID: Int32
        public let maskID: Int32
        /// Der Maskentext, den `build_sequence` in Anweisung, Optionen und Zustand durch ein
        /// Leerzeichen ersetzt. "[MASK]" bei ModernBERT, "<mask>" bei mmBERT.
        public let maskToken: String
        /// Temperatur je Fragetyp, Index 0 choice, 1 score, 2 noul.
        public let temperature: [Double]
        /// Temperatur je Eimer aus Fragetyp und Optionszahl, etwa `choice:3-5`.
        public let temperatureByOptions: [String: Double]
    }

    nonisolated public let metadata: Metadata
    nonisolated public let encoder: LayaEncoder
    /// Die Längen, die diese Laufzeit benutzt: `Configuration.sequenceLengths` geschnitten mit
    /// denen des Pakets, die größte immer dabei.
    nonisolated public let activeLengths: [Int]
    private let compiledURL: URL
    private let mlConfiguration: MLModelConfiguration
    /// Eine Instanz je Länge, angelegt beim ersten Gebrauch. Siehe `logits(for:)`.
    private var instances: [Int: MLModel]
    private var sequenceBuffers: [Int: (ids: MLMultiArray, mask: MLMultiArray)] = [:]
    private let markerPos: MLMultiArray
    private let markerMask: MLMultiArray
    private let qtype: MLMultiArray

    public init(modelURL: URL, tokenizerURL: URL, configuration: Configuration = .init()) throws {
        var units = configuration.computeUnits
        if units == .cpuAndNeuralEngine && !configuration.allowNeuralEngine {
            throw JevError.modelFile(
                "die Neural Engine ist für dieses Modell die falsche Wahl: feste Exporte rechnete "
                    + "sie falsch, das Paket mit mehreren Längen 40-mal langsamer als die GPU; "
                    + "mit allowNeuralEngine: true lässt sich das überstimmen")
        }
        if units == .all && !configuration.allowNeuralEngine { units = .cpuAndGPU }

        let mlConfiguration = MLModelConfiguration()
        mlConfiguration.computeUnits = units
        let compiled = try CompiledModel.url(for: modelURL)
        let model = try MLModel(contentsOf: compiled, configuration: mlConfiguration)
        self.compiledURL = compiled
        self.mlConfiguration = mlConfiguration

        let description = model.modelDescription
        let meta = description.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        func number(_ key: String, _ fallback: Int) -> Int { Int(meta[key] ?? "") ?? fallback }

        guard let idsInput = description.inputDescriptionsByName["input_ids"]?.multiArrayConstraint,
              let markerInput = description.inputDescriptionsByName["marker_pos"]?.multiArrayConstraint,
              description.inputDescriptionsByName["marker_mask"] != nil,
              description.outputDescriptionsByName["act_logits"] != nil else {
            throw JevError.modelFile(
                "kein laya-Modell: erwartet werden input_ids, attention_mask, marker_pos, "
                    + "marker_mask, qtype und die Ausgaben logits und act_logits")
        }
        // Aufgezählte Formen, falls das Paket mehrere Längen annimmt; sonst die eine feste.
        var lengths = idsInput.shapeConstraint.enumeratedShapes.compactMap { $0.last?.intValue }
        if lengths.isEmpty { lengths = [idsInput.shape.last?.intValue ?? 0] }
        lengths = Array(Set(lengths)).sorted()
        let length = lengths.last ?? 0
        let options = markerInput.shape.last?.intValue ?? 0

        func doubles(_ key: String) -> [Double] {
            guard let raw = meta[key]?.data(using: .utf8),
                  let list = try? JSONSerialization.jsonObject(with: raw) as? [Any] else { return [] }
            return list.compactMap { ($0 as? NSNumber)?.doubleValue }
        }
        func table(_ key: String) -> [String: Double] {
            guard let raw = meta[key]?.data(using: .utf8),
                  let dict = try? JSONSerialization.jsonObject(with: raw) as? [String: Any] else { return [:] }
            return dict.compactMapValues { ($0 as? NSNumber)?.doubleValue }
        }
        var temperature = doubles("laya.temperature")
        if temperature.count != 3 { temperature = [1, 1, 1] }

        let maxLength = number("laya.max_len", length)
        guard maxLength <= length else {
            throw JevError.modelFile("laya.max_len \(maxLength) ist größer als die exportierte Länge \(length)")
        }
        self.metadata = Metadata(
            checkpoint: meta["laya.checkpoint"] ?? "unbekannt",
            sequenceLength: length,
            sequenceLengths: lengths,
            maxLength: maxLength,
            maxOptions: options,
            headMaxLength: number("laya.head_max_len", 192),
            padID: Int32(number("laya.pad_id", 1)),
            clsID: Int32(number("laya.cls_id", 50281)),
            sepID: Int32(number("laya.sep_id", 50282)),
            maskID: Int32(number("laya.mask_id", 50284)),
            maskToken: meta["laya.mask_token"] ?? "[MASK]",
            temperature: temperature,
            temperatureByOptions: table("laya.temperature_by_options"))

        // Tokenizer und Paket müssen zusammengehören. Beide laya-Rückgrate nehmen jede
        // Token-ID des anderen klaglos an: ModernBERT-IDs liegen alle unter 50368, mmBERT hat
        // 256000 Zeilen. Ein falscher Tokenizer rechnet also nicht falsch mit einem Fehler,
        // sondern still mit Unsinn. Die Probe: der Maskentext des Pakets muss im Tokenizer
        // genau die Masken-ID des Pakets ergeben.
        let tokenizer = try LayaRuntime.tokenizer(at: tokenizerURL)
        guard tokenizer.id(forToken: metadata.maskToken) == metadata.maskID else {
            throw JevError.modelFile(
                "\(tokenizerURL.lastPathComponent) passt nicht zu \(modelURL.lastPathComponent) "
                    + "(\(metadata.checkpoint)): \(metadata.maskToken) ergibt dort nicht die ID \(metadata.maskID)")
        }
        self.encoder = LayaEncoder(tokenizer: tokenizer,
                                   maxLength: maxLength,
                                   headMaxLength: metadata.headMaxLength,
                                   clsID: metadata.clsID,
                                   sepID: metadata.sepID,
                                   maskID: metadata.maskID,
                                   padID: metadata.padID,
                                   maskToken: metadata.maskToken)

        let wanted = Set(configuration.sequenceLengths ?? lengths)
        self.activeLengths = lengths.filter { wanted.contains($0) || $0 == length }
        // Die schon geladene Instanz bedient die größte Länge, das ist die Vorgabeform im Paket.
        self.instances = [length: model]
        self.markerPos = try MLMultiArray(shape: [NSNumber(value: options)], dataType: .int32)
        self.markerMask = try MLMultiArray(shape: [NSNumber(value: options)], dataType: .int32)
        self.qtype = try MLMultiArray(shape: [1], dataType: .int32)
    }

    /// Welcher Tokenizer zu der Datei gehört, entschieden an ihrem Inhalt statt am Dateinamen.
    ///
    /// Die drei laya-Checkpoints teilen sich einen Modellvertrag, aber nicht den Tokenizer:
    /// die beiden ModernBERT-Ableger lesen Byte-Level-BPE, der mehrsprachige auf mmBERT
    /// Metaspace-BPE mit Byte-Rückfall. Das ist kein Schalter, den der Aufrufer setzen soll.
    public static func tokenizer(at url: URL) throws -> any TextTokenizer {
        let data = try Data(contentsOf: url)
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let type = ((root?["pre_tokenizer"] as? [String: Any])?["type"] as? String) ?? ""
        if type == "Metaspace" { return try MetaspaceTokenizer(contentsOf: url) }
        return try HFTokenizer(contentsOf: url)
    }

    /// Die kürzeste Länge, in die `tokens` passen. laya rechnet jede Frage nur so lang, wie sie
    /// ist; aufgefüllt wird hier nur bis zur nächsten Länge des Pakets.
    static func length(for tokens: Int, in lengths: [Int]) -> Int? {
        lengths.first { $0 >= tokens }
    }

    /// Rohe Ausgaben des Modells für ein Encoding.
    ///
    /// Die Frage läuft in der kürzesten Länge, in die sie passt, und zwar über eine eigene
    /// Instanz je Länge. Eine einzige Instanz, die zwischen den Längen wechselt, wäre die
    /// naheliegende Lösung und ist gemessen unbrauchbar: Core ML bereitet den Graphen auf der
    /// GPU bei jedem Wechsel neu vor, das kostet rund 450 ms statt 8 bis 12 ms, und keiner der
    /// Hinweise in `MLOptimizationHints` ändert daran etwas. Jede Instanz sieht deshalb nur ihre
    /// eine Länge. Das kostet wenig: beim englischen Checkpoint wuchs der Speicher mit drei
    /// Längen statt einer von 245 auf 320 MB, und die gelöschten, offenen Temp-Dateien von
    /// Core ML blieben bei 0,85 GiB, denn die legt es einmal je Paket an, nicht je Instanz.
    public func logits(for encoded: LayaEncoder.Encoded) throws -> (logits: [Double], act: [Double]) {
        let options = metadata.maxOptions
        guard let length = LayaRuntime.length(for: encoded.ids.count, in: activeLengths) else {
            throw JevError.sequenceTooLong(tokens: encoded.ids.count, limit: metadata.sequenceLength)
        }
        guard encoded.markers.count <= options else {
            throw JevError.tooManyOptions(count: encoded.markers.count, limit: options)
        }

        let (model, inputIDs, attentionMask) = try prepared(length)
        fill(inputIDs, count: length) { i in i < encoded.ids.count ? encoded.ids[i] : metadata.padID }
        fill(attentionMask, count: length) { i in i < encoded.ids.count ? 1 : 0 }
        // Ungenutzte Plätze zeigen auf 0 und werden über marker_mask ausgeblendet, genau wie
        // `marker_pos.clamp(min=0)` zusammen mit `masked_fill(~marker_mask, -1e4)` in laya.
        fill(markerPos, count: options) { i in i < encoded.markers.count ? encoded.markers[i] : 0 }
        fill(markerMask, count: options) { i in i < encoded.markers.count ? 1 : 0 }
        fill(qtype, count: 1) { _ in encoded.qtype }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputIDs, "attention_mask": attentionMask,
            "marker_pos": markerPos, "marker_mask": markerMask, "qtype": qtype,
        ])
        let output = try model.prediction(from: provider)
        guard let rawLogits = output.featureValue(for: "logits")?.multiArrayValue,
              let rawAct = output.featureValue(for: "act_logits")?.multiArrayValue else {
            throw JevError.modelOutput("logits oder act_logits fehlen in der Antwort")
        }
        let all = try read(rawLogits)
        return (Array(all.prefix(encoded.markers.count)), try read(rawAct))
    }

    /// Instanz und Eingabepuffer für eine Länge, beim ersten Gebrauch angelegt.
    private func prepared(_ length: Int) throws -> (MLModel, MLMultiArray, MLMultiArray) {
        let model: MLModel
        if let ready = instances[length] {
            model = ready
        } else {
            model = try MLModel(contentsOf: compiledURL, configuration: mlConfiguration)
            instances[length] = model
        }
        if let buffers = sequenceBuffers[length] { return (model, buffers.ids, buffers.mask) }
        let ids = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        sequenceBuffers[length] = (ids, mask)
        return (model, ids, mask)
    }

    /// Legt für jede aktive Länge die Instanz an und rechnet einmal, damit Core ML den Graphen
    /// für diese Länge vorbereitet. Danach wartet keine Anfrage mehr darauf.
    public func warmUp() throws {
        for length in activeLengths {
            let tokens = min(length, max(1, metadata.maxLength))
            var ids = [Int32](repeating: metadata.padID, count: tokens)
            ids[0] = metadata.clsID
            let probe = LayaEncoder.Encoded(ids: ids, markers: [0], qtype: 2, labels: ["x"],
                                            options: ["x"], legend: [], legendValues: [],
                                            stateTruncated: false)
            _ = try logits(for: probe)
        }
    }

    /// Mehrere Encodings, eines nach dem anderen.
    ///
    /// laya schickt mehrere Fragen als Batch durch einen Durchlauf. Die naheliegende Entsprechung
    /// wäre `MLModel.predictions(fromBatch:)`, und die gab es hier auch eine Zeit lang. Sie ist
    /// wieder draußen, aus zwei Gründen, beide gemessen:
    ///
    /// Sie bringt weniger, als man denkt. Bei vier Fragen blieb der Median gleich (110 gegen
    /// 115 ms), bei zehn sank er von 332,5 auf 267,7 ms. Core ML reiht den Stapel auf der GPU
    /// also weitgehend hintereinander auf und spart vor allem den Aufrufaufwand.
    ///
    /// Und sie bricht sporadisch ab, mit einer Objective-C-Ausnahme aus der E5-Laufzeit
    /// ("MPSGraph tensor shape is missing or has unexpected rank (expected 1, got 0)"). Die
    /// lässt sich in Swift nicht fangen, der Prozess endet mit Signal 6. Aufgetreten in einem
    /// von vier Läufen desselben Tests auf dem englischen Checkpoint, ohne dass sich an den
    /// Eingaben etwas geändert hätte. Ein Absturz, den kein `catch` erreicht, ist teurer als
    /// ein Fünftel bei zehn Fragen; schneller als laya ist der Port auch ohne Stapel.
    public func logits(for encodings: [LayaEncoder.Encoded]) throws -> [(logits: [Double], act: [Double])] {
        try encodings.map { try logits(for: $0) }
    }

    private func fill(_ array: MLMultiArray, count: Int, value: (Int) -> Int32) {
        array.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, strides in
            let step = strides.last ?? 1
            for i in 0 ..< count { buffer[i * step] = value(i) }
        }
    }

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
            throw JevError.modelOutput("unerwarteter Datentyp der Ausgabe: \(array.dataType.rawValue)")
        }
    }
}
