import Foundation
import JevDecisionKit

// Kleines Kommandozeilenwerkzeug, damit sich das Paket ohne Xcode-Projekt ausprobieren lässt.

struct Options {
    var engine = "kev"
    /// Jedes --model hängt an. Bei kev ergibt mehr als eines einen Pool.
    var modelList: [URL] = []
    var buckets = false
    var model: URL? { modelList.first }
    var tokenizer: URL?
    var models = URL(fileURLWithPath: "Models")
    var request: URL?
    var computeUnits = ""
    var repeats = 1
    var demo = false
    var serve = false
    var port: UInt16 = 8008
    var checkpoint: String?
    var language: String?
    var lengths: [Int]?
    var json: URL?
    var label = ""

    var modelURL: URL { model ?? models.appendingPathComponent("Kev06B-Q4-fp16.mlpackage") }

    /// Die drei Buckets 256, 512 und 1024, soweit sie unter --models liegen.
    ///
    /// L=3072 gehört nicht dazu: jeder geladene kev-Export belegt zur Laufzeit rund 14 GB Platte
    /// im Temp-Verzeichnis, und ein vierter Bucket für die seltenen Fälle mit mehr als 96 Optionen
    /// ist das nicht wert. Wer ihn braucht, gibt ihn zusätzlich mit --model an.
    static let bucketNames = ["Kev06B-L256-Q4-fp16", "Kev06B-Q4-fp16", "Kev06B-L1024-Q4K96-fp16"]

    var bucketURLs: [URL] {
        Options.bucketNames
            .map { models.appendingPathComponent("\($0).mlpackage") }
            .filter { FileManager.default.fileExists(atPath: $0.path) }
    }

    var tokenizerURL: URL { tokenizer ?? models.appendingPathComponent("tokenizer.json") }
}

/// Fehler in der Bedienung: Meldung auf stderr, Exit 2, wie bei `usage()`.
struct UsageError: Error, CustomStringConvertible {
    let description: String
}

func usage() -> Never {
    print("""
    jev - native Entscheidungen auf Core ML

      --engine kev|laya    welches Modell rechnet (Vorgabe kev)
      --models <pfad>      Verzeichnis mit Modellen und Tokenizern (Vorgabe Models)
      --model <pfad>       ein bestimmtes .mlpackage statt der Vorgabe; bei kev mehrfach angegeben
                           entsteht ein Pool, der je Anfrage den kleinsten passenden Export nimmt
      --buckets            kev-Pool aus Kev06B-L256-Q4, Kev06B-Q4 und Kev06B-L1024-Q4K96 unter
                           --models; weitere Exporte kommen mit --model dazu
      --tokenizer <pfad>   ein bestimmtes tokenizer.json statt der Vorgabe
      --request <pfad>     SystemOne-Anfrage als JSON
      --units cpu|gpu|all|ane
      --repeat <n>         Wiederholungen für die Zeitmessung
      --demo               eingebautes Beispiel: Skill-Routing
      --serve              startet POST /v1/systemone auf 127.0.0.1
      --port <n>           Port für --serve (Vorgabe 8008)

    nur mit --engine laya:
      --checkpoint <name>  english | multilingual | typed-decisions, sonst entscheidet die Schrift;
                           mit --model der Checkpoint des Pakets, sonst aus dem Dateinamen
      --lang <code>        Sprache vorgeben statt sie zu erkennen
                           (mit --serve: Vorgabe für Anfragen ohne task, lang oder Checkpoint)
      --lengths <liste>    nur diese Eingabelängen benutzen, etwa 128,512; die größte ist immer
                           dabei, jede weitere kostet 25 bis 50 MB Speicher
      --json <pfad>        Messwerte und Antworten als JSON schreiben
      --label <text>       Beschriftung für --json

    Die Vorgabe für --units ist all bei kev und gpu bei laya. Für laya ist das kein Geschmack:
    die Neural Engine rechnete die festen Exporte falsch und ist beim jetzigen Paket 40-mal
    langsamer als die GPU, siehe docs/laya-port.md.
    """)
    exit(2)
}

var options = Options()
var arguments = Array(CommandLine.arguments.dropFirst())
while let flag = arguments.first {
    arguments.removeFirst()
    func value() -> String {
        guard let v = arguments.first else { usage() }
        arguments.removeFirst()
        return v
    }
    switch flag {
    case "--engine": options.engine = value()
    case "--models": options.models = URL(fileURLWithPath: value())
    case "--checkpoint": options.checkpoint = value()
    case "--lang": options.language = value()
    case "--lengths": options.lengths = value().split(separator: ",").compactMap { Int($0) }
    case "--json": options.json = URL(fileURLWithPath: value())
    case "--label": options.label = value()
    case "--model": options.modelList.append(URL(fileURLWithPath: value()))
    case "--buckets": options.buckets = true
    case "--tokenizer": options.tokenizer = URL(fileURLWithPath: value())
    case "--request": options.request = URL(fileURLWithPath: value())
    case "--units": options.computeUnits = value()
    case "--repeat": options.repeats = Int(value()) ?? 1
    case "--demo": options.demo = true
    case "--serve": options.serve = true
    case "--port": options.port = UInt16(value()) ?? 8008
    case "-h", "--help": usage()
    default: usage()
    }
}
if options.request == nil && !options.demo && !options.serve { usage() }
guard ["kev", "laya"].contains(options.engine) else { usage() }

let units: MLComputeUnitsSelection = switch options.computeUnits {
case "cpu": .cpuOnly
case "gpu": .cpuAndGPU
case "ane": .cpuAndNeuralEngine
case "all": .all
default: options.engine == "laya" ? .cpuAndGPU : .all
}

@MainActor func timed<T>(_ body: () async throws -> T) async rethrows -> (T, Double) {
    let t0 = Date()
    let value = try await body()
    return (value, Date().timeIntervalSince(t0) * 1000)
}

let demoState: JevValue = .string("Erstelle mir bitte eine Tabelle mit den Umsaetzen der letzten "
                                  + "zwoelf Monate inklusive Quartalssummen.")
let demoSkills = [
    (name: "research", description: JevValue.string("Quellen recherchieren und zusammenfassen")),
    (name: "presentation", description: JevValue.string("Folien und Praesentationen bauen")),
    (name: "excel", description: JevValue.string("Tabellen, Kennzahlen und Berechnungen")),
    (name: "coding", description: JevValue.string("Software schreiben oder aendern")),
    (name: "none", description: JevValue.string("keiner dieser Skills passt")),
]
let demoInstruction = JevValue.string("Welcher Skill soll diese Aufgabe uebernehmen?")

func printProbabilities(_ entries: [(name: String, probability: Double)]) {
    let width = entries.map(\.name.count).max() ?? 0
    for entry in entries {
        let name = entry.name.padding(toLength: max(width, 8), withPad: " ", startingAt: 0)
        print(String(format: "%@  %.4f", name, entry.probability))
    }
}

// MARK: - laya

/// Welcher Checkpoint ein einzeln angegebenes Paket ist: aus --checkpoint, sonst aus dem
/// Dateinamen. Ob es stimmt, prüft `LayaRouted` beim Laden gegen die Metadaten des Pakets.
@MainActor func layaCheckpoint(for model: URL) throws -> LayaRouter.Checkpoint {
    if let name = options.checkpoint {
        guard let checkpoint = LayaRouter.Checkpoint.named(name) else {
            throw UsageError(description: "unbekannter Checkpoint \(name)")
        }
        return checkpoint
    }
    let file = model.lastPathComponent
    if file.hasPrefix("Laya-EN") { return .english }
    if file.hasPrefix("Laya-ML") { return .multilingual }
    if file.hasPrefix("Laya-TD") { return .typedDecisions }
    throw UsageError(description: "zu \(file) fehlt --checkpoint, der Dateiname verrät ihn nicht")
}

@MainActor func runLaya() async throws {
    if options.buckets { throw UsageError(description: "--buckets gibt es nur für --engine kev") }
    if options.modelList.count > 1 {
        throw UsageError(description: "--engine laya nimmt höchstens ein --model; "
            + "für mehrere Checkpoints --models mit dem Verzeichnis angeben")
    }
    let started = Date()
    let routed: LayaRouted
    var pinned = options.checkpoint
    if let model = options.model {
        let checkpoint = try layaCheckpoint(for: model)
        pinned = checkpoint.name
        // Der Tokenizer gehört zum Rückgrat: ModernBERT für english und typed-decisions,
        // mmBERT für multilingual. LayaRuntime prüft zusätzlich, ob er zum Paket passt.
        let family = checkpoint == .multilingual ? "multilingual" : "english"
        let tokenizer = options.tokenizer
            ?? options.models.appendingPathComponent("laya-tokenizer-\(family).json")
        routed = try LayaRouted(sources: [checkpoint: .init(modelURL: model, tokenizerURL: tokenizer)],
                                configuration: .init(computeUnits: units.value,
                                                     sequenceLengths: options.lengths),
                                fallback: checkpoint)
    } else {
        routed = try LayaRouted.discovering(in: options.models,
                                            configuration: .init(computeUnits: units.value,
                                                                 sequenceLengths: options.lengths))
    }
    print("laya bereit in \(String(format: "%.2f", Date().timeIntervalSince(started))) s, "
          + "verfügbar: \(routed.available.map(\.name).sorted().joined(separator: ", "))")

    if options.serve {
        // Was auf der Kommandozeile steht, gilt für Anfragen ohne eigenen Hinweis.
        try await routed.setDefaults(model: pinned, language: options.language)
        let warmUpMilliseconds = try await routed.warmUp()
        for member in await routed.members() {
            print("  \(member.file)  L=\(member.sequenceLength) K=\(member.maxOptions)")
        }
        print(String(format: "Laden und Warmlauf %.0f ms", warmUpMilliseconds))
        let server = JevServer(engine: routed, configuration: .init(port: options.port),
                               modelIdentifier: "laya-rl-agent",
                               modelDescription: "laya auf Core ML, lokal, ohne Netz",
                               modelAliases: LayaRouter.Checkpoint.allCases
                                   .filter(routed.available.contains).map(\.name),
                               optionLimit: .max)
        try server.start()
        print("POST http://127.0.0.1:\(options.port)/v1/systemone")
        while true { try await Task.sleep(for: .seconds(3600)) }
    }

    let questions: [(id: String, question: JevQuestion)]
    let state: JevValue
    if let requestURL = options.request {
        let parsed = try JevRequest.parse(jsonFile: requestURL, maxOptions: .max)
        state = parsed.state
        questions = parsed.questions
    } else {
        state = demoState
        questions = [("skill", .choice(instructions: demoInstruction, criteria: demoSkills))]
    }

    // Ein Lauf vorweg, ungemessen. Die Modelle werden erst beim ersten Gebrauch geladen, und
    // ohne diesen Lauf stünde bei --repeat 1 die Ladezeit als Latenz da.
    let warmStart = Date()
    _ = try await routed.answer(state: state, questions: questions,
                                model: pinned, language: options.language)
    print(String(format: "Laden und Warmlauf %.0f ms", Date().timeIntervalSince(warmStart) * 1000))

    var timings: [Double] = []
    var last: LayaRouted.Routed?
    for _ in 0 ..< max(1, options.repeats) {
        let (value, ms) = try await timed {
            try await routed.answer(state: state, questions: questions,
                                    model: pinned, language: options.language)
        }
        last = value
        timings.append(ms)
    }
    guard let last else { return }
    print("Checkpoint \(last.checkpoint): \(last.reason)\n")
    for (id, answer) in last.answers {
        switch answer.answer {
        case let .choice(choice):
            printProbabilities(choice.probabilities)
            print(String(format: "-> %@ = %@ (Konfidenz %.4f, Handlung %.4f)\n",
                         id, choice.choice, answer.confidence, answer.actProbability))
        case let .score(score):
            printProbabilities(zip(score.legend, score.probabilities).map { ($0.0, $0.1) })
            print(String(format: "-> %@ = %.4f (Konfidenz %.4f, Handlung %.4f)\n",
                         id, score.score, answer.confidence, answer.actProbability))
        case let .noul(noul):
            print(String(format: "-> %@ = %.4f (Konfidenz %.4f, Handlung %.4f)\n",
                         id, noul.noul, answer.confidence, answer.actProbability))
        }
    }
    let sorted = timings.sorted()
    let median = sorted[sorted.count / 2]
    print(String(format: "%d Frage(n) | %d Läufe, Median %.1f ms, schnellster %.1f ms",
                 questions.count, sorted.count, median, sorted[0]))

    if let out = options.json {
        var answers: [String: Any] = [:]
        for (id, answer) in last.answers {
            var entry: [String: Any] = ["confidence": answer.confidence,
                                        "action": ["act_probability": answer.actProbability]]
            switch answer.answer {
            case let .choice(choice):
                entry["type"] = "choice"
                entry["choice"] = choice.choice
            case let .score(score):
                entry["type"] = "score"
                entry["score"] = score.score
            case let .noul(noul):
                entry["type"] = "noul"
                entry["noul"] = noul.noul
            }
            answers[id] = entry
        }
        // p95 wie in laya_baseline.py: der Wert an der Stelle 0,95 der sortierten Liste.
        let p95 = sorted[min(sorted.count - 1, Int(0.95 * Double(sorted.count)))]
        let payload: [String: Any] = [
            "label": options.label.isEmpty ? "ohne Beschriftung" : options.label,
            "questions": questions.count,
            "routed_to": last.checkpoint,
            "reason": last.reason,
            "runs": sorted.count,
            "median_ms": median,
            "min_ms": sorted[0],
            "p95_ms": p95,
            "per_question_ms": median / Double(max(1, questions.count)),
            "answers": answers,
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: out)
        print("geschrieben: \(out.path)")
    }
}

// MARK: - kev

@MainActor func runKev() async throws {
    let started = Date()
    let kevConfiguration = JevRuntime.Configuration(computeUnits: units.value)
    let engine: any SystemOneEngine
    if options.buckets || options.modelList.count > 1 {
        // Mit --buckets kommen ausdrücklich angegebene Exporte dazu, statt still wegzufallen.
        var urls = options.buckets ? options.bucketURLs : []
        for url in options.modelList where !urls.contains(url) { urls.append(url) }
        guard !urls.isEmpty else {
            throw UsageError(description: "keine Fan-Out-Exporte unter \(options.models.path)")
        }
        if options.buckets && options.bucketURLs.count < Options.bucketNames.count {
            let found = Set(options.bucketURLs.map(\.lastPathComponent))
            let missing = Options.bucketNames.filter { !found.contains("\($0).mlpackage") }
            FileHandle.standardError.write(Data(
                "jev: für --buckets fehlen unter \(options.models.path): \(missing.joined(separator: ", "))\n".utf8))
        }
        engine = try SystemOnePool(modelURLs: urls, tokenizerURL: options.tokenizerURL,
                                   configuration: kevConfiguration)
    } else {
        engine = try SystemOne(modelURL: options.modelURL, tokenizerURL: options.tokenizerURL,
                               configuration: kevConfiguration)
    }
    let loadedMembers = await engine.members()
    print(String(format: "%d Export(e) geladen in %.2f s", loadedMembers.count, Date().timeIntervalSince(started)))
    for member in loadedMembers {
        print("  \(member.file)  L=\(member.sequenceLength) Q=\(member.maxQuestions) K=\(member.maxOptions)")
    }
    let warmUpMilliseconds = try await engine.warmUp()
    print(String(format: "Warmlauf %.0f ms", warmUpMilliseconds))

    if options.demo {
        var timings: [Double] = []
        var answer: ChoiceAnswer?
        let demoRequest = JevRequest(state: demoState, questions: [
            ("skill", .choice(instructions: demoInstruction, criteria: demoSkills)),
        ])
        for _ in 0 ..< max(1, options.repeats) {
            let (response, ms) = try await timed { try await engine.answer(demoRequest) }
            if case let .choice(choice)? = response["skill"] { answer = choice }
            timings.append(ms)
        }
        let sorted = timings.sorted()
        let median = sorted[sorted.count / 2]
        if let answer {
            let width = answer.probabilities.map(\.name.count).max() ?? 0
            for entry in answer.probabilities {
                let name = entry.name.padding(toLength: max(width, 8), withPad: " ", startingAt: 0)
                print(String(format: "%@  %.2f", name, entry.probability))
            }
            print(String(format: "\n-> %@ (Konfidenz %.2f) | %d Läufe, Median %.1f ms, schnellster %.1f ms",
                         answer.choice, answer.confidence, sorted.count, median, sorted[0]))
        }
    }

    if options.serve {
        let server = JevServer(engine: engine, configuration: .init(port: options.port))
        try server.start()
        print("POST http://127.0.0.1:\(options.port)/v1/systemone")
        while true { try await Task.sleep(for: .seconds(3600)) }
    }

    if let requestURL = options.request {
        let request = try JevRequest.parse(jsonFile: requestURL)
        var timings: [Double] = []
        var response: SystemOneResponse?
        for _ in 0 ..< max(1, options.repeats) {
            let (value, took) = try await timed { try await engine.answer(request) }
            response = value
            timings.append(took)
        }
        guard let response else { return }
        var payload: [String: Any] = [:]
        for (id, answer) in response.answers { payload[id] = answer.jsonObject(id: id) }
        let data = try JSONSerialization.data(withJSONObject: ["answers": payload, "model": response.model,
                                                               "usage": ["input_tokens": response.usage.inputTokens,
                                                                         "output_tokens": response.usage.outputTokens]],
                                              options: [.prettyPrinted, .sortedKeys])
        print(String(data: data, encoding: .utf8) ?? "")
        let sorted = timings.sorted()
        print(String(format: "%d Frage(n), %d Modelldurchlauf/Durchläufe, %d Eingabe-Token | %d Läufe, Median %.1f ms",
                     response.answers.count, response.usage.passes, response.usage.inputTokens,
                     sorted.count, sorted[sorted.count / 2]))
    }
}

// Fehler als Meldung und Exit-Code, nicht als Absturz. Ein `try` auf oberster Ebene endet sonst
// in einem Swift-Fatal-Error samt Absturzbericht, auch bei einer ordentlichen Ablehnung wie
// "zu wenig Platz" oder einem unbekannten Checkpoint.
do {
    if options.engine == "laya" {
        try await runLaya()
    } else {
        try await runKev()
    }
} catch let error as UsageError {
    FileHandle.standardError.write(Data("jev: \(error.description)\n".utf8))
    exit(2)
} catch let error as JevError {
    FileHandle.standardError.write(Data("jev: \(error.description)\n".utf8))
    exit(1)
} catch {
    FileHandle.standardError.write(Data("jev: \(error.localizedDescription)\n".utf8))
    exit(1)
}
