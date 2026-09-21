import Foundation

/// laya hinter derselben HTTP-Schnittstelle wie kev.
///
/// laya selbst hat keinen Server. Die Form der Antwort gibt deshalb `laya.router.Router.predict`
/// vor: je Frage `action.act_probability`, eine Konfidenz auch bei `noul`, bei `score` die
/// `legend`, alle Zahlen auf vier Stellen gerundet, und auf oberster Ebene das `routing`. Das
/// kev-Format wäre ebenfalls TypeSafe-kompatibel, ließe aber genau das weg, was laya ausmacht.
///
/// Die Hinweise von `Router.predict` kommen aus der Anfrage: `task` und `lang` als eigene Felder,
/// `model` nur, wenn es einen Checkpoint nennt. Der TypeSafe-Client schickt dort `kev-latest`,
/// und das soll die Schrifterkennung nicht abschalten.
extension LayaRouted: SystemOneEngine {
    /// Das Modellfeld als Routing-Hinweis, falls es einen Checkpoint nennt. `laya` heißt wie in
    /// `laya.router` der englische.
    static func explicitModel(_ request: JevRequest) -> String? {
        LayaRouter.Checkpoint.named(request.model) == nil ? nil : request.model
    }

    public func answer(_ request: JevRequest) async throws -> SystemOneResponse {
        let started = Date()
        let routed = try await answer(state: request.state, questions: request.questions,
                                      model: Self.explicitModel(request), task: request.task,
                                      language: request.language)
        var rendered: [String: String] = [:]
        for (id, answer) in routed.answers { rendered[id] = LayaAnswerJSON.render(answer) }
        let routing = LayaAnswerJSON.routing(checkpoint: routed.checkpoint, reason: routed.reason,
                                             detection: routed.detection, workflow: routed.workflow)
        return SystemOneResponse(
            // So benennt laya.agent.system_one sich selbst; die Modellkennung der Anfrage ist
            // für kev gedacht und stünde hier falsch.
            model: "laya-rl-agent",
            answers: routed.answers.map { ($0.id, $0.answer.answer) },
            // laya zählt die Token aller Sequenzen und keine Ausgabetoken, es erzeugt keinen Text.
            usage: JevUsage(inputTokens: routed.answers.reduce(0) { $0 + $1.answer.inputTokens },
                            outputTokens: 0, stateTokens: 0, passes: routed.answers.count),
            latencyMilliseconds: Date().timeIntervalSince(started) * 1000,
            renderedAnswers: rendered,
            extraFields: [("routing", routing)])
    }

    public func outputTokens(of answers: [(id: String, answer: JevAnswer)]) async -> Int { 0 }

    /// Wie beim Beantworten: denselben Checkpoint wählen, jede Frage kodieren, die Grenzen des
    /// Pakets prüfen. Was laya ablehnt, weil die Optionen nicht vor max_len passen, wirft schon
    /// `encode` als `optionsDoNotFit`. Die Prüfung gegen `maxOptions` greift bei den mitgelieferten
    /// Exporten nie, denn dort ist K gleich L; sie schützt Exporte mit kleinerem K.
    public func check(_ request: JevRequest) async throws {
        guard !request.questions.isEmpty else { throw JevError.invalidJSON("questions ist leer") }
        let decision = try route(state: request.state, questionIDs: request.questions.map(\.id),
                                 model: Self.explicitModel(request), task: request.task,
                                 language: request.language)
        let system = try engine(decision.checkpoint)
        for (_, question) in request.questions {
            let encoded = try system.runtime.encoder.encode(state: request.state, question: question)
            let limit = system.runtime.metadata.maxOptions
            if encoded.markers.count > limit {
                throw JevError.tooManyOptions(count: encoded.markers.count, limit: limit)
            }
        }
    }

    /// Lädt jeden bereitstehenden Checkpoint in jeder aktiven Länge und rechnet einmal, damit
    /// die erste echte Anfrage weder lädt noch den Graphen vorbereitet.
    @discardableResult
    public func warmUp() async throws -> Double {
        let started = Date()
        for checkpoint in availableInOrder {
            try await engine(checkpoint).runtime.warmUp()
        }
        return Date().timeIntervalSince(started) * 1000
    }

    /// Ein Eintrag je Checkpoint. Ein laya-Durchlauf beantwortet eine Frage, daher Phase 1.
    public func members() async -> [EngineMember] {
        var out: [EngineMember] = []
        for checkpoint in availableInOrder {
            guard let system = try? engine(checkpoint) else { continue }
            let meta = system.runtime.metadata
            out.append(EngineMember(file: modelName(of: checkpoint), contract: .singleQuestion,
                                    sequenceLength: meta.sequenceLength, maxQuestions: 1,
                                    maxOptions: meta.maxOptions))
        }
        return out.sorted { $0.sequenceLength < $1.sequenceLength }
    }
}

/// Eine laya-Antwort als JSON, Feld für Feld wie `laya.agent.system_one` sie baut.
enum LayaAnswerJSON {
    /// `round(x, 4)` in Python und dann `repr`: auf vier Stellen korrekt gerundet, kürzeste
    /// Darstellung. `%.4f` rundet den exakten Binärwert, genau wie Pythons `round`.
    static func number(_ x: Double) -> String {
        JevValue.pythonFloat(Double(String(format: "%.4f", x)) ?? x)
    }

    static func quote(_ s: String) -> String { PythonJSON.quote(s, ensureASCII: false) }

    /// Woher die Gewichte stammen, wie `_repo_str` in `laya.router` es schreibt.
    static func repo(of checkpoint: String) -> String {
        checkpoint == "english" ? "convaiinnovations/laya" : "convaiinnovations/laya/\(checkpoint)"
    }

    /// `RouteDecision` mit denselben Schlüsseln in derselben Reihenfolge. Zwei Abweichungen:
    /// `reason` ist deutsch, und `repo` ist immer eine Zeichenkette. laya legt beim erkannten
    /// typed-decisions-Ablauf das rohe Tupel ab, und das wird im JSON zur Liste.
    static func routing(checkpoint: String, reason: String,
                        detection: LayaRouter.Detection?, workflow: String?) -> String {
        var detectionJSON = "null"
        if let detection {
            let profile = detection.scriptProfile
                .map { "\(quote($0.name)): \(JevValue.pythonFloat($0.fraction))" }.joined(separator: ", ")
            detectionJSON = "{\"script\": \(quote(detection.script)), \"script_profile\": {\(profile)}, "
                + "\"language\": \(detection.language.map(quote) ?? "null"), "
                + "\"is_english\": \(detection.isEnglish ? "true" : "false"), "
                + "\"non_latin_fraction\": \(JevValue.pythonFloat(detection.nonLatinFraction))}"
        }
        return "{\"model\": \(quote(checkpoint)), \"repo\": \(quote(repo(of: checkpoint))), "
            + "\"reason\": \(quote(reason)), \"detection\": \(detectionJSON), "
            + "\"workflow\": \(workflow.map(quote) ?? "null")}"
    }

    static func render(_ answer: LayaAnswer) -> String {
        let action = "{\"act_probability\": \(number(answer.actProbability))}"
        let confidence = number(answer.confidence)
        switch answer.answer {
        case let .choice(choice):
            let probabilities = choice.probabilities
                .map { "\(quote($0.name)): \(number($0.probability))" }.joined(separator: ", ")
            return "{\"type\": \"choice\", \"choice\": \(quote(choice.choice)), "
                + "\"probabilities\": {\(probabilities)}, \"confidence\": \(confidence), \"action\": \(action)}"
        case let .score(score):
            // Die Stufen so, wie sie kamen: laya legt die Kriterien unverändert in die Legende.
            let legendValues = answer.legend.isEmpty ? score.legend.map { JevValue.string($0) } : answer.legend
            let legend = legendValues.enumerated()
                .map { "\"\($0.offset)\": \(PythonJSON.dumps($0.element, ensureASCII: false))" }
                .joined(separator: ", ")
            let probabilities = score.probabilities.enumerated()
                .map { "\"\($0.offset)\": \(number($0.element))" }.joined(separator: ", ")
            return "{\"type\": \"score\", \"score\": \(number(score.score)), \"legend\": {\(legend)}, "
                + "\"probabilities\": {\(probabilities)}, \"confidence\": \(confidence), \"action\": \(action)}"
        case let .noul(noul):
            return "{\"type\": \"noul\", \"noul\": \(number(noul.noul)), "
                + "\"confidence\": \(confidence), \"action\": \(action)}"
        }
    }
}
