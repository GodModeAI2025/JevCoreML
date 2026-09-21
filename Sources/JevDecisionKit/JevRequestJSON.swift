import Foundation

extension JevRequest {
    /// Wie `MAX_OPTIONS` in `kev/api.py`.
    public static let maxOptions = 255

    /// Zwei Nachkommastellen, kaufmaennisch gerundet, wie Pythons `round(x, 2)`.
    public static func roundHalfToEven(_ x: Double) -> Double {
        guard x.isFinite else { return x }
        return Double(String(format: "%.2f", x)) ?? x
    }

    /// Liest das TypeSafe-kompatible Anfrageformat von `POST /v1/systemone`.
    ///
    /// `maxOptions` ist kevs Grenze aus `kev/api.py`. laya kennt keine, dort entscheidet die
    /// Maschine selbst, wie viele Optionen in die Sequenz passen; dafür `.max`.
    public static func parse(json data: Data, maxOptions: Int = JevRequest.maxOptions) throws -> JevRequest {
        try parse(try JevValue.parse(json: data), maxOptions: maxOptions)
    }

    /// Wie `parse(json:)`, aber auf einem bereits gelesenen Wert.
    public static func parse(_ root: JevValue, maxOptions: Int = JevRequest.maxOptions) throws -> JevRequest {
        guard case let .object(fields) = root else {
            throw JevError.invalidJSON("Anfrage ist kein Objekt")
        }
        // state ist in der Referenz ein Pflichtfeld ohne Default. Fehlt es ganz, antwortet
        // kev.serve mit 422; ein ausdrueckliches null bleibt dagegen erlaubt.
        guard let state = fields.first(where: { $0.key == "state" })?.value else {
            throw JevError.invalidJSON("state fehlt")
        }
        // Vorgabe wie `SystemOneRequest.model` in kev/api.py. Ein Wert, der keine Zeichenkette
        // ist, wird dort von pydantic abgelehnt, nicht stillschweigend ersetzt.
        var model = "kev-latest"
        switch fields.first(where: { $0.key == "model" })?.value {
        case let .string(m)?: model = m
        case .none: break
        default: throw JevError.invalidJSON("model muss eine Zeichenkette sein")
        }
        // Nur für laya. Fehlt das Feld oder ist es null, entscheidet die Schrift.
        func hint(_ name: String) throws -> String? {
            switch fields.first(where: { $0.key == name })?.value {
            case let .string(s)?: return s
            case .none, .null?: return nil
            default: throw JevError.invalidJSON("\(name) muss eine Zeichenkette sein")
            }
        }
        let task = try hint("task")
        let language = try hint("lang")
        guard case let .object(questions)? = fields.first(where: { $0.key == "questions" })?.value else {
            throw JevError.invalidJSON("questions fehlt oder ist kein Objekt")
        }
        guard !questions.isEmpty else { throw JevError.invalidJSON("questions ist leer") }

        var parsed: [(id: String, question: JevQuestion)] = []
        for (id, raw) in questions {
            guard case let .object(q) = raw else {
                throw JevError.invalidJSON("Frage \(id) ist kein Objekt")
            }
            func field(_ name: String) -> JevValue? { q.first { $0.key == name }?.value }
            guard case let .string(type)? = field("type") else {
                throw JevError.invalidJSON("Frage \(id) hat kein type")
            }
            guard let instructions = field("instructions") else {
                throw JevError.invalidJSON("Frage \(id) hat keine instructions")
            }
            let criteria = field("criteria")

            switch type {
            case "noul":
                var whenTrue = JevValue.null
                var whenFalse = JevValue.null
                switch criteria {
                case let .object(pairs)?:
                    whenTrue = pairs.first { $0.key == "true" }?.value ?? .null
                    whenFalse = pairs.first { $0.key == "false" }?.value ?? .null
                case .none, .null?:
                    break
                default:
                    // Die Referenz validiert criteria als Objekt oder null. Eine Liste oder eine
                    // Zeichenkette still zu verwerfen hiesse, eine falsche Anfrage zu beantworten.
                    throw JevError.invalidJSON("noul \(id): criteria muss ein Objekt oder null sein")
                }
                parsed.append((id, .noul(instructions: instructions, whenTrue: whenTrue, whenFalse: whenFalse)))
            case "choice":
                guard case let .object(pairs)? = criteria, !pairs.isEmpty else {
                    throw JevError.invalidJSON("choice \(id) braucht criteria als Objekt")
                }
                guard pairs.count <= maxOptions else {
                    throw JevError.invalidJSON("choice \(id) hat \(pairs.count) Optionen, erlaubt sind \(maxOptions)")
                }
                parsed.append((id, .choice(instructions: instructions,
                                           criteria: pairs.map { (name: $0.key, description: $0.value) })))
            case "score":
                guard case let .array(levels)? = criteria, levels.count >= 2 else {
                    throw JevError.invalidJSON("score \(id) braucht criteria als Liste mit mindestens zwei Stufen")
                }
                guard levels.count <= maxOptions else {
                    throw JevError.invalidJSON("score \(id) hat \(levels.count) Stufen, erlaubt sind \(maxOptions)")
                }
                parsed.append((id, .score(instructions: instructions, levels: levels)))
            default:
                throw JevError.invalidJSON("unbekannter Fragetyp \(type) bei \(id)")
            }
        }
        return JevRequest(state: state, model: model, questions: parsed, task: task, language: language)
    }

    public static func parse(jsonFile url: URL, maxOptions: Int = JevRequest.maxOptions) throws -> JevRequest {
        try parse(json: try Data(contentsOf: url), maxOptions: maxOptions)
    }
}

/// Schreibt die Antwort so, wie `kev.api` sie serialisiert: zwei Nachkommastellen als Text,
/// nicht als naechstgelegener Double, und dieselben Trenner wie Pythons `json.dumps`.
///
/// Beides zusammen entscheidet ueber `usage.output_tokens`, das kev als Zahl der Token dieser
/// Zeichenkette definiert. Mit `JSONSerialization` und 17-stelligen Doubles kam dort rund die
/// Haelfte zu viel heraus.
enum KevAnswerJSON {
    static func text(_ answers: [(id: String, answer: JevAnswer)], pretty: Bool,
                     rendered: [String: String] = [:]) -> String {
        let pairs = answers.map { "\(quote($0.id)): \(rendered[$0.id] ?? value($0.answer))" }
        return "{" + pairs.joined(separator: ", ") + "}"
    }

    static func number(_ x: Double) -> String {
        let rounded = JevRequest.roundHalfToEven(x)
        if rounded == rounded.rounded() && abs(rounded) < 1e15 {
            return String(format: "%.1f", rounded)     // wie Pythons repr(0.0) -> "0.0"
        }
        var s = String(format: "%.2f", rounded)
        while s.hasSuffix("0") && !s.hasSuffix(".0") { s.removeLast() }
        return s
    }

    static func quote(_ s: String) -> String {
        var out = "\""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }

    static func value(_ answer: JevAnswer) -> String {
        switch answer {
        case let .noul(a):
            return "{\"type\": \"noul\", \"noul\": \(number(a.noul))}"
        case let .choice(a):
            let dist = a.probabilities.map { "\(quote($0.name)): \(number($0.probability))" }
            return "{\"type\": \"choice\", \"choice\": \(quote(a.choice)), "
                + "\"confidence\": \(number(a.confidence)), "
                + "\"probabilities\": {" + dist.joined(separator: ", ") + "}}"
        case let .score(a):
            let legend = a.legend.enumerated().map { "\(quote(String($0.offset))): \(quote($0.element))" }
            let dist = a.probabilities.enumerated().map { "\(quote(String($0.offset))): \(number($0.element))" }
            return "{\"type\": \"score\", \"score\": \(number(a.score)), "
                + "\"legend\": {" + legend.joined(separator: ", ") + "}, "
                + "\"probabilities\": {" + dist.joined(separator: ", ") + "}, "
                + "\"confidence\": \(number(a.confidence))}"
        }
    }
}

extension JevAnswer {
    /// Serialisiert wie `kev.api.to_answers`, inklusive der dortigen Rundung auf zwei Stellen.
    public func jsonObject(id: String) -> [String: Any] {
        // Pythons round(x, 2) rundet den exakten Binaerwert und waehlt bei Gleichstand die
        // gerade Ziffer. (x * 100).rounded() rundet dagegen von null weg und verschiebt den
        // Wert vorher ueber eine Multiplikation, die selbst schon rundet.
        func r2(_ x: Double) -> Double { JevRequest.roundHalfToEven(x) }
        switch self {
        case let .noul(a):
            return ["type": "noul", "noul": r2(a.noul)]
        case let .choice(a):
            var dist: [String: Double] = [:]
            for entry in a.probabilities { dist[entry.name] = r2(entry.probability) }
            return ["type": "choice", "choice": a.choice, "confidence": r2(a.confidence), "probabilities": dist]
        case let .score(a):
            var legend: [String: String] = [:]
            for (i, text) in a.legend.enumerated() { legend[String(i)] = text }
            var dist: [String: Double] = [:]
            for (i, p) in a.probabilities.enumerated() { dist[String(i)] = r2(p) }
            return ["type": "score", "score": r2(a.score), "legend": legend,
                    "probabilities": dist, "confidence": r2(a.confidence)]
        }
    }
}
