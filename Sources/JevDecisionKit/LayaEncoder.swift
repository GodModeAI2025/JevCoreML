import Foundation

/// Serialisierung wie Pythons `json.dumps`.
///
/// laya baut Teile der Sequenz aus JSON: ein Zustand, der kein String ist, geht durch
/// `json.dumps(state, ensure_ascii=False)`, eine Anweisung, die kein String ist, durch
/// `json.dumps(ins)` mit der Voreinstellung `ensure_ascii=True`. Die beiden unterscheiden sich
/// bei jedem Zeichen oberhalb ASCII, und der Unterschied landet direkt im Tokenizer.
enum PythonJSON {
    static func dumps(_ value: JevValue, ensureASCII: Bool) -> String {
        switch value {
        case .null: return "null"
        case let .bool(b): return b ? "true" : "false"
        case let .int(i): return String(i)
        case let .bigint(s): return s
        // json.dumps schreibt NaN, Infinity und -Infinity, nicht Pythons str() mit nan und inf.
        case let .double(d) where d.isNaN: return "NaN"
        case let .double(d) where d.isInfinite: return d > 0 ? "Infinity" : "-Infinity"
        case let .double(d): return JevValue.pythonFloat(d)
        case let .string(s): return quote(s, ensureASCII: ensureASCII)
        case let .array(items):
            return "[" + items.map { dumps($0, ensureASCII: ensureASCII) }.joined(separator: ", ") + "]"
        case let .object(pairs):
            let body = pairs.map {
                quote($0.key, ensureASCII: ensureASCII) + ": " + dumps($0.value, ensureASCII: ensureASCII)
            }
            return "{" + body.joined(separator: ", ") + "}"
        }
    }

    static func quote(_ s: String, ensureASCII: Bool) -> String {
        var out = "\""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else if ensureASCII && scalar.value >= 0x7F {
                    // Ab 0x7F, nicht erst darüber: Python maskiert alles außerhalb von Leerzeichen
                    // bis Tilde, also auch DEL.
                    // Python schreibt oberhalb der BMP ein Ersatzzeichenpaar, nicht \U########.
                    if scalar.value > 0xFFFF {
                        let v = scalar.value - 0x10000
                        out += String(format: "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF))
                    } else {
                        out += String(format: "\\u%04x", scalar.value)
                    }
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }
}

/// Baut die Eingabesequenz eines laya-Durchlaufs.
///
/// Das Format ist
///
///     [CLS] "<typ> question: <anweisung>" [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] zustand [SEP]
///
/// Gelesen wird an den `[MASK]`-Positionen. Anders als bei kev steht der Zustand hinten und wird
/// von rechts gekürzt, und der Kopf hat ein eigenes Budget: passen die Optionen nicht hinein,
/// wird erst jede Option gekürzt und danach die Anweisung.
public struct LayaEncoder: Sendable {
    public struct Encoded: Sendable {
        public let ids: [Int32]
        public let markers: [Int32]
        public let qtype: Int32
        /// Namen in Beschriftungsreihenfolge: Optionsnamen bei `choice`, Stufenindizes bei
        /// `score`, `["false", "true"]` bei `noul`.
        public let labels: [String]
        /// Die Optionstexte, so wie sie in der Sequenz stehen. Für Tests und Diagnose.
        public let options: [String]
        /// Die Legende der Antwort. Bei `score` sind das die Stufen selbst, nicht die daraus
        /// gebauten Optionstexte: laya gibt `{"0": "not urgent", ...}` zurück und nicht
        /// `{"0": "level 0: not urgent"}`. Als Text, weil `ScoreAnswer` Text führt; eine Stufe,
        /// die kein String ist, steht hier als kompaktes JSON.
        public let legend: [String]
        /// Dieselben Stufen als Rohwerte, so wie laya sie in `legend` zurückgibt: eine Zahl
        /// bleibt eine Zahl, ein Objekt ein Objekt.
        public let legendValues: [JevValue]
        /// Wurde der Zustand beschnitten, weil er nicht in das Längenbudget passte?
        public let stateTruncated: Bool
    }

    public let tokenizer: any TextTokenizer
    public let maxLength: Int
    public let headMaxLength: Int
    public let clsID: Int32
    public let sepID: Int32
    public let maskID: Int32
    public let padID: Int32
    /// Der Maskentext, den laya in Anweisung, Optionen und Zustand durch ein Leerzeichen ersetzt.
    /// Ersetzt wird immer mit `.literal`: ohne die Option sucht Foundation nach kanonischer
    /// Äquivalenz und findet `[MASK]` nicht, wenn eine kombinierende Marke direkt folgt. Dann
    /// bliebe der Text stehen, der Tokenizer machte daraus das echte Markertoken, und die
    /// Sequenz enthielte einen Marker mitten im Zustand. Python ersetzt Code-Punkt für Code-Punkt.
    public let maskToken: String

    /// Wie viele Token eine einzelne Option höchstens beiträgt, ohne den Marker.
    static let optionTokenLimit = 48
    /// Untergrenze für das Kopfbudget, bevor Optionen gekürzt werden.
    static let optionBudgetFloor = 16
    /// So viele Token der Anweisung bleiben in jedem Fall stehen.
    static let headMinimum = 8

    public init(tokenizer: any TextTokenizer, maxLength: Int, headMaxLength: Int,
                clsID: Int32, sepID: Int32, maskID: Int32, padID: Int32,
                maskToken: String = "[MASK]") {
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.headMaxLength = headMaxLength
        self.clsID = clsID
        self.sepID = sepID
        self.maskID = maskID
        self.padID = padID
        self.maskToken = maskToken
    }

    // MARK: - Darstellung

    /// `laya.common.render_criterion`: Strings gehen durch, alles andere wird kompaktes JSON.
    func renderCriterion(_ value: JevValue) -> String {
        if case let .string(s) = value { return s }
        return PythonJSON.dumps(value, ensureASCII: false)
    }

    /// `laya.common.serialize_state`.
    public func serializeState(_ state: JevValue) -> String {
        if case let .string(s) = state { return s }
        return PythonJSON.dumps(state, ensureASCII: false)
    }

    /// `laya.agent.Agent._to_internal`: eine Anweisung, die kein String ist, wird JSON, und zwar
    /// mit `ensure_ascii=True`, anders als der Zustand.
    func renderInstructions(_ value: JevValue) -> String {
        if case let .string(s) = value { return s }
        return PythonJSON.dumps(value, ensureASCII: true)
    }

    /// `laya.common.render_options`. Die Reihenfolge ist die Beschriftungsreihenfolge.
    public func renderOptions(_ question: JevQuestion)
        -> (labels: [String], texts: [String], legend: [String], values: [JevValue], qtype: Int32) {
        switch question {
        case let .choice(_, criteria):
            let texts = criteria.map { pair -> String in
                // Nur null und "" heißen "keine Beschreibung". 0 und false sind gültige Werte.
                if case .null = pair.description { return pair.name }
                if case let .string(s) = pair.description, s.isEmpty { return pair.name }
                return "\(pair.name): \(renderCriterion(pair.description))"
            }
            return (criteria.map(\.name), texts, criteria.map(\.name), criteria.map { .string($0.name) }, 0)
        case let .score(_, levels):
            let texts = levels.enumerated().map { "level \($0.offset): \(renderCriterion($0.element))" }
            return (levels.indices.map(String.init), texts, levels.map(renderCriterion), levels, 1)
        case let .noul(_, whenTrue, whenFalse):
            func text(_ value: JevValue, fallback: String) -> String {
                if case .null = value { return fallback }
                if case let .string(s) = value, s.isEmpty { return fallback }
                return renderCriterion(value)
            }
            return (["false", "true"],
                    ["false: " + text(whenFalse, fallback: "no, the statement does not hold"),
                     "true: " + text(whenTrue, fallback: "yes, the statement holds")],
                    ["false", "true"],
                    [.string("false"), .string("true")],
                    2)
        }
    }

    static func typeName(_ qtype: Int32) -> String {
        switch qtype {
        case 0: return "choice"
        case 1: return "score"
        default: return "noul"
        }
    }

    static func instructions(of question: JevQuestion) -> JevValue {
        switch question {
        case let .choice(instructions, _): return instructions
        case let .score(instructions, _): return instructions
        case let .noul(instructions, _, _): return instructions
        }
    }

    // MARK: - Sequenz

    public func encode(state: JevValue, question: JevQuestion) throws -> Encoded {
        let (labels, optionTexts, legend, legendValues, qtype) = renderOptions(question)
        guard !optionTexts.isEmpty else { throw JevError.noOptions }

        let ins = renderInstructions(LayaEncoder.instructions(of: question))
            .replacingOccurrences(of: maskToken, with: " ", options: .literal)
        var headIDs = tokenizer.encode("\(LayaEncoder.typeName(qtype)) question: \(ins)")

        var optionIDs = optionTexts.map { text -> [Int32] in
            let body = tokenizer.encode(" " + text.replacingOccurrences(of: maskToken, with: " ", options: .literal))
            return [maskID] + body.prefix(LayaEncoder.optionTokenLimit)
        }
        var budget = headMaxLength - optionIDs.reduce(0) { $0 + $1.count }
        if budget < LayaEncoder.optionBudgetFloor {
            // Erst die Optionen gleichmäßig kürzen, dann neu rechnen. Der Marker zählt mit,
            // deshalb bleiben mindestens vier Token je Option stehen.
            let perOption = max(4, (headMaxLength - LayaEncoder.optionBudgetFloor) / max(1, optionIDs.count))
            optionIDs = optionIDs.map { Array($0.prefix(perOption)) }
            budget = headMaxLength - optionIDs.reduce(0) { $0 + $1.count }
        }
        headIDs = Array(headIDs.prefix(max(LayaEncoder.headMinimum, budget)))

        var ids: [Int32] = [clsID]
        ids.append(contentsOf: headIDs)
        ids.append(sepID)
        var markers: [Int32] = []
        for option in optionIDs {
            markers.append(Int32(ids.count))
            ids.append(contentsOf: option)
        }
        ids.append(sepID)

        let room = max(0, maxLength - ids.count - 1)
        let stateIDs = tokenizer.encode(serializeState(state).replacingOccurrences(of: maskToken, with: " ", options: .literal))
        let kept = Array(stateIDs.prefix(room))
        ids.append(contentsOf: kept)
        ids.append(sepID)

        let truncated = Array(ids.prefix(maxLength))
        let keptMarkers = markers.filter { Int($0) < maxLength }
        // laya bricht hier ab ("question options exceed head_max_len"), statt eine Frage mit
        // weniger Optionen zu beantworten, als gestellt wurden.
        guard keptMarkers.count == optionTexts.count else {
            throw JevError.optionsDoNotFit(count: optionTexts.count, fitting: keptMarkers.count,
                                           length: maxLength)
        }
        return Encoded(ids: truncated,
                       markers: keptMarkers,
                       qtype: qtype,
                       labels: labels,
                       options: optionTexts,
                       legend: legend,
                       legendValues: legendValues,
                       stateTruncated: kept.count < stateIDs.count || ids.count > maxLength)
    }
}
