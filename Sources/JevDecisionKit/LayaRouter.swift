import Foundation

/// Wahl des laya-Checkpoints anhand der Schrift des Zustands.
///
/// Das ist kein Beiwerk, sondern der Grund, warum es mehr als einen Checkpoint gibt. Der
/// englische bricht auf fremder Schrift nicht sanft ein, er fällt auf Raten zurück und meldet
/// dabei hohe Sicherheit. Auf den eigenen deutschen Referenzanfragen ist der Unterschied
/// messbar: `skill` bekommt vom englischen Checkpoint 0,0934 Konfidenz, vom mehrsprachigen
/// 0,9501.
///
/// Die Schrifterkennung ist exakt, der Sprachtipp für lateinische Schrift ist eine Heuristik
/// aus Funktionswörtern und Diakritika und gibt sich als solche zu erkennen.
public enum LayaRouter {
    public enum Checkpoint: String, Sendable, CaseIterable {
        case english, multilingual, typedDecisions

        public var name: String {
            self == .typedDecisions ? "typed-decisions" : rawValue
        }

        /// Die Namen, die Leute tippen. Entspricht `_ALIASES` in laya.router.
        public static func named(_ raw: String) -> Checkpoint? {
            switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            case "english", "en", "laya", "default": return .english
            case "multilingual", "multi", "ml", "laya-multilingual": return .multilingual
            case "typed-decisions", "typed", "typed_decisions",
                 "laya-typed-decisions", "decisions": return .typedDecisions
            default: return nil
            }
        }
    }

    public struct Detection: Sendable, Equatable {
        public let script: String
        public let scriptProfile: [(name: String, fraction: Double)]
        public let language: String?
        public let isEnglish: Bool
        public let nonLatinFraction: Double

        public static func == (lhs: Detection, rhs: Detection) -> Bool {
            lhs.script == rhs.script && lhs.language == rhs.language
                && lhs.isEnglish == rhs.isEnglish && lhs.nonLatinFraction == rhs.nonLatinFraction
        }
    }

    public struct Decision: Sendable {
        public let checkpoint: Checkpoint
        public let reason: String
        public let detection: Detection?
        /// Der erkannte typed-decisions-Ablauf, falls die Fragen-IDs genau passen.
        public let workflow: String?
    }

    // MARK: - Schrift

    /// Unicode-Bereiche, die der englische Checkpoint nicht lesen kann. Reihenfolge wie in laya,
    /// denn der erste Treffer gewinnt und Hangul überlappt mit Han.
    static let scriptRanges: [(name: String, ranges: [ClosedRange<UInt32>])] = [
        ("greek", [0x0370 ... 0x03FF, 0x1F00 ... 0x1FFF]),
        ("cyrillic", [0x0400 ... 0x052F, 0x2DE0 ... 0x2DFF, 0xA640 ... 0xA69F]),
        ("hebrew", [0x0590 ... 0x05FF]),
        ("arabic", [0x0600 ... 0x06FF, 0x0750 ... 0x077F, 0x08A0 ... 0x08FF,
                    0xFB50 ... 0xFDFF, 0xFE70 ... 0xFEFF]),
        ("devanagari", [0x0900 ... 0x097F, 0xA8E0 ... 0xA8FF]),
        ("bengali", [0x0980 ... 0x09FF]),
        ("gurmukhi", [0x0A00 ... 0x0A7F]),
        ("gujarati", [0x0A80 ... 0x0AFF]),
        ("oriya", [0x0B00 ... 0x0B7F]),
        ("tamil", [0x0B80 ... 0x0BFF]),
        ("telugu", [0x0C00 ... 0x0C7F]),
        ("kannada", [0x0C80 ... 0x0CFF]),
        ("malayalam", [0x0D00 ... 0x0D7F]),
        ("sinhala", [0x0D80 ... 0x0DFF]),
        ("thai", [0x0E00 ... 0x0E7F]),
        ("lao", [0x0E80 ... 0x0EFF]),
        ("tibetan", [0x0F00 ... 0x0FFF]),
        ("myanmar", [0x1000 ... 0x109F]),
        ("georgian", [0x10A0 ... 0x10FF]),
        ("ethiopic", [0x1200 ... 0x137F]),
        ("khmer", [0x1780 ... 0x17FF]),
        ("hangul", [0x1100 ... 0x11FF, 0x3130 ... 0x318F, 0xAC00 ... 0xD7AF]),
        ("kana", [0x3040 ... 0x309F, 0x30A0 ... 0x30FF, 0x31F0 ... 0x31FF]),
        ("han", [0x3400 ... 0x4DBF, 0x4E00 ... 0x9FFF, 0xF900 ... 0xFAFF]),
    ]

    /// Pythons `str.isalpha()`: genau die Kategorien Lu, Ll, Lt, Lm, Lo.
    /// Nicht `properties.isAlphabetic`, das zählt auch Vokalzeichen mit, die Python nicht zählt.
    static func isLetter(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter:
            return true
        default:
            return false
        }
    }

    /// Die Strings eines Zustands, in der Reihenfolge, in der laya sie einsammelt. Schlüssel
    /// bleiben draußen, die sind meist englisch und würden die Erkennung verfälschen.
    public static func stateText(_ state: JevValue, maxCharacters: Int = 4000) -> String {
        var parts: [String] = []
        func walk(_ value: JevValue, depth: Int) {
            guard depth <= 6 else { return }
            switch value {
            case let .string(s): parts.append(s)
            case let .array(items): for item in items { walk(item, depth: depth + 1) }
            case let .object(pairs): for pair in pairs { walk(pair.value, depth: depth + 1) }
            default: break
            }
        }
        walk(state, depth: 0)
        return String(parts.joined(separator: " ").prefix(maxCharacters))
    }

    static func counts(in text: String) -> [(name: String, count: Int)] {
        var latin = 0
        var others: [(name: String, count: Int)] = []
        for scalar in text.unicodeScalars where isLetter(scalar) {
            let cp = scalar.value
            if cp < 0x0250 || (0x1E00 ... 0x1EFF).contains(cp) {
                latin += 1
                continue
            }
            for entry in scriptRanges where entry.ranges.contains(where: { $0.contains(cp) }) {
                if let index = others.firstIndex(where: { $0.name == entry.name }) {
                    others[index].count += 1
                } else {
                    others.append((entry.name, 1))
                }
                break
            }
        }
        return [(name: "latin", count: latin)] + others
    }

    public static func detectScript(_ text: String) -> String {
        let tally = counts(in: text)
        let total = tally.reduce(0) { $0 + $1.count }
        guard total > 0 else { return "unknown" }
        // Pythons max() nimmt den ersten Höchstwert in Einfügereihenfolge. In `detect_script`
        // kommt "latin" erst nach den anderen Schriften ins dict, verliert also jeden Gleichstand:
        // halb lateinisch, halb kyrillisch ist dort kyrillisch. `script_profile` führt "latin"
        // dagegen vorn; deshalb liefert `counts` es vorn und hier wird umgestellt.
        let ordered = Array(tally.dropFirst()) + [tally[0]]
        var best = ordered[0]
        for entry in ordered.dropFirst() where entry.count > best.count { best = entry }
        return best.name
    }

    public static func scriptProfile(_ text: String) -> [(name: String, fraction: Double)] {
        let tally = counts(in: text)
        let total = tally.reduce(0) { $0 + $1.count }
        guard total > 0 else { return [] }
        return tally.filter { $0.count > 0 }.map { ($0.name, Double($0.count) / Double(total)) }
    }

    // MARK: - Sprache

    static let stopwords: [(language: String, words: Set<String>)] = [
        ("en", ["the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
                "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
                "will", "can", "would", "there", "their", "what", "which", "please", "we", "i"]),
        ("fr", ["le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
                "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce"]),
        ("de", ["der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
                "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber"]),
        ("es", ["el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
                "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus"]),
        ("pt", ["os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
                "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela"]),
        ("it", ["il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
                "questo", "questa", "anche", "come", "più", "nella", "alla"]),
        ("nl", ["het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
                "door", "maar", "ook", "worden", "deze", "naar", "wordt"]),
    ]

    static let nonEnglishDiacritics = Set("àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö")

    /// `[^\W\d_]+` in Pythons re: Wortzeichen ohne Dezimalziffern und Unterstrich, also
    /// Buchstaben plus die Zahlkategorien Nl und No.
    static func words(in text: String) -> [String] {
        var out: [String] = []
        var current = String.UnicodeScalarView()
        for scalar in text.unicodeScalars {
            let category = scalar.properties.generalCategory
            let isWord = isLetter(scalar) || category == .letterNumber || category == .otherNumber
            if isWord {
                current.append(scalar)
            } else if !current.isEmpty {
                out.append(String(current).lowercased())
                current = String.UnicodeScalarView()
            }
        }
        if !current.isEmpty { out.append(String(current).lowercased()) }
        return out
    }

    public static func guessLatinLanguage(_ text: String) -> String? {
        let found = words(in: text)
        guard found.count >= 4 else { return nil }
        var scores: [String: Int] = [:]
        for entry in stopwords {
            scores[entry.language] = found.reduce(0) { $0 + (entry.words.contains($1) ? 1 : 0) }
        }
        let lowered = text.lowercased()
        let diacritics = lowered.reduce(0) { $0 + (nonEnglishDiacritics.contains($1) ? 1 : 0) }
        let diacriticRate = Double(diacritics) / Double(max(1, lowered.count))
        let english = scores["en"] ?? 0

        var bestLanguage: String?
        var best = 0
        for entry in stopwords where entry.language != "en" {
            let score = scores[entry.language] ?? 0
            if bestLanguage == nil || score > best {
                bestLanguage = entry.language
                best = score
            }
        }
        if best == 0 && diacriticRate < 0.02 { return english > 0 ? "en" : nil }
        if let language = bestLanguage, best >= max(2, english + 2) { return language }
        if diacriticRate >= 0.04, let language = bestLanguage, best >= english { return language }
        return english > 0 ? "en" : nil
    }

    public static func analyse(_ state: JevValue) -> Detection {
        let text = stateText(state)
        let profile = scriptProfile(text)
        let script = detectScript(text)
        let latin = profile.first { $0.name == "latin" }?.fraction ?? 0.0
        // round(x, 4) in Python rundet den exakten Binärwert zur Hälfte gerade; `%.4f` tut dasselbe.
        // Mal 10 000, runden, durch 10 000 wiche an Grenzfällen um eine Stelle ab.
        let nonLatin = profile.isEmpty ? 0.0 : Double(String(format: "%.4f", 1.0 - latin)) ?? 0.0
        if script == "unknown" {
            return Detection(script: "unknown", scriptProfile: profile, language: nil,
                             isEnglish: true, nonLatinFraction: 0.0)
        }
        if script != "latin" {
            return Detection(script: script, scriptProfile: profile, language: nil,
                             isEnglish: false, nonLatinFraction: nonLatin)
        }
        let language = guessLatinLanguage(text)
        return Detection(script: "latin", scriptProfile: profile, language: language,
                         isEnglish: language == nil || language == "en", nonLatinFraction: nonLatin)
    }

    // MARK: - Entscheidung

    /// Die vier Abläufe, auf die der typed-decisions-Checkpoint feinjustiert ist. Die IDs müssen
    /// exakt passen, sonst fängt ein fremdes Schema mit einer Frage namens `urgency` ihn ein.
    static let typedDecisionWorkflows: [(name: String, ids: Set<String>)] = [
        ("agent_trace_observability", ["action", "needs_review", "outcome", "risk", "urgency"]),
        ("customer_service", ["action", "category", "churn_risk", "needs_human", "urgency"]),
        ("invoice_processing", ["discrepancy_severity", "disposition", "duplicate",
                                "matches_order", "urgency"]),
        ("security_incidents", ["credential_compromise", "disposition", "severity",
                                "true_positive", "urgency"]),
    ]

    public static func matchTypedDecisionsWorkflow(_ questionIDs: [String]) -> String? {
        let ids = Set(questionIDs)
        return typedDecisionWorkflows.first { $0.ids == ids }?.name
    }

    /// Vorrang: ausdrückliches Modell, dann ausdrückliche Aufgabe, dann der erkannte Ablauf
    /// (nur auf Wunsch), dann ausdrückliche Sprache, dann Schrift und Sprache, dann die Vorgabe.
    public static func decide(state: JevValue,
                              questionIDs: [String] = [],
                              model: String? = nil,
                              task: String? = nil,
                              language: String? = nil,
                              autoTaskDetection: Bool = false,
                              fallback: Checkpoint = .english) throws -> Decision {
        if let model {
            guard let checkpoint = Checkpoint.named(model) else {
                throw JevError.invalidRoute("unbekanntes Modell \(model)")
            }
            return Decision(checkpoint: checkpoint, reason: "ausdrückliches Modell \(model)",
                            detection: nil, workflow: nil)
        }
        if let task {
            let normalized = task.lowercased().replacingOccurrences(of: "-", with: "_")
            let name = normalized == "typed_decisions" ? "typed-decisions" : task
            guard let checkpoint = Checkpoint.named(name) else {
                throw JevError.invalidRoute("unbekannte Aufgabe \(task)")
            }
            return Decision(checkpoint: checkpoint, reason: "ausdrückliche Aufgabe \(task)",
                            detection: nil, workflow: nil)
        }
        let workflow = matchTypedDecisionsWorkflow(questionIDs)
        if let workflow, autoTaskDetection {
            return Decision(checkpoint: .typedDecisions,
                            reason: "die Fragen-IDs entsprechen dem Ablauf \(workflow)",
                            detection: nil, workflow: workflow)
        }
        if let language {
            let head = language.lowercased().split(separator: "-").first.map(String.init) ?? ""
            let checkpoint: Checkpoint = ["en", "eng", "english"].contains(head) ? .english : .multilingual
            return Decision(checkpoint: checkpoint, reason: "ausdrückliche Sprache \(language)",
                            detection: nil, workflow: workflow)
        }

        let detection = analyse(state)
        if detection.script == "unknown" {
            return Decision(checkpoint: fallback,
                            reason: "keine Buchstaben im Zustand, also die Vorgabe (\(fallback.name))",
                            detection: detection, workflow: workflow)
        }
        if detection.script != "latin" {
            let percent = Int((100 * detection.nonLatinFraction).rounded())
            return Decision(checkpoint: .multilingual,
                            reason: "fremde Schrift (\(detection.script), \(percent) % der Buchstaben); "
                                + "der englische Checkpoint kann sie nicht lesen",
                            detection: detection, workflow: workflow)
        }
        if !detection.isEnglish {
            return Decision(checkpoint: .multilingual,
                            reason: "lateinische Schrift, aber die Sprache sieht nach "
                                + "\(detection.language ?? "?") aus, nicht nach Englisch",
                            detection: detection, workflow: workflow)
        }
        return Decision(checkpoint: .english, reason: "englischer Text in lateinischer Schrift",
                        detection: detection, workflow: workflow)
    }
}
