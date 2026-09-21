import Foundation

/// Mailtexte aufbereiten, wie `laya.email` es tut: zitierten Verlauf, Signaturen und
/// Haftungsausschlüsse entfernen, damit das Modell den eigentlichen Text liest.
///
/// Die Vorlage ist Python-Regex, die Ausführung hier ICU. Beide heißen Regex und meinen an vier
/// Stellen etwas anderes, deshalb ist keines der Muster wörtlich übernommen:
///
/// - `\s` ist in Python genau `str.isspace`, 29 Zeichen. ICU kennt U+000B, U+001C bis U+001F und
///   U+0085 nicht als Leerraum.
/// - `\w` ist in Python genau `[\p{L}\p{N}_]`. ICU nimmt kombinierende Marken und einige Zeichen
///   mehr dazu.
/// - `$` passt in Python am Ende oder vor einem letzten `\n`. ICU nimmt dafür auch U+2028, U+0085
///   und `\r`. Die Zeilen hier enthalten kein `\n` mehr, also ist `\z` das Python-Verhalten.
/// - `.` schließt in Python nur `\n` aus, in ICU alle Zeilentrenner. Hier steht `[^\n]`.
///
/// Geprüft gegen laya über einen Korpus aus handverlesenen und erzeugten Mails, siehe
/// `LayaEmailTests` und `Exporter/dump_laya_email.py`.
public enum LayaEmail {
    /// `str.isspace` in Python 3.12, Unicode 15.0. Aus Python abgeleitet, nicht geraten.
    static let pythonSpaceScalars: Set<UInt32> = [
        0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x85, 0xA0, 0x1680,
        0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A,
        0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
    ]

    /// Dieselbe Menge als Regex-Klasse, ohne die eckigen Klammern.
    static let space = #"\x{09}-\x{0D}\x{1C}-\x{20}\x{85}\x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}"#
    static let word = #"\p{L}\p{N}_"#

    static func regex(_ pattern: String) -> NSRegularExpression {
        // Die Muster sind fest und hier geprüft; ein Fehler darin ist ein Programmierfehler.
        try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
    }

    static let quoteHeaders: [NSRegularExpression] = [
        regex(#"^[\#(space)]*On [^\n]{0,300}wrote:[\#(space)]*\z"#),
        regex(#"^[\#(space)]*-{2,}[\#(space)]*(Original|Forwarded) Message[\#(space)]*-{2,}"#),
        regex(#"^[\#(space)]*_{8,}[\#(space)]*\z"#),
        regex(#"^[\#(space)]*From:[\#(space)][^\n]+\z"#),
    ]

    static let signatureMarkers: [NSRegularExpression] = [
        regex(#"^[\#(space)]*--[\#(space)]*\z"#),
        regex(#"^[\#(space)]*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\#(word) ,!.]*\z"#),
        regex(#"^[\#(space)]*sent from my (iphone|android|mobile|ipad)"#),
    ]

    static let disclaimer = regex(
        #"(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|"#
            + #"if you (have )?received this (e-?mail|message) in error)"#)

    static let paragraphBreak = regex(#"\n[\#(space)]*\n"#)
    static let blanks = regex(#"[ \t]+"#)

    /// `re.match`: am Anfang verankert, das Ende offen.
    static func matches(_ pattern: NSRegularExpression, _ text: String) -> Bool {
        let range = NSRange(location: 0, length: (text as NSString).length)
        return pattern.firstMatch(in: text, options: [.anchored], range: range) != nil
    }

    /// `re.search`: irgendwo im Text.
    static func contains(_ pattern: NSRegularExpression, _ text: String) -> Bool {
        let range = NSRange(location: 0, length: (text as NSString).length)
        return pattern.firstMatch(in: text, options: [], range: range) != nil
    }

    // MARK: - Pythons Zeichenkettenmethoden, auf Skalaren statt Graphemen

    static func isSpace(_ scalar: Unicode.Scalar) -> Bool { pythonSpaceScalars.contains(scalar.value) }

    static func strip(_ text: String) -> String { rstrip(lstrip(text)) }

    static func lstrip(_ text: String) -> String {
        String(String.UnicodeScalarView(text.unicodeScalars.drop(while: isSpace)))
    }

    static func rstrip(_ text: String) -> String {
        var scalars = Array(text.unicodeScalars)
        while let last = scalars.last, isSpace(last) { scalars.removeLast() }
        return String(String.UnicodeScalarView(scalars))
    }

    /// `str.replace` auf Skalarebene. `String.replacingOccurrences` vergleicht nach kanonischer
    /// Äquivalenz und behandelt `\r\n` als ein Zeichen; beides ist hier falsch.
    static func replace(_ text: String, _ target: [Unicode.Scalar], _ replacement: [Unicode.Scalar]) -> String {
        let scalars = Array(text.unicodeScalars)
        var out = String.UnicodeScalarView()
        var i = 0
        while i < scalars.count {
            if i + target.count <= scalars.count, Array(scalars[i ..< i + target.count]) == target {
                out.append(contentsOf: replacement)
                i += target.count
            } else {
                out.append(scalars[i])
                i += 1
            }
        }
        return String(out)
    }

    /// `str.split("\n")`, auf Skalaren. Swifts `split` sähe in `\r\n` ein einziges Zeichen.
    static func splitLines(_ text: String) -> [String] {
        var out: [String] = []
        var current = String.UnicodeScalarView()
        for scalar in text.unicodeScalars {
            if scalar == "\n" {
                out.append(String(current))
                current = String.UnicodeScalarView()
            } else {
                current.append(scalar)
            }
        }
        out.append(String(current))
        return out
    }

    /// `re.split` ohne Gruppen.
    static func split(_ text: String, by pattern: NSRegularExpression) -> [String] {
        let ns = text as NSString
        var pieces: [String] = []
        var last = 0
        for match in pattern.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            pieces.append(ns.substring(with: NSRange(location: last, length: match.range.location - last)))
            last = match.range.location + match.range.length
        }
        pieces.append(ns.substring(from: last))
        return pieces
    }

    // MARK: - Öffentliche Schnittstelle

    /// `laya.email.clean_email_body`.
    public static func cleanBody(_ body: String, maxCharacters: Int = 3000) -> String {
        var text = replace(body, ["\r", "\n"], ["\n"])
        text = replace(text, ["\r"], ["\n"])
        // Ein wörtliches Backslash-n, wie es aus schlecht exportierten Mails kommt.
        text = replace(text, ["\\", "n"], ["\n"])

        var lines: [String] = []
        for line in splitLines(text) {
            // Der Verlauf beginnt erst nach eigenem Text. Steht ein Zitatkopf ganz oben, bleibt er.
            if !lines.isEmpty, quoteHeaders.contains(where: { matches($0, line) }) { break }
            if lstrip(line).unicodeScalars.first == ">" { continue }
            lines.append(rstrip(line))
        }

        // Signaturen werden nur im hinteren Teil gesucht: ab 60 % der Zeilen, spätestens acht
        // vor dem Ende, frühestens ab Zeile 1. Genau so rechnet laya, samt Abschneiden nach int().
        var cut = lines.count
        let start = max(1, min(Int(Double(lines.count) * 0.6), lines.count - 8))
        if start < lines.count {
            for i in start ..< lines.count
                where strip(lines[i]).unicodeScalars.count <= 40
                && signatureMarkers.contains(where: { matches($0, lines[i]) }) {
                cut = i
                break
            }
        }
        lines = Array(lines.prefix(cut))

        let paragraphs = split(lines.joined(separator: "\n"), by: paragraphBreak)
            .filter { !contains(disclaimer, $0) }
        let joined = paragraphs.map(strip).filter { !$0.isEmpty }.joined(separator: "\n\n")
        let ns = joined as NSString
        let collapsed = blanks.stringByReplacingMatches(
            in: joined, range: NSRange(location: 0, length: ns.length), withTemplate: " ")
        return String(String.UnicodeScalarView(collapsed.unicodeScalars.prefix(maxCharacters)))
    }

    /// `laya.email.email_state`: Betreff, bereinigter Text, Absender und weitere Felder als
    /// Zustand. Felder mit `null` fallen weg; ein weiteres Feld mit einem schon vorhandenen
    /// Namen ersetzt dessen Wert an dessen Stelle, wie `dict.update` in Python.
    public static func state(subject: String?, body: String?, sender: String? = nil,
                             clean: Bool = true,
                             extra: [(key: String, value: JevValue)] = []) -> JevValue {
        var pairs: [(key: String, value: JevValue)] = [
            ("subject", .string(strip(subject ?? ""))),
            ("body", .string(clean ? cleanBody(body ?? "") : (body ?? ""))),
        ]
        if let sender, !sender.isEmpty { pairs.append(("from", .string(sender))) }
        for (key, value) in extra {
            if case .null = value { continue }
            if let index = pairs.firstIndex(where: { $0.key == key }) {
                pairs[index].value = value
            } else {
                pairs.append((key, value))
            }
        }
        return .object(pairs)
    }
}
