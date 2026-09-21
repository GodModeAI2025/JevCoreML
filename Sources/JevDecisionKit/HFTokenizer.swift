import Foundation

/// BPE-Tokenizer aus einer `tokenizer.json`, byte-identisch zur Rust-Implementierung von HuggingFace.
///
/// Die Kette ist genau die aus der Datei: Split auf den nicht normalisierten Added-Token, dann NFC,
/// dann Split auf den normalisierten Added-Token, dann der Pretokenizer-Regex, dann die
/// Byte-Level-Abbildung, dann BPE über die Merge-Ränge. Jeder dieser Schritte ist eine Stelle, an
/// der ein Nachbau lautlos abweicht, deshalb prüft `TokenizerParityTests` alle Fälle gegen aus
/// Python gezogene IDs statt gegen Augenschein.
///
/// Zwei Pretokenizer kommen vor und werden aus der Datei erkannt:
/// `Sequence[Split(regex), ByteLevel]` wie bei Qwen3, und `ByteLevel(use_regex: true)` wie bei den
/// ModernBERT-Ablegern. Die beiden Regexe sind nicht dieselben, und der Unterschied ist nicht
/// kosmetisch: Qwen3 trennt jede Ziffer einzeln, GPT-2 fasst Ziffernketten zusammen.
public struct HFTokenizer: Sendable {
    /// ICU definiert `\s` als `[\t\n\f\r\p{Z}]`, die von HuggingFace benutzte Rust-Crate dagegen als
    /// `\p{White_Space}`. Das unterscheidet sich bei U+000B und U+0085. Die ausgeschriebene Klasse
    /// bildet die Rust-Semantik ab, die hier die Referenz ist.
    static let whitespaceClass = #"\x{09}\x{0A}\x{0B}\x{0C}\x{0D}\x{20}\x{85}\p{Z}"#

    public enum Flavor: String, Sendable {
        /// `Sequence[Split(regex, isolated), ByteLevel(use_regex: false)]`
        case qwenSplit
        /// `ByteLevel(use_regex: true)`, der Regex steckt in der Rust-Crate selbst
        case byteLevelRegex
    }

    /// Die Unicode-Version von Oniguruma, der Regex-Engine hinter HuggingFace tokenizers.
    ///
    /// Gemessen über einen Split-Pretokenizer, der Treffer entfernt, für jeden Codepunkt:
    /// Oniguruma kennt alle 4302 Buchstaben aus Unicode 16.0 und keinen aus 17.0.
    static let referenceRegexVersion = (major: 16, minor: 0)

    /// Was ICU unter `\p{L}` und `\p{N}` mehr versteht als Oniguruma.
    ///
    /// Zwei Gruppen, beide gemessen. Zeichen nach Unicode 16.0: ICU kennt hier schon Unicode 17
    /// und zählt 4662 neue Zeichen zu den Buchstaben oder Zahlen, für Oniguruma sind sie nicht
    /// zugewiesen. Und private Codepunkte, denen Apples ICU eigene Eigenschaften gibt, etwa
    /// U+F8A1 bis U+F8A7 als Ziffern; für die Referenz sind sie private Zeichen und nichts sonst.
    /// Ohne Abzug schnitt der Pretokenizer an 4693 Codepunkten anders als die Referenz.
    ///
    /// Aus der Unicode-Version berechnet, nicht als feste Liste: bringt ein späteres macOS
    /// Unicode 18, stimmt die Regel weiter.
    static let icuOnlyScalars: String = {
        var ranges: [(UInt32, UInt32)] = [(0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD)]
        var start: UInt32?
        var previous: UInt32 = 0
        for value in UInt32(0) ... 0x10FFFF {
            if let scalar = Unicode.Scalar(value), let age = scalar.properties.age,
               (age.major, age.minor) > referenceRegexVersion {
                if start == nil { start = value }
                previous = value
            } else if let first = start {
                ranges.append((first, previous))
                start = nil
            }
        }
        if let first = start { ranges.append((first, previous)) }
        return ranges.map { $0.0 == $0.1
            ? String(format: #"\x{%X}"#, $0.0)
            : String(format: #"\x{%X}-\x{%X}"#, $0.0, $0.1) }.joined()
    }()

    /// `\p{L}` und `\p{N}` so, wie Oniguruma sie versteht.
    static let letter = #"[\p{L}--["# + icuOnlyScalars + "]]"
    static let number = #"[\p{N}--["# + icuOnlyScalars + "]]"

    /// Der Split-Regex aus tokenizer.json.
    static let qwenPattern: String = {
        let ws = whitespaceClass
        let l = letter, n = number
        return [
            #"(?i:'s|'t|'re|'ve|'m|'ll|'d)"#,
            #"[^\r\n"# + l + n + "]?" + l + "+",
            n,
            " ?[^" + ws + l + n + #"]+[\r\n]*"#,
            "[" + ws + #"]*[\r\n]+"#,
            "[" + ws + "]+(?![^" + ws + "])",
            "[" + ws + "]+",
        ].joined(separator: "|")
    }()

    /// Der Regex, den `ByteLevel` mit `use_regex: true` fest eingebaut hat. Das ist der von GPT-2:
    /// ohne `(?i:)` vor den Kontraktionen und mit `\p{N}+` statt `\p{N}`.
    static let byteLevelPattern: String = {
        let ws = whitespaceClass
        let l = letter, n = number
        return [
            #"'s|'t|'re|'ve|'m|'ll|'d"#,
            " ?" + l + "+",
            " ?" + n + "+",
            " ?[^" + ws + l + n + "]+",
            "[" + ws + "]+(?![^" + ws + "])",
            "[" + ws + "]+",
        ].joined(separator: "|")
    }()

    /// Bytes, die in gültigem UTF-8 nirgends stehen können: 0xC0 und 0xC1 wären überlange
    /// Kodierungen, 0xF5 bis 0xFF lägen oberhalb von U+10FFFF. Der OLMo-Wortschatz hinter laya
    /// lässt genau diese dreizehn Byte-Token weg, und das ist kein Mangel: ein Swift-`String`
    /// kann sie nicht enthalten, also wird auch nie danach gesucht. Qwens Wortschatz führt alle
    /// 256, deshalb fiel es dort nicht auf.
    static let unreachableInUTF8: Set<Int> = Set([0xC0, 0xC1]).union(0xF5 ... 0xFF)

    struct Merge: Sendable {
        let rank: Int32
        let id: Int32
    }

    /// Ein Added-Token, so wie HuggingFaces `AddedVocabulary` es behandelt.
    struct Added: Sendable {
        let utf8: [UInt8]
        let id: Int32
        /// `lstrip`: der Treffer schluckt das Leerzeichen davor mit. `[MASK]` in BERT-Ablegern.
        let lstrip: Bool
    }

    let vocab: [String: Int32]
    let merges: [UInt64: Merge]
    let byteToID: [Int32]
    let addedByContent: [String: Int32]
    /// Added-Token mit `normalized: false`, gesucht im Rohtext vor NFC.
    let addedRaw: [Added]
    /// Added-Token mit `normalized: true`, gesucht im Text nach NFC.
    let addedNormalized: [Added]
    /// Mögliche erste Bytes beider Listen. Spart im Normalfall jeden Vergleich.
    let rawFirstBytes: [Bool]
    let normalizedFirstBytes: [Bool]
    let splitter: NSRegularExpression
    let specialEscape: NSRegularExpression
    public let flavor: Flavor

    /// Anzahl der Merges, die übersprungen wurden, weil ein Operand fehlte. Muss 0 sein.
    public let droppedMerges: Int

    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        // JSONSerialization entfernt ein fuehrendes U+FEFF aus jeder dekodierten Zeichenkette.
        // Ein Wortschatz, der solche Token fuehrt, faellt dabei lautlos zusammen: zwei Eintraege
        // werden einer, und der letzte gewinnt. Byte-Level-Wortschaetze haben kein U+FEFF, weil
        // das Alphabet nur druckbare Zeichen benutzt, aber geraten wird das hier nicht. Findet
        // sich die Bytefolge in der Datei, ist der schnelle Weg nicht mehr zulaessig.
        if data.range(of: Data([0xEF, 0xBB, 0xBF])) != nil {
            throw JevError.tokenizerFile(
                "die Datei enthaelt U+FEFF; JSONSerialization verschluckt es, "
                    + "dieser Wortschatz braucht den BOM-sicheren Leser")
        }
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw JevError.tokenizerFile("tokenizer.json ist kein Objekt")
        }
        guard let model = root["model"] as? [String: Any],
              let rawVocab = model["vocab"] as? [String: Any],
              let rawMerges = model["merges"] as? [Any] else {
            throw JevError.tokenizerFile("model.vocab oder model.merges fehlt")
        }
        if let type = model["type"] as? String, type != "BPE" {
            throw JevError.tokenizerFile("unerwarteter Modelltyp: \(type)")
        }
        if let ignore = model["ignore_merges"] as? Bool, ignore {
            throw JevError.tokenizerFile("ignore_merges=true wird nicht unterstützt")
        }
        if let byteFallback = model["byte_fallback"] as? Bool, byteFallback {
            throw JevError.tokenizerFile("byte_fallback=true wird nicht unterstützt")
        }
        if let normalizer = root["normalizer"] as? [String: Any],
           let type = normalizer["type"] as? String, type != "NFC" {
            throw JevError.tokenizerFile("unerwarteter Normalizer: \(type)")
        }
        self.flavor = try HFTokenizer.flavor(of: root["pre_tokenizer"])

        var vocab = [String: Int32](minimumCapacity: rawVocab.count * 2)
        for (token, value) in rawVocab {
            if let n = value as? NSNumber { vocab[token] = n.int32Value }
        }
        self.vocab = vocab

        var merges = [UInt64: Merge](minimumCapacity: rawMerges.count * 2)
        var dropped = 0
        for (rank, entry) in rawMerges.enumerated() {
            var left: String?
            var right: String?
            if let pair = entry as? [String], pair.count == 2 {
                left = pair[0]; right = pair[1]
            } else if let text = entry as? String {
                // Ältere Dateien speichern Merges als "a b".
                if let space = text.firstIndex(of: " ") {
                    left = String(text[text.startIndex ..< space])
                    right = String(text[text.index(after: space)...])
                }
            }
            guard let a = left, let b = right,
                  let ia = vocab[a], let ib = vocab[b], let merged = vocab[a + b] else {
                dropped += 1
                continue
            }
            merges[HFTokenizer.key(ia, ib)] = Merge(rank: Int32(rank), id: merged)
        }
        self.merges = merges
        self.droppedMerges = dropped

        let table = HFTokenizer.bytesToUnicode()
        var byteToID = [Int32](repeating: -1, count: 256)
        for b in 0 ..< 256 {
            if let id = vocab[String(table[b])] {
                byteToID[b] = id
            } else if !HFTokenizer.unreachableInUTF8.contains(b) {
                throw JevError.tokenizerFile("Byte-Token für 0x\(String(b, radix: 16)) fehlt im Vokabular")
            }
        }
        self.byteToID = byteToID

        var addedByContent: [String: Int32] = [:]
        var raw: [Added] = []
        var normalized: [Added] = []
        for entry in (root["added_tokens"] as? [[String: Any]] ?? []) {
            guard let content = entry["content"] as? String,
                  let id = (entry["id"] as? NSNumber)?.int32Value else { continue }
            if (entry["rstrip"] as? Bool) == true {
                throw JevError.tokenizerFile("rstrip=true bei Added-Token \(content) wird nicht unterstützt")
            }
            if (entry["single_word"] as? Bool) == true {
                throw JevError.tokenizerFile("single_word=true bei Added-Token \(content) wird nicht unterstützt")
            }
            let token = Added(utf8: Array(content.utf8), id: id,
                              lstrip: (entry["lstrip"] as? Bool) == true)
            addedByContent[content] = id
            // HuggingFace führt zwei Tries: einen über dem Rohtext, einen über dem normalisierten.
            // `normalized: true` heißt, der Token wird erst nach NFC gesucht.
            if (entry["normalized"] as? Bool) == true {
                normalized.append(token)
            } else {
                raw.append(token)
            }
        }
        // Längster Treffer gewinnt, so wie HuggingFaces AddedVocabulary.
        raw.sort { $0.utf8.count > $1.utf8.count }
        normalized.sort { $0.utf8.count > $1.utf8.count }
        self.addedByContent = addedByContent
        self.addedRaw = raw
        self.addedNormalized = normalized
        self.rawFirstBytes = HFTokenizer.firstByteTable(raw)
        self.normalizedFirstBytes = HFTokenizer.firstByteTable(normalized)

        self.splitter = try NSRegularExpression(
            pattern: self.flavor == .qwenSplit ? HFTokenizer.qwenPattern : HFTokenizer.byteLevelPattern)
        self.specialEscape = try NSRegularExpression(pattern: #"<\|([A-Za-z0-9_]+)\|>"#)
    }

    // MARK: - Öffentliche Schnittstelle

    /// Token-IDs für beliebigen Text, inklusive Added-Token-Erkennung.
    /// Entspricht `tok(text, add_special_tokens=False).input_ids`.
    public func encode(_ text: String) -> [Int32] {
        var out: [Int32] = []
        out.reserveCapacity(text.utf8.count / 3 + 8)
        for segment in split(text, on: addedRaw, firstBytes: rawFirstBytes) {
            switch segment {
            case let .token(id):
                out.append(id)
            case let .text(part):
                encodeOrdinary(part, into: &out)
            }
        }
        return out
    }

    /// Token-IDs für aufrufergestellten Text. Entspricht `kev.model.user_tokens`: Zeichenketten der
    /// Form `<|name|>` werden vorher zu `<¦name¦>` entschärft, damit niemand Kevs Delimiter fälschen
    /// und Optionsgrenzen verschieben kann.
    public func encodeUserText(_ text: String) -> [Int32] {
        encode(escapeSpecials(text))
    }

    public func escapeSpecials(_ text: String) -> String {
        let bar = String(Unicode.Scalar(0xA6)!)
        let range = NSRange(text.startIndex ..< text.endIndex, in: text)
        return specialEscape.stringByReplacingMatches(
            in: text, range: range, withTemplate: "<\(bar)$1\(bar)>")
    }

    /// ID eines benannten Sondertokens, etwa `<|fim_prefix|>`.
    public func id(forToken token: String) -> Int32? {
        addedByContent[token] ?? vocab[token]
    }

    public var vocabularySize: Int { vocab.count + addedByContent.count }

    // MARK: - Schritte der Kette

    enum Segment {
        case text(String)
        case token(Int32)
    }

    /// Zerlegt den Text an den Added-Token einer Liste. Längster Treffer an der linkesten Stelle,
    /// wie `AhoCorasick` mit `LeftmostLongest` in der Rust-Fassung.
    func split(_ text: String, on candidates: [Added], firstBytes: [Bool]) -> [Segment] {
        guard !candidates.isEmpty else { return text.isEmpty ? [] : [.text(text)] }
        let bytes = Array(text.utf8)
        var segments: [Segment] = []
        var pending: [UInt8] = []
        var i = 0

        func flush() {
            guard !pending.isEmpty else { return }
            // String(bytes:encoding:) entfernt ein führendes BOM, String(decoding:as:) nicht.
            // Der Python-Tokenizer behält U+FEFF als eigenes Token, also muss es hier stehen bleiben.
            segments.append(.text(String(decoding: pending, as: UTF8.self)))
            pending.removeAll(keepingCapacity: true)
        }

        outer: while i < bytes.count {
            if firstBytes[Int(bytes[i])] {
                for candidate in candidates {
                    let n = candidate.utf8.count
                    if i + n <= bytes.count, Array(bytes[i ..< i + n]) == candidate.utf8 {
                        if candidate.lstrip { HFTokenizer.stripTrailingWhitespace(&pending) }
                        flush()
                        segments.append(.token(candidate.id))
                        i += n
                        continue outer
                    }
                }
            }
            pending.append(bytes[i])
            i += 1
        }
        flush()
        return segments
    }

    /// Unicode-Leerraum im Sinn von Rusts `char::is_whitespace`, das `lstrip` in der Referenz
    /// benutzt. Dieselben 25 Zeichen, die Oniguruma unter `\s` versteht.
    static let whitespaceScalars: Set<UInt32> = [
        0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680,
        0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A,
        0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
    ]

    /// `lstrip` eines Added-Tokens: der Treffer schluckt allen Leerraum davor.
    ///
    /// Die erste Fassung nahm nur Leerzeichen, Tab, LF und CR und behauptete im Kommentar, mehr
    /// tue die Referenz auch nicht. Das stimmte nicht: "a\u{00A0}[MASK]" ergab ein Token zu viel.
    static func stripTrailingWhitespace(_ bytes: inout [UInt8]) {
        guard !bytes.isEmpty else { return }
        var scalars = Array(String(decoding: bytes, as: UTF8.self).unicodeScalars)
        let before = scalars.count
        while let last = scalars.last, whitespaceScalars.contains(last.value) { scalars.removeLast() }
        if scalars.count != before { bytes = Array(String(String.UnicodeScalarView(scalars)).utf8) }
    }

    static func firstByteTable(_ candidates: [Added]) -> [Bool] {
        var table = [Bool](repeating: false, count: 256)
        for c in candidates where !c.utf8.isEmpty { table[Int(c.utf8[0])] = true }
        return table
    }

    /// Welcher Pretokenizer in der Datei steht. Alles andere wird abgelehnt statt geraten.
    static func flavor(of node: Any?) throws -> Flavor {
        guard let node = node as? [String: Any], let type = node["type"] as? String else {
            throw JevError.tokenizerFile("pre_tokenizer fehlt")
        }
        if type == "ByteLevel" {
            if (node["use_regex"] as? Bool) == false {
                throw JevError.tokenizerFile("ByteLevel ohne use_regex und ohne Split davor")
            }
            if (node["add_prefix_space"] as? Bool) == true {
                throw JevError.tokenizerFile("add_prefix_space=true wird nicht unterstützt")
            }
            return .byteLevelRegex
        }
        guard type == "Sequence", let parts = node["pretokenizers"] as? [[String: Any]] else {
            throw JevError.tokenizerFile("unerwarteter Pretokenizer: \(type)")
        }
        let types = parts.compactMap { $0["type"] as? String }
        guard types == ["Split", "ByteLevel"] else {
            throw JevError.tokenizerFile("unerwartete Pretokenizer-Folge: \(types.joined(separator: "+"))")
        }
        if (parts[1]["add_prefix_space"] as? Bool) == true {
            throw JevError.tokenizerFile("add_prefix_space=true wird nicht unterstützt")
        }
        return .qwenSplit
    }

    func pretokenize(_ text: String) -> [String] {
        let ns = text as NSString
        var pieces: [String] = []
        var last = 0
        splitter.enumerateMatches(in: text, range: NSRange(location: 0, length: ns.length)) { match, _, _ in
            guard let match, match.range.length > 0 else { return }
            if match.range.location > last {
                pieces.append(ns.substring(with: NSRange(location: last, length: match.range.location - last)))
            }
            pieces.append(ns.substring(with: match.range))
            last = match.range.location + match.range.length
        }
        if last < ns.length {
            pieces.append(ns.substring(from: last))
        }
        return pieces
    }

    func encodeOrdinary(_ text: String, into out: inout [Int32]) {
        guard !text.isEmpty else { return }
        // NFC, genau wie der Normalizer in tokenizer.json. Danach der zweite Trie-Durchgang:
        // Added-Token mit `normalized: true` werden erst jetzt sichtbar. Bei den
        // ModernBERT-Ablegern sind das 109 Einrückungsketten, die sonst über BPE liefen und
        // andere IDs ergäben.
        let normalized = HFTokenizer.normalizedNFC(text)
        for segment in split(normalized, on: addedNormalized, firstBytes: normalizedFirstBytes) {
            switch segment {
            case let .token(id):
                out.append(id)
            case let .text(part):
                for piece in pretokenize(part) {
                    var symbols: [Int32] = []
                    symbols.reserveCapacity(piece.utf8.count)
                    for byte in piece.utf8 { symbols.append(byteToID[Int(byte)]) }
                    applyMerges(&symbols)
                    out.append(contentsOf: symbols)
                }
            }
        }
    }

    /// Die Unicode-Version der NFC-Tabellen in HuggingFace tokenizers.
    ///
    /// Die Referenz normalisiert mit festen Tabellen, nicht mit denen des Systems. Gemessen mit
    /// `tokenizers.normalizers.NFC`: jede kombinierende Marke bis Unicode 9.0 sortiert sie
    /// kanonisch um, keine einzige der 154 Marken ab 10.0. Zeichen nach 9.0 kennt sie also nicht,
    /// für sie sind das Starter ohne Zerlegung und ohne Komposition.
    static let referenceNFCVersion = (major: 9, minor: 0)

    static func knownToReferenceNFC(_ scalar: Unicode.Scalar) -> Bool {
        guard let age = scalar.properties.age else { return false }
        return (age.major, age.minor) <= referenceNFCVersion
    }

    /// NFC wie die Referenz, aus Foundation und zwei Korrekturen.
    ///
    /// Die erste betrifft die Version. Foundation normalisiert mit den Tabellen des Systems, hier
    /// Unicode 17, die Referenz mit denen von 9.0. Zeichen, die danach dazukamen, gehen deshalb
    /// unverändert durch und trennen den Text in Abschnitte; über so ein Zeichen hinweg wird
    /// weder umsortiert noch komponiert. Für die Abschnitte dazwischen gilt Foundation, und das
    /// ist exakt, denn Unicode garantiert, dass sich die Normalisierung einmal zugewiesener
    /// Zeichen nie mehr ändert. Ohne diese Trennung sortierte Foundation etwa U+08CA vor ein
    /// Kasra, während die Referenz die Reihenfolge lässt, und die Token-IDs wichen ab.
    ///
    /// Die zweite sind drei Fehler in Foundation: astrale Starter, siehe `foundationNFC`, und
    /// zwei bei Hangul, einer beim Zusammensetzen und einer beim Verschlucken von U+11A7.
    static func normalizedNFC(_ text: String) -> String {
        // Alles unterhalb von U+0560 ist entweder vor 9.0 zugewiesen oder gar nicht, und
        // astral ist dort nichts. Lateinische, griechische und kyrillische Texte brauchen
        // deshalb keine Einzelprüfung.
        if text.unicodeScalars.allSatisfy({ $0.value < 0x0560 }) {
            return text.decomposedStringWithCanonicalMapping.precomposedStringWithCanonicalMapping
        }
        var out = String.UnicodeScalarView()
        var segment = String.UnicodeScalarView()
        func flush() {
            guard !segment.isEmpty else { return }
            out.append(contentsOf: foundationNFC(String(segment)).unicodeScalars)
            segment = String.UnicodeScalarView()
        }
        for scalar in text.unicodeScalars {
            // U+11A7 steht für "kein Schlusskonsonant" und komponiert nach dem Standard nie.
            // Foundation verschluckt es aber hinter einer zerlegten LV-Silbe: aus U+1100 U+1161
            // U+11A7 wird U+AC00. Als Starter ohne jede Komposition trennt es exakt so wie ein
            // Zeichen, das die Referenz nicht kennt.
            if knownToReferenceNFC(scalar), scalar.value != 0x11A7 {
                segment.append(scalar)
            } else {
                flush()
                out.append(scalar)
            }
        }
        flush()
        return String(out)
    }

    /// Foundations NFC, abgesichert gegen einen Fehler bei Startern oberhalb der BMP.
    ///
    /// `precomposedStringWithCanonicalMapping` behandelt einen astralen Starter so, als wäre er
    /// auf 16 Bit gekürzt, sobald eine kombinierende Marke folgt: aus U+200C5 U+0301 wird U+01FA,
    /// ein lateinisches Ǻ, weil 0x200C5 & 0xFFFF gerade 0x00C5 ergibt.
    ///
    /// Ob das passiert ist, zeigt der Rückweg. NFC verlässt die kanonische Äquivalenzklasse nie,
    /// also haben Ein- und Ausgabe dieselbe NFD. Nur wenn nicht, wird Stück für Stück gerechnet.
    /// Die erste Fassung hat astrale Starter mit Marken stattdessen grundsätzlich unberührt
    /// gelassen. Das war zu grob: U+2F800 hat eine eigene Zerlegung zu U+4E3D, und Kaithi
    /// U+11099 U+110BA komponiert tatsächlich zu U+1109A. Foundation rechnet beides richtig.
    static func foundationNFC(_ text: String) -> String {
        // Erst zerlegen, dann komponieren, wie der Standard NFC definiert. Foundation komponiert
        // eine schon zusammengesetzte Hangul-Silbe nicht weiter: aus U+AC00 U+11A8 wird nicht
        // U+AC01, während die zerlegte Form U+1100 U+1161 U+11A8 richtig zu U+AC01 wird. Der
        // Umweg über NFD gibt Foundation immer die zerlegte Form.
        let result = text.decomposedStringWithCanonicalMapping.precomposedStringWithCanonicalMapping
        guard text.unicodeScalars.contains(where: { $0.value > 0xFFFF }) else { return result }
        if sameNFD(result, text) { return result }

        // Stück für Stück: Folgen aus Starter und Marken so lange zusammen normalisieren, wie der
        // Rückweg stimmt; so bleiben Kompositionen über mehrere Starter erhalten, etwa bei Hangul.
        // Eine Folge, an der er scheitert, verliert nur die Komposition mit ihrem Starter.
        let scalars = Array(text.unicodeScalars)
        var clusters: [ArraySlice<Unicode.Scalar>] = []
        var i = 0
        while i < scalars.count {
            var end = i + 1
            while end < scalars.count, scalars[end].properties.canonicalCombiningClass != .notReordered {
                end += 1
            }
            clusters.append(scalars[i ..< end])
            i = end
        }
        var out = String.UnicodeScalarView()
        var buffer = String.UnicodeScalarView()
        func flush() {
            guard !buffer.isEmpty else { return }
            out.append(contentsOf: String(buffer).precomposedStringWithCanonicalMapping.unicodeScalars)
            buffer = String.UnicodeScalarView()
        }
        for cluster in clusters {
            var candidate = buffer
            candidate.append(contentsOf: cluster)
            let candidateText = String(candidate)
            if sameNFD(candidateText.precomposedStringWithCanonicalMapping, candidateText) {
                buffer = candidate
                continue
            }
            flush()
            let alone = String(String.UnicodeScalarView(cluster))
            if sameNFD(alone.precomposedStringWithCanonicalMapping, alone) {
                buffer.append(contentsOf: cluster)
            } else {
                // Der Starter für sich, die Marken für sich: die Marken werden kanonisch
                // sortiert, nur die fehlerhafte Komposition mit dem Starter entfällt.
                let starter = String(cluster.first!)
                let marks = String(String.UnicodeScalarView(cluster.dropFirst()))
                out.append(contentsOf: starter.precomposedStringWithCanonicalMapping.unicodeScalars)
                out.append(contentsOf: marks.precomposedStringWithCanonicalMapping.unicodeScalars)
            }
        }
        flush()
        return String(out)
    }

    static func sameNFD(_ a: String, _ b: String) -> Bool {
        Array(a.decomposedStringWithCanonicalMapping.unicodeScalars)
            == Array(b.decomposedStringWithCanonicalMapping.unicodeScalars)
    }

    func applyMerges(_ symbols: inout [Int32]) {
        guard symbols.count > 1 else { return }
        while symbols.count > 1 {
            var bestRank = Int32.max
            var bestKey: UInt64 = 0
            var bestID: Int32 = 0
            var found = false
            for j in 0 ..< (symbols.count - 1) {
                let key = HFTokenizer.key(symbols[j], symbols[j + 1])
                if let merge = merges[key], merge.rank < bestRank {
                    bestRank = merge.rank
                    bestKey = key
                    bestID = merge.id
                    found = true
                }
            }
            guard found else { return }
            var merged: [Int32] = []
            merged.reserveCapacity(symbols.count)
            var j = 0
            while j < symbols.count {
                if j + 1 < symbols.count, HFTokenizer.key(symbols[j], symbols[j + 1]) == bestKey {
                    merged.append(bestID)
                    j += 2
                } else {
                    merged.append(symbols[j])
                    j += 1
                }
            }
            symbols = merged
        }
    }

    @inline(__always)
    static func key(_ a: Int32, _ b: Int32) -> UInt64 {
        UInt64(UInt32(bitPattern: a)) << 32 | UInt64(UInt32(bitPattern: b))
    }

    /// GPT-2-Byte-Level-Tabelle: jedes der 256 Bytes bekommt ein druckbares Zeichen.
    static func bytesToUnicode() -> [Unicode.Scalar] {
        var codes: [UInt32] = []
        codes.append(contentsOf: 33 ... 126)
        codes.append(contentsOf: 161 ... 172)
        codes.append(contentsOf: 174 ... 255)
        var table = [Unicode.Scalar?](repeating: nil, count: 256)
        for c in codes { table[Int(c)] = Unicode.Scalar(c)! }
        var next: UInt32 = 0
        for b in 0 ..< 256 where table[b] == nil {
            table[b] = Unicode.Scalar(256 + next)!
            next += 1
        }
        return table.map { $0! }
    }
}

/// Der Name, unter dem der Tokenizer im kev-Pfad eingeführt wurde. Dort liest er dieselbe Datei
/// wie zuvor, der Typ kann inzwischen nur mehr.
public typealias Qwen3Tokenizer = HFTokenizer
