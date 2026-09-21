import Foundation

/// Minimaler JSON-Leser, der die Schlüsselreihenfolge von Objekten beibehält.
///
/// `JSONSerialization` liefert ein `Dictionary` und wirft die Reihenfolge weg. Für Kev ist das
/// kein Schönheitsfehler: `render` schreibt die Feldnamen in den Prompt, eine andere Reihenfolge
/// ergibt andere Token und damit eine andere Entscheidung.
struct JSONOrderedReader {
    /// Groesste erlaubte Schachtelungstiefe. Der Leser ist rekursiv, und ohne Grenze beendet
    /// ein Koerper aus lauter oeffnenden Klammern den Prozess ueber einen Stapelueberlauf.
    static let maximumDepth = 128

    let bytes: [UInt8]
    var i = 0
    var depth = 0

    init(data: [UInt8]) { self.bytes = data }

    var atEnd: Bool { i >= bytes.count }

    mutating func skipWhitespace() {
        while i < bytes.count, bytes[i] == 0x20 || bytes[i] == 0x09 || bytes[i] == 0x0A || bytes[i] == 0x0D {
            i += 1
        }
    }

    mutating func parseValue() throws -> JevValue {
        skipWhitespace()
        guard i < bytes.count else { throw JevError.invalidJSON("unerwartetes Ende") }
        switch bytes[i] {
        case UInt8(ascii: "{"), UInt8(ascii: "["):
            guard depth < JSONOrderedReader.maximumDepth else {
                throw JevError.invalidJSON("mehr als \(JSONOrderedReader.maximumDepth) Ebenen verschachtelt")
            }
            depth += 1
            defer { depth -= 1 }
            return bytes[i] == UInt8(ascii: "{") ? try parseObject() : try parseArray()
        case UInt8(ascii: "\""): return .string(try parseString())
        case UInt8(ascii: "t"): try expect("true"); return .bool(true)
        case UInt8(ascii: "f"): try expect("false"); return .bool(false)
        case UInt8(ascii: "n"):
            // Pythons json akzeptiert NaN und Infinity als Erweiterung, pydantic reicht sie durch,
            // und kev.api.render gibt sie als nan, inf und -inf aus.
            if matches("null") { try expect("null"); return .null }
            try expect("NaN"); return .double(.nan)
        case UInt8(ascii: "N"): try expect("NaN"); return .double(.nan)
        case UInt8(ascii: "I"): try expect("Infinity"); return .double(.infinity)
        case UInt8(ascii: "-") where matches("-Infinity"):
            try expect("-Infinity"); return .double(-.infinity)
        default: return try parseNumber()
        }
    }

    private func matches(_ literal: String) -> Bool {
        let want = Array(literal.utf8)
        return i + want.count <= bytes.count && Array(bytes[i ..< i + want.count]) == want
    }

    private mutating func expect(_ literal: String) throws {
        let want = Array(literal.utf8)
        guard i + want.count <= bytes.count, Array(bytes[i ..< i + want.count]) == want else {
            throw JevError.invalidJSON("erwartet: \(literal)")
        }
        i += want.count
    }

    private mutating func parseObject() throws -> JevValue {
        i += 1
        var pairs: [(key: String, value: JevValue)] = []
        skipWhitespace()
        if i < bytes.count, bytes[i] == UInt8(ascii: "}") { i += 1; return .object(pairs) }
        while true {
            skipWhitespace()
            let key = try parseString()
            skipWhitespace()
            guard i < bytes.count, bytes[i] == UInt8(ascii: ":") else {
                throw JevError.invalidJSON("Doppelpunkt fehlt")
            }
            i += 1
            let value = try parseValue()
            // Python baut daraus ein dict: der letzte Wert gewinnt, an der Stelle der ersten
            // Nennung. Wer stattdessen beide Paare behaelt, bekommt bei doppelten Kriterien eine
            // Option zu viel und damit ein anderes K, einen anderen Softmax, eine andere Auswahl.
            // Gleich heißt byte-gleich. Swifts `==` vergleicht nach kanonischer Äquivalenz und
            // legte "café" vorkomponiert und zerlegt zu einem Schlüssel zusammen; Python behält
            // beide, also stünde im Zustand ein Feld weniger als in der Referenz.
            if let existing = pairs.firstIndex(where: { $0.key.utf8.elementsEqual(key.utf8) }) {
                pairs[existing].value = value
            } else {
                pairs.append((key: key, value: value))
            }
            skipWhitespace()
            guard i < bytes.count else { throw JevError.invalidJSON("unerwartetes Ende im Objekt") }
            if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
            if bytes[i] == UInt8(ascii: "}") { i += 1; return .object(pairs) }
            throw JevError.invalidJSON("unerwartetes Zeichen im Objekt")
        }
    }

    private mutating func parseArray() throws -> JevValue {
        i += 1
        var items: [JevValue] = []
        skipWhitespace()
        if i < bytes.count, bytes[i] == UInt8(ascii: "]") { i += 1; return .array(items) }
        while true {
            items.append(try parseValue())
            skipWhitespace()
            guard i < bytes.count else { throw JevError.invalidJSON("unerwartetes Ende im Array") }
            if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
            if bytes[i] == UInt8(ascii: "]") { i += 1; return .array(items) }
            throw JevError.invalidJSON("unerwartetes Zeichen im Array")
        }
    }

    mutating func parseString() throws -> String {
        guard i < bytes.count, bytes[i] == UInt8(ascii: "\"") else {
            throw JevError.invalidJSON("Zeichenkette erwartet")
        }
        i += 1
        var scalars = String.UnicodeScalarView()
        var raw: [UInt8] = []

        func flushRaw() {
            guard !raw.isEmpty else { return }
            // String(decoding:as:) statt String(bytes:encoding:): letzteres verschluckt ein BOM.
            scalars.append(contentsOf: String(decoding: raw, as: UTF8.self).unicodeScalars)
            raw.removeAll(keepingCapacity: true)
        }

        while i < bytes.count {
            let b = bytes[i]
            if b == UInt8(ascii: "\"") {
                i += 1
                flushRaw()
                return String(scalars)
            }
            if b == UInt8(ascii: "\\") {
                flushRaw()
                i += 1
                guard i < bytes.count else { throw JevError.invalidJSON("abgeschnittene Escape-Sequenz") }
                let e = bytes[i]
                i += 1
                switch e {
                case UInt8(ascii: "\""): scalars.append("\"")
                case UInt8(ascii: "\\"): scalars.append("\\")
                case UInt8(ascii: "/"): scalars.append("/")
                case UInt8(ascii: "b"): scalars.append(Unicode.Scalar(8))
                case UInt8(ascii: "f"): scalars.append(Unicode.Scalar(12))
                case UInt8(ascii: "n"): scalars.append(Unicode.Scalar(10))
                case UInt8(ascii: "r"): scalars.append(Unicode.Scalar(13))
                case UInt8(ascii: "t"): scalars.append(Unicode.Scalar(9))
                case UInt8(ascii: "u"):
                    let first = try parseHex4()
                    if first >= 0xD800, first <= 0xDBFF,
                       i + 1 < bytes.count, bytes[i] == UInt8(ascii: "\\"), bytes[i + 1] == UInt8(ascii: "u") {
                        i += 2
                        let second = try parseHex4()
                        if second >= 0xDC00, second <= 0xDFFF {
                            let combined = 0x10000 + (UInt32(first - 0xD800) << 10) + UInt32(second - 0xDC00)
                            scalars.append(Unicode.Scalar(combined) ?? Unicode.Scalar(0xFFFD)!)
                        } else {
                            scalars.append(Unicode.Scalar(0xFFFD)!)
                            scalars.append(Unicode.Scalar(second) ?? Unicode.Scalar(0xFFFD)!)
                        }
                    } else {
                        scalars.append(Unicode.Scalar(first) ?? Unicode.Scalar(0xFFFD)!)
                    }
                default:
                    throw JevError.invalidJSON("unbekannte Escape-Sequenz")
                }
                continue
            }
            raw.append(b)
            i += 1
        }
        throw JevError.invalidJSON("nicht abgeschlossene Zeichenkette")
    }

    private mutating func parseHex4() throws -> UInt16 {
        guard i + 4 <= bytes.count else { throw JevError.invalidJSON("abgeschnittene \\u-Sequenz") }
        var v: UInt16 = 0
        for _ in 0 ..< 4 {
            let c = bytes[i]
            let d: UInt16
            switch c {
            case UInt8(ascii: "0") ... UInt8(ascii: "9"): d = UInt16(c - UInt8(ascii: "0"))
            case UInt8(ascii: "a") ... UInt8(ascii: "f"): d = UInt16(c - UInt8(ascii: "a") + 10)
            case UInt8(ascii: "A") ... UInt8(ascii: "F"): d = UInt16(c - UInt8(ascii: "A") + 10)
            default: throw JevError.invalidJSON("keine Hexziffer")
            }
            v = v << 4 | d
            i += 1
        }
        return v
    }

    /// Zahlen nach der JSON-Grammatik, nicht nach „alles, was Double schluckt".
    ///
    /// Vorher sammelte der Leser Ziffern, Punkt, e, E, Plus und Minus in beliebiger Folge und
    /// reichte das Ergebnis weiter. Damit gingen `007`, `+1`, `1.`, `.5` und `1e` durch, die
    /// kein gültiges JSON sind, und der Leser akzeptierte Dokumente, die die Referenz ablehnt.
    private mutating func parseNumber() throws -> JevValue {
        let start = i

        func digit() -> Bool {
            i < bytes.count && bytes[i] >= UInt8(ascii: "0") && bytes[i] <= UInt8(ascii: "9")
        }

        if i < bytes.count, bytes[i] == UInt8(ascii: "-") { i += 1 }

        guard digit() else { throw JevError.invalidJSON("Ziffer erwartet") }
        if bytes[i] == UInt8(ascii: "0") {
            i += 1
            if digit() { throw JevError.invalidJSON("führende Null") }
        } else {
            while digit() { i += 1 }
        }

        var isDouble = false
        if i < bytes.count, bytes[i] == UInt8(ascii: ".") {
            isDouble = true
            i += 1
            guard digit() else { throw JevError.invalidJSON("Ziffer nach dem Punkt erwartet") }
            while digit() { i += 1 }
        }
        if i < bytes.count, bytes[i] == UInt8(ascii: "e") || bytes[i] == UInt8(ascii: "E") {
            isDouble = true
            i += 1
            if i < bytes.count, bytes[i] == UInt8(ascii: "+") || bytes[i] == UInt8(ascii: "-") { i += 1 }
            guard digit() else { throw JevError.invalidJSON("Ziffer im Exponenten erwartet") }
            while digit() { i += 1 }
        }

        let text = String(decoding: bytes[start ..< i], as: UTF8.self)
        if !isDouble {
            if let n = Int(text) { return .int(n) }
            // Jenseits von Int64: die Ziffern behalten, statt über Double Stellen zu verlieren.
            return .bigint(text)
        }
        guard let d = Double(text) else { throw JevError.invalidJSON("ungültige Zahl: \(text)") }
        return .double(d)
    }
}
