import Foundation

/// JSON-Leser für Tokenizer-Dateien.
///
/// Zwei Gründe, warum weder `JSONSerialization` noch `JSONOrderedReader` hier passen:
///
/// `JSONSerialization` entfernt ein führendes U+FEFF aus jeder dekodierten Zeichenkette. Der
/// mmBERT-Wortschatz führt Token wie `U+FEFF #` neben `#`, die fallen dabei zusammen, und der
/// zuletzt gelesene gewinnt. Messbar war das an einer URL: `#` bekam die ID von `U+FEFF #`.
///
/// `JSONOrderedReader` hält die Schlüsselreihenfolge und sucht dafür je Schlüssel linear nach
/// einem Duplikat. Bei 256.000 Einträgen ist das quadratisch und läuft nicht mehr durch.
///
/// Und beide würden die Schlüssel als `String` ablegen. Swift vergleicht Zeichenketten nach
/// kanonischer Äquivalenz, ein Wortschatz braucht aber Byte-Identität: U+2126 und U+03A9 sind
/// für Swift derselbe Schlüssel, ebenso U+0341 und U+0301. Im mmBERT-Wortschatz stehen alle
/// vier als eigene Token, und beim Einlesen in ein `[String: …]` überschrieb jeweils der
/// zweite den ersten. Deshalb bleiben Schlüssel hier Bytes.
enum TokenizerJSON {
    indirect enum Node {
        case object([(key: [UInt8], value: Node)])
        case array([Node])
        case string(String)
        case number(Double)
        case bool(Bool)
        case null

        var string: String? { if case let .string(s) = self { return s }; return nil }
        var bool: Bool? { if case let .bool(b) = self { return b }; return nil }
        /// Nur endliche Ganzzahlen in sicherem Bereich. `Int(1e400)` hielte den Prozess an,
        /// statt einen Fehler zu werfen.
        var int: Int? {
            if case let .number(d) = self, d.isFinite, d.rounded() == d, abs(d) <= 9_007_199_254_740_992 {
                return Int(d)
            }
            return nil
        }
        var array: [Node]? { if case let .array(a) = self { return a }; return nil }
        var object: [(key: [UInt8], value: Node)]? {
            if case let .object(o) = self { return o }
            return nil
        }

        /// Für die kleinen Konfigurationsobjekte. Über dem Wortschatz wird nie so gesucht,
        /// dort wird iteriert.
        subscript(key: String) -> Node? {
            guard case let .object(pairs) = self else { return nil }
            let want = Array(key.utf8)
            return pairs.first { $0.key == want }?.value
        }
    }

    static func parse(contentsOf url: URL) throws -> Node {
        var parser = Parser(bytes: Array(try Data(contentsOf: url)))
        parser.skipWhitespace()
        let value = try parser.parseValue()
        parser.skipWhitespace()
        guard parser.index == parser.bytes.count else {
            throw JevError.tokenizerFile("hinter dem JSON steht noch etwas")
        }
        return value
    }

    struct Parser {
        let bytes: [UInt8]
        var index = 0
        /// Verschachtelungstiefe. Ohne Grenze läuft eine Datei aus 300.000 `[` in einen
        /// Stapelüberlauf, und der Prozess endet mit SIGSEGV statt mit einem Fehler.
        var depth = 0
        static let maximumDepth = 256

        mutating func skipWhitespace() {
            while index < bytes.count {
                switch bytes[index] {
                case 0x20, 0x09, 0x0A, 0x0D: index += 1
                default: return
                }
            }
        }

        mutating func parseValue() throws -> Node {
            guard index < bytes.count else { throw JevError.tokenizerFile("Datei endet zu früh") }
            switch bytes[index] {
            case UInt8(ascii: "{"), UInt8(ascii: "["):
                depth += 1
                defer { depth -= 1 }
                guard depth <= Parser.maximumDepth else {
                    throw JevError.tokenizerFile("JSON tiefer als \(Parser.maximumDepth) Ebenen")
                }
                return bytes[index] == UInt8(ascii: "{") ? try parseObject() : try parseArray()
            case UInt8(ascii: "\""): return .string(try parseString())
            case UInt8(ascii: "t"): try expect("true"); return .bool(true)
            case UInt8(ascii: "f"): try expect("false"); return .bool(false)
            case UInt8(ascii: "n"): try expect("null"); return .null
            default: return .number(try parseNumber())
            }
        }

        mutating func expect(_ word: String) throws {
            let raw = Array(word.utf8)
            guard index + raw.count <= bytes.count,
                  Array(bytes[index ..< index + raw.count]) == raw else {
                throw JevError.tokenizerFile("erwartet wurde \(word)")
            }
            index += raw.count
        }

        mutating func parseObject() throws -> Node {
            index += 1
            var out: [(key: [UInt8], value: Node)] = []
            skipWhitespace()
            if index < bytes.count, bytes[index] == UInt8(ascii: "}") { index += 1; return .object(out) }
            while true {
                skipWhitespace()
                let key = try parseStringBytes()
                skipWhitespace()
                guard index < bytes.count, bytes[index] == UInt8(ascii: ":") else {
                    throw JevError.tokenizerFile("Doppelpunkt fehlt")
                }
                index += 1
                skipWhitespace()
                out.append((key: key, value: try parseValue()))
                skipWhitespace()
                guard index < bytes.count else { throw JevError.tokenizerFile("Objekt nicht geschlossen") }
                if bytes[index] == UInt8(ascii: ",") { index += 1; continue }
                if bytes[index] == UInt8(ascii: "}") { index += 1; return .object(out) }
                throw JevError.tokenizerFile("Komma oder schließende Klammer fehlt")
            }
        }

        mutating func parseArray() throws -> Node {
            index += 1
            var out: [Node] = []
            skipWhitespace()
            if index < bytes.count, bytes[index] == UInt8(ascii: "]") { index += 1; return .array(out) }
            while true {
                skipWhitespace()
                out.append(try parseValue())
                skipWhitespace()
                guard index < bytes.count else { throw JevError.tokenizerFile("Array nicht geschlossen") }
                if bytes[index] == UInt8(ascii: ",") { index += 1; continue }
                if bytes[index] == UInt8(ascii: "]") { index += 1; return .array(out) }
                throw JevError.tokenizerFile("Komma oder schließende Klammer fehlt")
            }
        }

        mutating func parseString() throws -> String {
            // String(decoding:as:) statt String(bytes:encoding:): letzteres verschluckt ein
            // führendes BOM, und genau davon lebt dieser Wortschatz.
            String(decoding: try parseStringBytes(), as: UTF8.self)
        }

        mutating func parseStringBytes() throws -> [UInt8] {
            guard index < bytes.count, bytes[index] == UInt8(ascii: "\"") else {
                throw JevError.tokenizerFile("Zeichenkette erwartet")
            }
            index += 1
            var raw: [UInt8] = []
            var pendingHighSurrogate: UInt32?

            func flushSurrogate() {
                // Ein einzelnes hohes Ersatzzeichen ohne Partner: als Ersatzzeichen ablegen,
                // so wie es jeder JSON-Leser tut, statt die Datei abzulehnen.
                if pendingHighSurrogate != nil {
                    raw.append(contentsOf: Array("\u{FFFD}".utf8))
                    pendingHighSurrogate = nil
                }
            }

            while index < bytes.count {
                let byte = bytes[index]
                if byte == UInt8(ascii: "\"") {
                    index += 1
                    flushSurrogate()
                    return raw
                }
                if byte != UInt8(ascii: "\\") {
                    flushSurrogate()
                    raw.append(byte)
                    index += 1
                    continue
                }
                index += 1
                guard index < bytes.count else { break }
                let escape = bytes[index]
                index += 1
                switch escape {
                case UInt8(ascii: "u"):
                    guard index + 4 <= bytes.count, let value = hex(bytes[index ..< index + 4]) else {
                        throw JevError.tokenizerFile("kaputte \\u-Folge")
                    }
                    index += 4
                    if let high = pendingHighSurrogate {
                        if value >= 0xDC00 && value <= 0xDFFF {
                            let scalar = 0x10000 + ((high - 0xD800) << 10) + (value - 0xDC00)
                            raw.append(contentsOf: Array(String(Unicode.Scalar(scalar)!).utf8))
                            pendingHighSurrogate = nil
                            continue
                        }
                        flushSurrogate()
                    }
                    if value >= 0xD800 && value <= 0xDBFF {
                        pendingHighSurrogate = value
                    } else if value >= 0xDC00 && value <= 0xDFFF {
                        raw.append(contentsOf: Array("\u{FFFD}".utf8))
                    } else {
                        raw.append(contentsOf: Array(String(Unicode.Scalar(value)!).utf8))
                    }
                default:
                    flushSurrogate()
                    switch escape {
                    case UInt8(ascii: "\""): raw.append(UInt8(ascii: "\""))
                    case UInt8(ascii: "\\"): raw.append(UInt8(ascii: "\\"))
                    case UInt8(ascii: "/"): raw.append(UInt8(ascii: "/"))
                    case UInt8(ascii: "b"): raw.append(0x08)
                    case UInt8(ascii: "f"): raw.append(0x0C)
                    case UInt8(ascii: "n"): raw.append(0x0A)
                    case UInt8(ascii: "r"): raw.append(0x0D)
                    case UInt8(ascii: "t"): raw.append(0x09)
                    default: throw JevError.tokenizerFile("unbekannte Escape-Folge")
                    }
                }
            }
            throw JevError.tokenizerFile("Zeichenkette nicht geschlossen")
        }

        func hex(_ slice: ArraySlice<UInt8>) -> UInt32? {
            var value: UInt32 = 0
            for byte in slice {
                let digit: UInt32
                switch byte {
                case UInt8(ascii: "0") ... UInt8(ascii: "9"): digit = UInt32(byte - UInt8(ascii: "0"))
                case UInt8(ascii: "a") ... UInt8(ascii: "f"): digit = UInt32(byte - UInt8(ascii: "a")) + 10
                case UInt8(ascii: "A") ... UInt8(ascii: "F"): digit = UInt32(byte - UInt8(ascii: "A")) + 10
                default: return nil
                }
                value = value << 4 | digit
            }
            return value
        }

        mutating func parseNumber() throws -> Double {
            let start = index
            while index < bytes.count, TokenizerJSON.Parser.isNumberByte(bytes[index]) {
                index += 1
            }
            guard index > start,
                  let value = Double(String(decoding: bytes[start ..< index], as: UTF8.self)) else {
                throw JevError.tokenizerFile("keine gültige Zahl")
            }
            return value
        }

        static func isNumberByte(_ b: UInt8) -> Bool {
            switch b {
            case UInt8(ascii: "0") ... UInt8(ascii: "9"),
                 UInt8(ascii: "-"), UInt8(ascii: "+"), UInt8(ascii: "."),
                 UInt8(ascii: "e"), UInt8(ascii: "E"):
                return true
            default:
                return false
            }
        }
    }
}
