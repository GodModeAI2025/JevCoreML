import Foundation

/// Der Inhaltstyp, den die SystemOne-Schnittstelle für Zustand, Anweisungen und Kriterien kennt.
///
/// Entspricht `JSONContent` in `kev/api.py`. Objekte behalten ihre Reihenfolge, weil `render`
/// die Feldnamen als Labels in den Prompt schreibt und eine andere Reihenfolge andere Token ergäbe.
public enum JevValue: Sendable, Equatable {
    case null
    case string(String)
    case int(Int)
    /// Eine Ganzzahl, die nicht in Int64 passt. Python kennt keine Breitenbegrenzung und gibt
    /// sie mit allen Ziffern aus; über Double gerendert verlöre sie ab der 17. Stelle genau das.
    case bigint(String)
    case double(Double)
    case bool(Bool)
    case array([JevValue])
    case object([(key: String, value: JevValue)])

    public static func == (lhs: JevValue, rhs: JevValue) -> Bool {
        switch (lhs, rhs) {
        case (.null, .null): return true
        // Byte-gleich, nicht kanonisch gleich: zwei Zeichenketten, die das Modell als
        // verschiedene Token sieht, sind verschiedene Werte.
        case let (.string(a), .string(b)): return a.utf8.elementsEqual(b.utf8)
        case let (.int(a), .int(b)): return a == b
        case let (.bigint(a), .bigint(b)): return a == b
        case let (.double(a), .double(b)): return a == b
        case let (.bool(a), .bool(b)): return a == b
        case let (.array(a), .array(b)): return a == b
        case let (.object(a), .object(b)):
            return a.count == b.count
                && zip(a, b).allSatisfy { $0.key.utf8.elementsEqual($1.key.utf8) && $0.value == $1.value }
        default: return false
        }
    }
}

extension JevValue: ExpressibleByStringLiteral, ExpressibleByIntegerLiteral,
                    ExpressibleByFloatLiteral, ExpressibleByBooleanLiteral,
                    ExpressibleByNilLiteral, ExpressibleByArrayLiteral,
                    ExpressibleByDictionaryLiteral {
    public init(stringLiteral value: String) { self = .string(value) }
    public init(integerLiteral value: Int) { self = .int(value) }
    public init(floatLiteral value: Double) { self = .double(value) }
    public init(booleanLiteral value: Bool) { self = .bool(value) }
    public init(nilLiteral: ()) { self = .null }
    public init(arrayLiteral elements: JevValue...) { self = .array(elements) }
    /// Achtung: Dictionary-Literale haben keine garantierte Reihenfolge. Für stabile Prompts
    /// `JevValue.object([...])` mit einem Array von Paaren verwenden.
    public init(dictionaryLiteral elements: (String, JevValue)...) {
        self = .object(elements.map { (key: $0.0, value: $0.1) })
    }
}

extension JevValue {
    /// Flacht Text, Objekte und Listen genau so ab, wie `kev.api.render` es tut.
    ///
    /// Die Zahlformatierung folgt Pythons `str()`: Ganzzahlen ohne Nachkommastelle,
    /// Gleitkommazahlen immer mit, Wahrheitswerte gross geschrieben.
    public func render(indent: Int = 0) -> String {
        let pad = String(repeating: "  ", count: indent)
        switch self {
        case .null:
            return ""
        case let .string(s):
            return s
        case let .int(i):
            return String(i)
        case let .bigint(digits):
            return digits
        case let .double(d):
            return JevValue.pythonFloat(d)
        case let .bool(b):
            return b ? "True" : "False"
        case let .array(items):
            return items.map { item in
                let body = item.render(indent: indent + 1)
                return "\(pad)- \(JevValue.lstrip(body))"
            }.joined(separator: "\n")
        case let .object(pairs):
            return pairs.map { key, value in
                switch value {
                case .object, .array:
                    return "\(pad)\(key):\n\(value.render(indent: indent + 1))"
                default:
                    return "\(pad)\(key): \(value.render())"
                }
            }.joined(separator: "\n")
        }
    }

    /// Wie Pythons `str.lstrip()` ohne Argument, also je Codepoint und nach `str.isspace()`.
    ///
    /// `Character.isWhitespace` beurteilt ein ganzes Graphemcluster und folgt der Eigenschaft
    /// White_Space. Python nimmt zusätzlich die vier Informationstrenner U+001C bis U+001F,
    /// die in einem aus JSON gelesenen Zustand durchaus vorkommen können.
    static func isPythonSpace(_ scalar: Unicode.Scalar) -> Bool {
        (0x1C ... 0x1F).contains(scalar.value) || scalar.properties.isWhitespace
    }

    static func lstrip(_ s: String) -> String {
        let scalars = s.unicodeScalars
        guard let i = scalars.firstIndex(where: { !isPythonSpace($0) }) else { return "" }
        return String(String.UnicodeScalarView(scalars[i...]))
    }

    /// Pythons `str(float)`: kürzeste Darstellung, die den Wert zurückliest, aber nie ohne Punkt.
    static func pythonFloat(_ d: Double) -> String {
        if d.isNaN { return "nan" }
        if d.isInfinite { return d > 0 ? "inf" : "-inf" }
        var s = "\(d)"
        if s.hasSuffix(".0") && abs(d) >= 1e16 {
            s = String(format: "%.16g", d)
        }
        if s.contains("e") {
            // Swift schreibt 1e+20, Python ebenso; Swift schreibt aber 1e-05 als 1e-05.
            return s
        }
        return s
    }
}

extension JevValue {
    /// Baut den Wert aus einem bereits geparsten JSON-Objekt. Reihenfolge bleibt erhalten,
    /// indem die Rohbytes noch einmal gelesen werden, statt `JSONSerialization` zu vertrauen.
    public static func parse(json data: Data) throws -> JevValue {
        var reader = JSONOrderedReader(data: Array(data))
        let value = try reader.parseValue()
        reader.skipWhitespace()
        guard reader.atEnd else { throw JevError.invalidJSON("Zeichen nach dem Ende des Dokuments") }
        return value
    }

    public static func parse(json string: String) throws -> JevValue {
        try parse(json: Data(string.utf8))
    }
}
