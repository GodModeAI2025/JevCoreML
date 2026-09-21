import Foundation

/// Was der laya-Encoder von einem Tokenizer braucht.
public protocol TextTokenizer: Sendable {
    func encode(_ text: String) -> [Int32]
    func id(forToken token: String) -> Int32?
}

extension HFTokenizer: TextTokenizer {}

/// BPE über Zeichen statt über Bytes, mit Metaspace und Byte-Rückfall.
///
/// Das ist die SentencePiece-Linie: Leerzeichen werden zu `▁`, der Text wird an diesen Marken
/// zerlegt, und jedes Zeichen, das nicht im Wortschatz steht, zerfällt in `<0xXX>`-Token für seine
/// UTF-8-Bytes. Der mehrsprachige laya-Checkpoint benutzt sie, der englische nicht, und die
/// beiden Wege haben außer der Merge-Schleife nichts gemeinsam.
public struct MetaspaceTokenizer: Sendable, TextTokenizer {
    struct Merge: Sendable {
        let rank: Int32
        let id: Int32
    }

    struct Added: Sendable {
        let utf8: [UInt8]
        let id: Int32
        /// Der Treffer schluckt den Leerraum davor, wie in HuggingFace. mmBERTs `<mask>` hat das.
        let lstrip: Bool
    }

    static let replacement: Unicode.Scalar = "\u{2581}"

    /// Wortschatz über Bytes, nicht über `String`. Swift vergleicht Zeichenketten nach
    /// kanonischer Äquivalenz, und dieser Wortschatz führt U+0341 neben U+0301 sowie U+2126
    /// neben U+03A9. Als `String` geschlüsselt fielen sie paarweise zusammen: aus "Ω" wurde
    /// die ID von "℧", und "▁" plus "Ω" verschmolz zu einem Merge, den es nicht gibt.
    let vocab: [[UInt8]: Int32]
    /// Der heiße Pfad: ein einzelner Skalar direkt auf seine ID, ohne ein Byte-Array zu bauen.
    let scalarID: [UInt32: Int32]
    let merges: [UInt64: Merge]
    let added: [Added]
    let addedByContent: [[UInt8]: Int32]
    let addedFirstBytes: [Bool]
    /// `<0xXX>` je Byte, -1 wenn der Wortschatz ihn nicht führt.
    let byteTokenID: [Int32]
    let unkID: Int32
    let byteFallback: Bool
    let fuseUnk: Bool
    /// Merges, deren Operanden fehlten. Muss 0 sein.
    public let droppedMerges: Int

    public init(contentsOf url: URL) throws {
        let top = try TokenizerJSON.parse(contentsOf: url)
        guard let model = top["model"], let rawVocab = model["vocab"]?.object,
              let rawMerges = model["merges"]?.array else {
            throw JevError.tokenizerFile("model.vocab oder model.merges fehlt")
        }
        if let type = model["type"]?.string, type != "BPE" {
            throw JevError.tokenizerFile("unerwarteter Modelltyp: \(type)")
        }
        for key in ["continuing_subword_prefix", "end_of_word_suffix"] {
            if let value = model[key]?.string, !value.isEmpty {
                throw JevError.tokenizerFile("\(key) wird nicht unterstützt")
            }
        }
        if model["ignore_merges"]?.bool == true {
            throw JevError.tokenizerFile("ignore_merges=true wird nicht unterstützt")
        }

        // Normalizer und Pretokenizer müssen die sein, für die dieser Code geschrieben ist.
        let marker = String(MetaspaceTokenizer.replacement)
        let normalizer = top["normalizer"]
        guard normalizer?["type"]?.string == "Replace",
              normalizer?["pattern"]?["String"]?.string == " ",
              normalizer?["content"]?.string == marker else {
            throw JevError.tokenizerFile("erwartet wird Replace(\" \" -> \"\u{2581}\") als Normalizer")
        }
        let pre = top["pre_tokenizer"]
        guard pre?["type"]?.string == "Metaspace",
              pre?["replacement"]?.string == marker,
              pre?["split"]?.bool != false else {
            throw JevError.tokenizerFile("erwartet wird Metaspace mit split=true")
        }
        guard pre?["prepend_scheme"]?.string == "always" else {
            throw JevError.tokenizerFile("nur prepend_scheme=always wird unterstützt")
        }

        var vocab = [[UInt8]: Int32](minimumCapacity: rawVocab.count * 2)
        var scalarID = [UInt32: Int32](minimumCapacity: 4096)
        for pair in rawVocab {
            guard let raw = pair.value.int, let id = Int32(exactly: raw) else {
                throw JevError.tokenizerFile("ungültige ID im Wortschatz")
            }
            vocab[pair.key] = id
            let decoded = String(decoding: pair.key, as: UTF8.self)
            if decoded.unicodeScalars.count == 1, Array(decoded.utf8) == pair.key {
                scalarID[decoded.unicodeScalars.first!.value] = id
            }
        }
        self.scalarID = scalarID

        var merges = [UInt64: Merge](minimumCapacity: rawMerges.count * 2)
        var dropped = 0
        for (rank, entry) in rawMerges.enumerated() {
            var left: [UInt8]?
            var right: [UInt8]?
            if let pair = entry.array, pair.count == 2 {
                left = pair[0].string.map { Array($0.utf8) }
                right = pair[1].string.map { Array($0.utf8) }
            } else if let line = entry.string {
                // Ältere Dateien schreiben das Paar als eine Zeile mit Leerzeichen dazwischen.
                let raw = Array(line.utf8)
                if let space = raw.firstIndex(of: UInt8(ascii: " ")) {
                    left = Array(raw[..<space])
                    right = Array(raw[(space + 1)...])
                }
            }
            guard let a = left, let b = right,
                  let leftID = vocab[a], let rightID = vocab[b], let mergedID = vocab[a + b] else {
                dropped += 1
                continue
            }
            let key = HFTokenizer.key(leftID, rightID)
            if merges[key] == nil {
                merges[key] = Merge(rank: Int32(rank), id: mergedID)
            }
        }
        self.droppedMerges = dropped
        self.vocab = vocab
        self.merges = merges

        var added: [Added] = []
        var byContent: [[UInt8]: Int32] = [:]
        if let entries = top["added_tokens"]?.array {
            for entry in entries {
                guard let content = entry["content"]?.string, let raw = entry["id"]?.int,
                      let id = Int32(exactly: raw) else {
                    throw JevError.tokenizerFile("Added-Token ohne gültige ID")
                }
                if entry["normalized"]?.bool == true {
                    throw JevError.tokenizerFile("normalized=true bei Added-Token \(content) wird hier nicht erwartet")
                }
                // Was hier nicht nachgebaut ist, wird abgelehnt statt still anders gerechnet.
                for flag in ["rstrip", "single_word"] where entry[flag]?.bool == true {
                    throw JevError.tokenizerFile("\(flag)=true bei Added-Token \(content) wird nicht unterstützt")
                }
                added.append(Added(utf8: Array(content.utf8), id: id, lstrip: entry["lstrip"]?.bool == true))
                byContent[Array(content.utf8)] = id
            }
        }
        added.sort { $0.utf8.count > $1.utf8.count }
        self.added = added
        self.addedByContent = byContent
        var firstBytes = [Bool](repeating: false, count: 256)
        for token in added where !token.utf8.isEmpty { firstBytes[Int(token.utf8[0])] = true }
        self.addedFirstBytes = firstBytes

        self.byteFallback = model["byte_fallback"]?.bool ?? false
        self.fuseUnk = model["fuse_unk"]?.bool ?? false
        let unkToken = model["unk_token"]?.string.map { Array($0.utf8) }
        self.unkID = unkToken.flatMap { vocab[$0] ?? byContent[$0] } ?? -1
        var byteIDs = [Int32](repeating: -1, count: 256)
        for b in 0 ..< 256 {
            let name = Array(String(format: "<0x%02X>", b).utf8)
            byteIDs[b] = vocab[name] ?? byContent[name] ?? -1
        }
        self.byteTokenID = byteIDs
    }

    public var vocabularySize: Int { vocab.count + addedByContent.count }

    public func id(forToken token: String) -> Int32? {
        let key = Array(token.utf8)
        return addedByContent[key] ?? vocab[key]
    }

    public func encode(_ text: String) -> [Int32] {
        guard !text.isEmpty else { return [] }
        var out: [Int32] = []
        out.reserveCapacity(text.utf8.count / 3 + 8)
        for segment in split(text) {
            switch segment {
            case let .token(id):
                out.append(id)
            case let .text(part):
                encodeOrdinary(part, into: &out)
            }
        }
        return out
    }

    enum Segment {
        case text(String)
        case token(Int32)
    }

    func split(_ text: String) -> [Segment] {
        guard !added.isEmpty else { return [.text(text)] }
        let bytes = Array(text.utf8)
        var segments: [Segment] = []
        var pending: [UInt8] = []
        var i = 0

        func flush() {
            guard !pending.isEmpty else { return }
            segments.append(.text(String(decoding: pending, as: UTF8.self)))
            pending.removeAll(keepingCapacity: true)
        }

        outer: while i < bytes.count {
            if addedFirstBytes[Int(bytes[i])] {
                for candidate in added {
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

    func encodeOrdinary(_ text: String, into out: inout [Int32]) {
        guard !text.isEmpty else { return }
        for piece in MetaspaceTokenizer.pretokenize(text) {
            var symbols: [Int32] = []
            symbols.reserveCapacity(piece.unicodeScalars.count)
            for scalar in piece.unicodeScalars {
                appendSymbol(for: scalar, into: &symbols)
            }
            applyMerges(&symbols)
            out.append(contentsOf: symbols)
        }
    }

    /// Normalizer und Metaspace in einem Schritt: Leerzeichen zu `▁`, vorne eines anfügen, wenn
    /// noch keines steht, und an jeder Marke trennen, wobei die Marke zum folgenden Stück gehört.
    static func pretokenize(_ text: String) -> [String] {
        var scalars = Array(text.unicodeScalars)
        for i in scalars.indices where scalars[i] == " " { scalars[i] = replacement }
        if scalars.first != replacement { scalars.insert(replacement, at: 0) }
        var pieces: [String] = []
        var current = String.UnicodeScalarView()
        for scalar in scalars {
            if scalar == replacement, !current.isEmpty {
                pieces.append(String(current))
                current = String.UnicodeScalarView()
            }
            current.append(scalar)
        }
        if !current.isEmpty { pieces.append(String(current)) }
        return pieces
    }

    private func appendSymbol(for scalar: Unicode.Scalar, into symbols: inout [Int32]) {
        if let id = scalarID[scalar.value] {
            symbols.append(id)
            return
        }
        if byteFallback {
            let bytes = Array(String(scalar).utf8)
            if bytes.allSatisfy({ byteTokenID[Int($0)] >= 0 }) {
                for byte in bytes { symbols.append(byteTokenID[Int(byte)]) }
                return
            }
        }
        // fuse_unk: mehrere unbekannte Zeichen hintereinander werden ein einziges Unbekannt.
        if fuseUnk, symbols.last == unkID { return }
        symbols.append(unkID)
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
}
