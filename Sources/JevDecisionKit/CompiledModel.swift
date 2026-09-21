import CoreML
import CryptoKit
import Foundation

/// Übersetzt ein `.mlpackage` bei Bedarf nach `.mlmodelc` und merkt sich das Ergebnis.
///
/// `MLModel(contentsOf:)` lädt nur kompilierte Modelle. `MLModel.compileModel(at:)` legt das
/// Ergebnis in einem temporären Verzeichnis ab, das das System jederzeit löschen darf, deshalb
/// wird es hier in den Caches-Ordner umgezogen und über Pfad und Änderungsdatum wiedererkannt.
public enum CompiledModel {
    public static func url(for model: URL) throws -> URL {
        if model.pathExtension == "mlmodelc" { return model }
        guard FileManager.default.fileExists(atPath: model.path) else {
            throw JevError.modelFile("nicht gefunden: \(model.path)")
        }

        let cache = try cacheDirectory()
        let target = cache.appendingPathComponent(cacheName(for: model)).appendingPathExtension("mlmodelc")
        if FileManager.default.fileExists(atPath: target.path) { return target }

        let compiled = try MLModel.compileModel(at: model)

        // Atomar einhängen statt löschen und dann verschieben. Zwei Prozesse, die denselben
        // Eintrag noch nicht gefunden haben, kompilieren parallel; im alten Ablauf lag zwischen
        // removeItem und moveItem ein Fenster, in dem der Eintrag gar nicht existierte und ein
        // dritter Prozess ihn mitten im Laden verlor.
        let staging = target.deletingLastPathComponent()
            .appendingPathComponent("\(target.lastPathComponent).\(ProcessInfo.processInfo.processIdentifier).staging")
        try? FileManager.default.removeItem(at: staging)
        do {
            try FileManager.default.moveItem(at: compiled, to: staging)
        } catch {
            // Wenn schon der Umzug in den Cache scheitert, ist der temporäre Pfad brauchbar.
            return compiled
        }

        if FileManager.default.fileExists(atPath: target.path) {
            // Ein anderer Prozess war schneller. Dessen Eintrag hat denselben Inhaltshash,
            // also ist er gleichwertig; der eigene wird verworfen.
            try? FileManager.default.removeItem(at: staging)
            return target
        }
        do {
            try FileManager.default.moveItem(at: staging, to: target)
            evictOlderEntries(of: model, keeping: target, in: cache)
            return target
        } catch {
            // Rennen verloren: der Eintrag wurde zwischenzeitlich angelegt.
            try? FileManager.default.removeItem(at: staging)
            return FileManager.default.fileExists(atPath: target.path) ? target : compiled
        }
    }

    /// Entfernt ältere Einträge desselben Modells.
    ///
    /// Der Schlüssel hängt am Inhalt, also ergibt jeder neue Export einen neuen Eintrag, und die
    /// alten blieben liegen. Nach zwei Tagen Arbeit waren das 44 Einträge und 51 GB, davon 37 GB
    /// für Fassungen, die kein `.mlpackage` mehr hatte. Die Platte lief voll.
    ///
    /// Weg kommt nur, was denselben Modellnamen trägt und einen anderen Schlüssel hat. Ein Prozess,
    /// der einen alten Eintrag gerade geladen hat, verliert nichts: geladene Gewichte sind
    /// eingeblendet, und die Datei verschwindet erst, wenn er sie freigibt.
    static func evictOlderEntries(of model: URL, keeping target: URL, in cache: URL) {
        let prefix = model.deletingPathExtension().lastPathComponent + "-"
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: cache.path) else { return }
        for name in names where name.hasPrefix(prefix) && name.hasSuffix(".mlmodelc")
            && name != target.lastPathComponent {
            // Nur Schlüssel aus 16 Hexziffern: "Kev06B-fp16-" darf nicht "Kev06B-fp16-palette8-…" treffen.
            let key = name.dropFirst(prefix.count).dropLast(".mlmodelc".count)
            guard key.count == 16, key.allSatisfy(\.isHexDigit) else { continue }
            try? FileManager.default.removeItem(at: cache.appendingPathComponent(name))
        }
    }

    private static func cacheDirectory() throws -> URL {
        let base = try FileManager.default.url(for: .cachesDirectory, in: .userDomainMask,
                                               appropriateFor: nil, create: true)
        let dir = base.appendingPathComponent("JevDecisionKit", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Schlüssel aus dem Inhalt des Pakets, nicht aus dem Datum des Wurzelverzeichnisses.
    ///
    /// Unter APFS ändert sich das Datum eines Verzeichnisses nur, wenn Einträge darin angelegt,
    /// gelöscht oder umbenannt werden. Wer die Dateien im `.mlpackage` an Ort und Stelle
    /// überschreibt, etwa mit `cp -R neu/. alt/` oder `rsync --inplace`, bekam denselben
    /// Schlüssel und damit das alte kompilierte Modell, ohne jeden Hinweis.
    private static func cacheName(for model: URL) -> String {
        var hasher = SHA256()
        hasher.update(data: Data(model.standardizedFileURL.path.utf8))
        let fm = FileManager.default
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey, .isRegularFileKey]
        if let walker = fm.enumerator(at: model, includingPropertiesForKeys: keys) {
            var entries: [String] = []
            for case let url as URL in walker {
                guard let values = try? url.resourceValues(forKeys: Set(keys)),
                      values.isRegularFile == true else { continue }
                let size = values.fileSize ?? 0
                let stamp = values.contentModificationDate?.timeIntervalSince1970 ?? 0
                entries.append("\(url.lastPathComponent)|\(size)|\(stamp)")
            }
            // Die Reihenfolge des Enumerators ist nicht zugesichert, deshalb sortiert.
            for entry in entries.sorted() { hasher.update(data: Data(entry.utf8)) }
        }
        let hex = hasher.finalize().prefix(8).map { String(format: "%02x", $0) }.joined()
        return "\(model.deletingPathExtension().lastPathComponent)-\(hex)"
    }
}
