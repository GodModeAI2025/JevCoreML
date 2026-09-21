import AppKit
import Foundation
import JevDecisionKit
import SwiftUI

/// Startpunkt. Mit `--selbsttest [Modellordner]` rechnet die App eine Triage ohne Fenster und
/// beendet sich; so lässt sich prüfen, ob Paket und Modelle zusammenpassen.
@main
enum Main {
    static func main() {
        let arguments = CommandLine.arguments
        if let index = arguments.firstIndex(of: "--selbsttest") {
            let path = arguments.indices.contains(index + 1) ? arguments[index + 1] : nil
            Task { await SelfTest.run(modelsPath: path) }
            dispatchMain()
        }
        if let index = arguments.firstIndex(of: "--bildschirmfoto"), arguments.indices.contains(index + 1) {
            // Rendert das Fenster nach einer echten Entscheidung in eine PNG-Datei, ohne es zu
            // zeigen. Für README und Landingpage.
            let output = arguments[index + 1]
            let path = arguments.indices.contains(index + 2) && !arguments[index + 2].hasPrefix("--")
                ? arguments[index + 2] : nil
            NSApplication.shared.setActivationPolicy(.accessory)
            Task { await Screenshot.run(output: output, modelsPath: path) }
            NSApplication.shared.run()
        }
        JevDemoApp.main()
    }
}

struct JevDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 900, minHeight: 620)
        }
        .windowToolbarStyle(.unified)
    }
}

enum SelfTest {
    @MainActor
    static func run(modelsPath: String?) async {
        let directory = modelsPath.map { URL(fileURLWithPath: $0) } ?? DemoModel.defaultModelsDirectory()
        guard let directory else {
            print("Selbsttest: kein Modellordner gefunden")
            exit(1)
        }
        let model = DemoModel()
        model.choose(directory: directory)
        var failed = false
        for engine in [Engine.laya, .kev] {
            model.engine = engine
            model.questionSet = .triage
            await model.run()
            if let error = model.errorMessage {
                print("\(engine.title): Fehler: \(error)")
                failed = true
                continue
            }
            guard let result = model.result else { continue }
            print(String(format: "%@ | %.1f ms, erster Lauf %@ | %@", result.engine, result.milliseconds,
                         result.firstRunMilliseconds.map { String(format: "%.0f ms", $0) } ?? "warm", result.detail))
            for row in result.answers {
                print("  \(row.id): \(row.headline)")
            }
        }
        exit(failed ? 1 : 0)
    }
}

enum Screenshot {
    @MainActor
    static func run(output: String, modelsPath: String?) async {
        let model = DemoModel()
        if let modelsPath { model.choose(directory: URL(fileURLWithPath: modelsPath)) }
        model.questionSet = .triage
        await model.run()
        if let error = model.errorMessage {
            print("Bildschirmfoto: \(error)")
            exit(1)
        }
        let size = NSSize(width: 1180, height: 760)
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size),
                              styleMask: [.titled], backing: .buffered, defer: false)
        // Hell. Die dunkle Darstellung zeichnet macOS außerhalb des Bildschirms nur zur Hälfte.
        window.appearance = NSAppearance(named: .aqua)
        // Außerhalb jedes Bildschirms, aber sichtbar geschaltet: nur dann zeichnet macOS die
        // Seitenleiste mit ihrem durchscheinenden Hintergrund. Zu sehen ist davon nichts.
        window.setFrameOrigin(NSPoint(x: -20_000, y: -20_000))
        let host = NSHostingView(rootView: ContentView(model: model)
            .frame(width: size.width, height: size.height)
            .environment(\.colorScheme, .light))
        host.appearance = window.appearance
        host.frame = NSRect(origin: .zero, size: size)
        window.contentView = host
        window.orderFrontRegardless()
        // SwiftUI zeichnet über die Runloop; kurz laufen lassen, dann abgreifen.
        try? await Task.sleep(for: .milliseconds(1500))
        host.layoutSubtreeIfNeeded()
        guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else { exit(1) }
        host.cacheDisplay(in: host.bounds, to: bitmap)
        guard let png = bitmap.representation(using: .png, properties: [:]) else { exit(1) }
        do {
            try png.write(to: URL(fileURLWithPath: output))
            print("gespeichert: \(output)")
            exit(0)
        } catch {
            print("Bildschirmfoto: \(error)")
            exit(1)
        }
    }
}
