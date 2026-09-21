import AppKit
import SwiftUI

struct ContentView: View {
    @State var model: DemoModel

    init(model: DemoModel = DemoModel()) {
        _model = State(initialValue: model)
    }

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 260, ideal: 290)
        } detail: {
            VStack(spacing: 0) {
                input
                Divider()
                results
            }
        }
        .navigationTitle("Jev Demo")
    }

    // MARK: - Seitenleiste

    private var sidebar: some View {
        Form {
            Section("Modell") {
                Picker("Maschine", selection: $model.engine) {
                    ForEach(Engine.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                Text(model.engine.summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section("Fragensatz") {
                Picker("Fragensatz", selection: $model.questionSet) {
                    ForEach(QuestionSet.allCases) { Text($0.title).tag($0) }
                }
                .labelsHidden()
                Text(model.questionSet.summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if model.questionSet == .custom {
                Section("Eigene Frage") {
                    TextField("Frage", text: $model.customInstruction, axis: .vertical)
                    Text("Eine Option je Zeile, name: Beschreibung")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    TextEditor(text: $model.customOptions)
                        .font(.system(.body, design: .monospaced))
                        .frame(minHeight: 90)
                }
            }
            Section("Modellordner") {
                Text(shortPath(model.modelsDirectory) ?? "keiner gewählt")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .help(model.modelsDirectory?.standardizedFileURL.path ?? "")
                Button("Ordner wählen …", action: chooseFolder)
            }
        }
        .formStyle(.grouped)
    }

    /// Die letzten zwei Ordner, der ganze Pfad steht im Tooltip.
    private func shortPath(_ url: URL?) -> String? {
        guard let url = url?.standardizedFileURL else { return nil }
        return "…/" + url.pathComponents.suffix(2).joined(separator: "/")
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Auswählen"
        panel.message = "Den Ordner mit den .mlpackage-Dateien und Tokenizern wählen"
        if panel.runModal() == .OK, let url = panel.url { model.choose(directory: url) }
    }

    // MARK: - Eingabe

    private var input: some View {
        VStack(alignment: .leading, spacing: 10) {
            if model.questionSet == .email {
                TextField("Betreff", text: $model.subject)
                    .textFieldStyle(.roundedBorder)
            }
            TextEditor(text: $model.text)
                .font(.body)
                .frame(minHeight: 80, maxHeight: 160)
                .padding(6)
                .background(.background, in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.quaternary))
            HStack {
                Menu("Beispiel") {
                    ForEach(model.questionSet.samples, id: \.title) { sample in
                        Button(sample.title) { model.text = sample.text }
                    }
                }
                .fixedSize()
                Text(model.status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if model.isRunning { ProgressView().controlSize(.small) }
                Button {
                    Task { await model.run() }
                } label: {
                    Label("Entscheiden", systemImage: "bolt.fill")
                }
                .keyboardShortcut(.return, modifiers: .command)
                .buttonStyle(.borderedProminent)
                .disabled(model.isRunning || model.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(16)
    }

    // MARK: - Ergebnis

    @ViewBuilder
    private var results: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if let error = model.errorMessage {
                    Label(error, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                }
                if let result = model.result {
                    header(result)
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 260), spacing: 14, alignment: .top)],
                              alignment: .leading, spacing: 14) {
                        ForEach(result.answers) { AnswerCard(row: $0) }
                    }
                } else if model.errorMessage == nil {
                    ContentUnavailableView("Noch keine Entscheidung",
                                           systemImage: "bolt.horizontal",
                                           description: Text("Text eingeben oder ein Beispiel wählen, dann ⌘↩."))
                        .frame(maxWidth: .infinity, minHeight: 240)
                }
            }
            .padding(16)
        }
    }

    private func header(_ result: DemoResult) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text(result.engine).font(.headline)
                Spacer()
                Text(result.milliseconds.formatted(.number.precision(.fractionLength(1))) + " ms")
                    .font(.system(.title3, design: .rounded).monospacedDigit())
                    .bold()
            }
            Text(result.detail)
                .font(.callout)
                .foregroundStyle(.secondary)
            Text("\(result.answers.count) Antworten · \(result.inputTokens) Eingabetoken · lokal, ohne Netz · Median aus drei Läufen")
                .font(.caption)
                .foregroundStyle(.secondary)
            if let first = result.firstRunMilliseconds {
                Text("Der erste Lauf brauchte \(first.formatted(.number.precision(.fractionLength(0)))) ms, mit Laden und Vorbereiten.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

struct AnswerCard: View {
    let row: AnswerRow

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(row.id)
                    .font(.system(.subheadline, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer()
                Text(row.kind)
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.quaternary, in: Capsule())
            }
            Text(row.headline)
                .font(.title3.bold())
                .lineLimit(2)
            HStack(spacing: 12) {
                if let confidence = row.confidence {
                    Text("Konfidenz \(confidence, format: .number.precision(.fractionLength(2)))")
                }
                if let act = row.actProbability {
                    Text("Handlung \(act, format: .number.precision(.fractionLength(2)))")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            VStack(spacing: 4) {
                ForEach(Array(row.bars.sorted { $0.value > $1.value }.prefix(6).enumerated()), id: \.offset) { _, bar in
                    ProbabilityBar(label: bar.label, value: bar.value)
                }
            }
        }
        .padding(12)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
    }
}

struct ProbabilityBar: View {
    let label: String
    let value: Double

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.caption)
                .lineLimit(1)
                .frame(width: 110, alignment: .leading)
            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule().fill(.quaternary)
                    Capsule().fill(.tint)
                        .frame(width: max(2, proxy.size.width * min(max(value, 0), 1)))
                }
            }
            .frame(height: 6)
            Text(value, format: .percent.precision(.fractionLength(0)))
                .font(.caption.monospacedDigit())
                .frame(width: 40, alignment: .trailing)
        }
    }
}
