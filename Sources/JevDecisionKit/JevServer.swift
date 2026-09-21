import Foundation
import Network

/// Ein kleiner HTTP/1.1-Server, der `POST /v1/systemone` so beantwortet wie `kev.serve`
/// und damit wie die TypeSafe-Schnittstelle.
///
/// Zweck ist nicht Betrieb unter Last, sondern dass bestehender Jev-Code ohne Umbau auf
/// localhost zeigen kann. Deshalb bewusst schmal: nur die Methoden und Statuscodes, die
/// `vendor/kev/tests/test_api.py` prüft, Content-Length statt Chunking, Verbindung nach
/// der Antwort zu.
public final class JevServer: @unchecked Sendable {
    public struct Configuration: Sendable {
        public var port: UInt16
        public var host: String
        /// Sekunden, die eine Verbindung ohne vollständige Anfrage offen bleiben darf.
        public var idleTimeout: TimeInterval
        /// Sekunden, die ein Modelldurchlauf dauern darf, bevor die Verbindung fällt.
        public var requestTimeout: TimeInterval

        public init(port: UInt16 = 8008, host: String = "127.0.0.1",
                    idleTimeout: TimeInterval = 15, requestTimeout: TimeInterval = 300) {
            self.port = port
            self.host = host
            self.idleTimeout = idleTimeout
            self.requestTimeout = requestTimeout
        }
    }

    /// Obergrenze für den Körper einer Anfrage. Darüber wird die Verbindung geschlossen,
    /// statt beliebig viel Speicher zu sammeln.
    public static let maximumBodyBytes = 8 << 20
    /// Obergrenze für die gesamte Anfrage samt Kopfzeilen.
    public static let maximumRequestBytes = maximumBodyBytes + (64 << 10)
    /// Obergrenze für den Kopfbereich allein. Ein Strom ohne Leerzeile lief sonst unbegrenzt.
    public static let maximumHeaderBytes = 64 << 10

    private let engine: any SystemOneEngine
    private let configuration: Configuration
    // Ohne ausdrückliche Priorität erben Queue und Tasks die Vorgabe und landen mit ihrem
    // CPU-Anteil auf den Effizienzkernen. Gemessen kosteten zehn laya-Fragen im Server so
    // 121 ms, mit dieser Priorität 89 ms; die Kommandozeile auf dem Hauptthread braucht 83 ms.
    private let queue = DispatchQueue(label: "jev.server", qos: .userInitiated)
    private var listener: NWListener?
    /// Modellkennung, die in der Antwort steht, wenn die Anfrage keine mitschickt.
    public let modelIdentifier: String
    /// Beschreibung in `/v1/models`.
    public let modelDescription: String
    /// Weitere Namen, die im Feld `model` etwas bewirken: bei kev die üblichen Kennungen, bei
    /// laya die Checkpoints, die das Routing festlegen.
    public let modelAliases: [String]
    /// Höchstzahl Optionen je Frage beim Lesen der Anfrage. kev nimmt 255 wie `kev/api.py`,
    /// laya hat keine eigene Grenze und prüft erst an der Sequenz.
    public let optionLimit: Int

    /// Ein einzelner Export oder ein Pool aus mehreren. Der Server merkt keinen Unterschied,
    /// außer in `/api/info`, wo jeder Export mit seinem Budget steht.
    public init(engine: any SystemOneEngine, configuration: Configuration = .init(),
                modelIdentifier: String = "jev-latest",
                modelDescription: String = "Kev auf Core ML, lokal, ohne Netz",
                modelAliases: [String] = ["jev-latest", "kev-latest"],
                optionLimit: Int = JevRequest.maxOptions) {
        self.engine = engine
        self.configuration = configuration
        self.modelIdentifier = modelIdentifier
        self.modelDescription = modelDescription
        self.modelAliases = modelAliases
        self.optionLimit = optionLimit
    }

    public convenience init(systemOne: SystemOne, configuration: Configuration = .init(),
                            modelIdentifier: String = "jev-latest") {
        self.init(engine: systemOne, configuration: configuration, modelIdentifier: modelIdentifier)
    }

    /// Startet den Listener und wartet, bis er bereit ist.
    ///
    /// `NWListener` wirft beim Binden nicht. Ein belegter Port wird erst später über den
    /// `stateUpdateHandler` gemeldet. Ohne dieses Warten kehrte `start()` auch dann erfolgreich
    /// zurück, wenn nie gebunden wurde: der Prozess behauptete zu lauschen, und jede Anfrage
    /// landete beim ersten Server auf demselben Port, also beim falschen Modell.
    public func start(timeout: TimeInterval = 10) throws {
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        parameters.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(configuration.host),
                                                     port: NWEndpoint.Port(rawValue: configuration.port)!)
        let listener = try NWListener(using: parameters)
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }

        let ready = DispatchSemaphore(value: 0)
        let failure = Mutex<NWError?>(nil)
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready:
                ready.signal()
            case let .failed(error):
                failure.set(error)
                ready.signal()
            case .cancelled:
                ready.signal()
            default:
                break
            }
        }
        listener.start(queue: queue)
        self.listener = listener

        if ready.wait(timeout: .now() + timeout) == .timedOut {
            listener.cancel()
            self.listener = nil
            throw JevError.modelOutput("Listener auf \(configuration.host):\(configuration.port) wurde nicht bereit")
        }
        if let error = failure.get() {
            listener.cancel()
            self.listener = nil
            throw JevError.modelOutput("Port \(configuration.port) nicht verfügbar: \(error)")
        }
    }

    /// Winziger Schutz um einen Wert, der aus dem Listener-Callback geschrieben wird.
    private final class Mutex<Value>: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Value
        init(_ value: Value) { self.value = value }
        func set(_ newValue: Value) { lock.lock(); value = newValue; lock.unlock() }
        func get() -> Value { lock.lock(); defer { lock.unlock() }; return value }
    }

    deinit {
        listener?.cancel()
    }

    public func stop() {
        listener?.cancel()
        listener = nil
    }

    // MARK: - Verbindung

    /// Eine Verbindung samt allem, was beim Abbruch aufgeräumt werden muss.
    private final class Session: @unchecked Sendable {
        let connection: NWConnection
        private let lock = NSLock()
        private var work: Task<Void, Never>?
        private var timer: DispatchSourceTimer?
        private(set) var closed = false

        init(_ connection: NWConnection) { self.connection = connection }

        func attach(_ task: Task<Void, Never>) {
            lock.lock(); defer { lock.unlock() }
            if closed { task.cancel() } else { work = task }
        }

        func armIdleTimer(on queue: DispatchQueue, seconds: TimeInterval, onFire: @escaping () -> Void) {
            lock.lock(); defer { lock.unlock() }
            timer?.cancel()
            let t = DispatchSource.makeTimerSource(queue: queue)
            t.schedule(deadline: .now() + seconds)
            t.setEventHandler(handler: onFire)
            t.resume()
            timer = t
        }

        /// Beendet Verbindung, Zeitgeber und einen noch laufenden Modelldurchlauf.
        func close() {
            lock.lock()
            if closed { lock.unlock(); return }
            closed = true
            let task = work
            let t = timer
            work = nil; timer = nil
            lock.unlock()
            t?.cancel()
            task?.cancel()
            connection.cancel()
        }
    }

    private func accept(_ connection: NWConnection) {
        let session = Session(connection)
        // Eine Verbindung, die nichts mehr schickt und nichts mehr will, darf nicht ewig
        // einen Platz belegen. Ohne diesen Zeitgeber hielt ein halber Header sie offen.
        session.armIdleTimer(on: queue, seconds: configuration.idleTimeout) { session.close() }
        connection.stateUpdateHandler = { state in
            switch state {
            case .failed, .cancelled: session.close()
            default: break
            }
        }
        connection.start(queue: queue)
        receive(session, buffer: Data())
    }

    private func receive(_ session: Session, buffer: Data) {
        let connection = session.connection
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) { [weak self] data, _, complete, error in
            guard let self, !session.closed else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            if error != nil { session.close(); return }

            switch Request.parse(buffer) {
            case let .complete(request):
                session.armIdleTimer(on: self.queue, seconds: self.configuration.requestTimeout) {
                    session.close()
                }
                let task = Task(priority: .userInitiated) {
                    let response = await self.handle(request)
                    // Wer aufgelegt hat, bekommt keine Antwort mehr, und der Durchlauf war
                    // ohnehin schon abgebrochen.
                    guard !Task.isCancelled, !session.closed else { session.close(); return }
                    self.send(response, on: session)
                }
                session.attach(task)

            case let .malformed(status, detail):
                // Vorher war "kaputt" nicht von "noch unvollständig" zu unterscheiden, und der
                // Server las weiter, bis der Aufrufer aufgab oder der Puffer den Speicher fraß.
                self.send(Response.error(status, detail), on: session)

            case .incomplete:
                if complete {
                    session.close()
                } else if buffer.count > JevServer.maximumRequestBytes {
                    self.send(Response.error(413, "Anfrage größer als \(JevServer.maximumRequestBytes) Byte"),
                              on: session)
                } else {
                    self.receive(session, buffer: buffer)
                }
            }
        }
    }

    private func send(_ response: Response, on session: Session) {
        var head = "HTTP/1.1 \(response.status) \(Response.reason(response.status))\r\n"
        head += "Content-Type: application/json\r\n"
        head += "Content-Length: \(response.body.count)\r\n"
        head += "Connection: close\r\n\r\n"
        var out = Data(head.utf8)
        out.append(response.body)
        session.connection.send(content: out, completion: .contentProcessed { _ in session.close() })
    }

    // MARK: - Routen

    private func handle(_ request: Request) async -> Response {
        switch (request.method, request.path) {
        case ("POST", "/v1/systemone"):
            return await decide(request, separate: false)
        case ("POST", "/v1/systemone/separate"):
            return await decide(request, separate: true)
        case ("GET", "/v1/models"):
            return Response.json(200, [
                "models": [[
                    "id": modelIdentifier,
                    "name": modelIdentifier,
                    "description": modelDescription,
                    "aliases": modelAliases,
                ]],
            ])
        case ("GET", "/api/info"):
            // Die flachen Felder beschreiben den größten Export, so wie sie vor dem Pool den
            // einzigen beschrieben. Wer mehr wissen will, liest `members`.
            let members = await engine.members()
            guard let largest = members.last else {
                return Response.error(500, "keine Exporte geladen")
            }
            return Response.json(200, [
                "contract": largest.contract.rawValue,
                "sequence_length": largest.sequenceLength,
                "max_questions": largest.maxQuestions,
                "max_options": largest.maxOptions,
                "members": members.map {
                    var entry: [String: Any] = [
                        "file": $0.file, "contract": $0.contract.rawValue,
                        "sequence_length": $0.sequenceLength, "max_questions": $0.maxQuestions,
                        "max_options": $0.maxOptions]
                    // Ein Universalpaket nimmt mehrere Längen an; die Laufzeit wählt die kürzeste.
                    if $0.sequenceLengths.count > 1 { entry["sequence_lengths"] = $0.sequenceLengths }
                    return entry
                },
            ])
        case ("GET", "/healthz"):
            return Response.json(200, ["status": "ok"])
        default:
            return Response.error(404, "kein Handler für \(request.method) \(request.path)")
        }
    }

    private func decide(_ request: Request, separate: Bool) async -> Response {
        let parsed: JevRequest
        do {
            parsed = try JevRequest.parse(json: request.body, maxOptions: optionLimit)
        } catch {
            // Wie FastAPI: was die Anfrage nicht erfüllt, ist 422, nicht 400.
            return Response.error(422, String(describing: error))
        }

        // /separate rechnet jede Frage für sich, also wird auch jede für sich geprüft. Die ganze
        // Anfrage am Stück zu prüfen, lehnte fünf Fragen an einem Export für vier ab, obwohl
        // jede einzelne passt; kev.serve beantwortet sie.
        let requests = separate
            ? parsed.questions.map { question -> JevRequest in
                var single = parsed
                single.questions = [question]
                return single
            }
            : [parsed]

        // Budgets vorab prüfen: eine zu große Anfrage ist 422 mit Grund, kein Modellfehler.
        // Beim Pool heißt das, ob sie in irgendeinen Export passt, und der Grund nennt das
        // größte Budget.
        do {
            for single in requests { try await engine.check(single) }
        } catch let error as JevError {
            return Response.error(422, error.description)
        } catch {
            return Response.error(422, String(describing: error))
        }

        do {
            // Wer aufgelegt hat, soll den Backbone nicht mehr beschäftigen. Ohne diese Prüfung
            // lief der Durchlauf zu Ende und das Ergebnis ging ins Leere.
            try Task.checkCancellation()
            let response: SystemOneResponse
            if separate {
                var answers: [(id: String, answer: JevAnswer)] = []
                var rendered: [String: String] = [:]
                var extras: [(key: String, json: String)] = []
                var model = parsed.model
                var inputTokens = 0, passes = 0
                var milliseconds = 0.0
                for request in requests {
                    // Vor jedem Durchlauf, nicht nur vor dem ersten: sonst liefen bei 20 Fragen
                    // alle 20 weiter, nachdem der Client längst aufgelegt hat.
                    try Task.checkCancellation()
                    let single = try await engine.answer(request)
                    answers.append(contentsOf: single.answers)
                    rendered.merge(single.renderedAnswers) { _, new in new }
                    if extras.isEmpty { extras = single.extraFields; model = single.model }
                    inputTokens += single.usage.inputTokens
                    passes += single.usage.passes
                    milliseconds += single.latencyMilliseconds
                }
                // output_tokens zaehlt die Token der Antwort, und die Antwort gibt es nur einmal.
                // Je Frage aufsummiert zaehlte man die Klammern und Trenner mehrfach.
                let outputTokens = await engine.outputTokens(of: answers)
                response = SystemOneResponse(
                    model: model, answers: answers,
                    usage: JevUsage(inputTokens: inputTokens, outputTokens: outputTokens,
                                    stateTokens: 0, passes: passes),
                    latencyMilliseconds: milliseconds,
                    renderedAnswers: rendered, extraFields: extras)
            } else {
                response = try await engine.answer(parsed)
            }

            // Von Hand geschrieben statt ueber JSONSerialization: dort wird aus einer auf zwei
            // Stellen gerundeten 0.66 die Zeichenkette 0.66000000000000003, weil der Wert als
            // naechstgelegener Double abgelegt ist. Das steht dann so in der Antwort.
            let body = "{"
                + "\"model\": \(KevAnswerJSON.quote(response.model)), "
                + "\"answers\": \(KevAnswerJSON.text(response.answers, pretty: false, rendered: response.renderedAnswers)), "
                + "\"usage\": {\"input_tokens\": \(response.usage.inputTokens), "
                + "\"output_tokens\": \(response.usage.outputTokens)}, "
                + "\"latency_ms\": \(KevAnswerJSON.number(response.latencyMilliseconds)), "
                + "\"passes\": \(response.usage.passes)"
                + response.extraFields.map { ", \(KevAnswerJSON.quote($0.key)): \($0.json)" }.joined()
                + "}"
            return Response(status: 200, body: Data(body.utf8))
        } catch is CancellationError {
            return Response.error(499, "Verbindung abgebrochen")
        } catch let error as JevError {
            return Response.error(422, error.description)
        } catch {
            return Response.error(500, String(describing: error))
        }
    }

    // MARK: - HTTP

    struct Request {
        let method: String
        let path: String
        let body: Data

        /// Was beim Lesen herauskommen kann. Vorher stand ein einzelnes nil für beides,
        /// „noch unvollständig" und „kaputt", und der Server las bei kaputten Anfragen weiter,
        /// bis der Aufrufer aufgab oder der Puffer den Speicher fraß.
        enum Parse {
            case incomplete
            case malformed(Int, String)
            case complete(Request)
        }

        static func parse(_ buffer: Data) -> Parse {
            guard let separator = buffer.range(of: Data("\r\n\r\n".utf8)) else {
                return buffer.count > JevServer.maximumHeaderBytes
                    ? .malformed(431, "Kopfbereich größer als \(JevServer.maximumHeaderBytes) Byte")
                    : .incomplete
            }
            let headerData = buffer[buffer.startIndex ..< separator.lowerBound]
            guard let header = String(data: headerData, encoding: .utf8) else {
                return .malformed(400, "Kopfzeilen sind kein UTF-8")
            }
            let lines = header.components(separatedBy: "\r\n")
            let parts = lines[0].split(separator: " ")
            guard parts.count >= 2 else {
                return .malformed(400, "Anfragezeile unvollständig")
            }
            let method = String(parts[0])
            let path = String(parts[1]).components(separatedBy: "?")[0]

            var length: Int?
            var chunked = false
            for line in lines.dropFirst() {
                let lower = line.lowercased()
                if lower.hasPrefix("content-length:") {
                    let raw = line.dropFirst("content-length:".count).trimmingCharacters(in: .whitespaces)
                    guard let parsed = Int(raw) else {
                        return .malformed(400, "Content-Length ist keine Zahl: \(raw)")
                    }
                    // Eine negative Länge beendete den Prozess: buffer.index(_:offsetBy:) fällt
                    // bei negativem Versatz aus dem Puffer und löst einen Trap aus.
                    guard parsed >= 0 else { return .malformed(400, "Content-Length ist negativ") }
                    guard parsed <= JevServer.maximumBodyBytes else {
                        return .malformed(413, "Körper größer als \(JevServer.maximumBodyBytes) Byte")
                    }
                    if let existing = length, existing != parsed {
                        return .malformed(400, "widersprüchliche Content-Length-Kopfzeilen")
                    }
                    length = parsed
                } else if lower.hasPrefix("transfer-encoding:"),
                          lower.contains("chunked") {
                    chunked = true
                }
            }

            if chunked {
                // Nicht dekodiert, aber auch nicht stillschweigend als leerer Körper behandelt:
                // vorher bekam ein Client mit Chunking ein 422 über eine leere Anfrage.
                return .malformed(411, "Transfer-Encoding: chunked wird nicht unterstützt, bitte Content-Length senden")
            }

            let needed = length ?? 0
            if length == nil, method == "POST" {
                return .malformed(411, "POST ohne Content-Length")
            }
            let bodyStart = separator.upperBound
            let available = buffer.distance(from: bodyStart, to: buffer.endIndex)
            guard available >= needed else { return .incomplete }
            let body = buffer[bodyStart ..< buffer.index(bodyStart, offsetBy: needed)]
            return .complete(Request(method: method, path: path, body: body))
        }
    }

    struct Response {
        let status: Int
        let body: Data

        static func json(_ status: Int, _ object: [String: Any]) -> Response {
            let data = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]))
                ?? Data("{}".utf8)
            return Response(status: status, body: data)
        }

        static func error(_ status: Int, _ detail: String) -> Response {
            json(status, ["detail": detail])
        }

        static func reason(_ status: Int) -> String {
            switch status {
            case 400: return "Bad Request"
            case 411: return "Length Required"
            case 413: return "Payload Too Large"
            case 431: return "Request Header Fields Too Large"
            case 499: return "Client Closed Request"
            case 200: return "OK"
            case 404: return "Not Found"
            case 422: return "Unprocessable Entity"
            case 500: return "Internal Server Error"
            default: return "Status"
            }
        }
    }
}
