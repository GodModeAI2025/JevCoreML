import Foundation
import XCTest
@testable import JevDecisionKit

/// `jev --engine laya --serve`. Die Antwort soll aussehen wie `laya.Router.predict`, Schlüssel für
/// Schlüssel. Der Abgleich gegen laya selbst steht in `Exporter/compare_laya_server.py`; hier wird
/// festgehalten, was dabei herauskam, damit es ohne Python und ohne laufenden Server geprüft bleibt.
final class LayaServerTests: XCTestCase {
    // MARK: - Ohne Modell

    /// `round(x, 4)` und dann `repr`, wie laya es ausgibt. Die Fälle liegen an Rundungsgrenzen,
    /// an denen "mal 10 000, runden, teilen" anders entschiede als Python.
    func testNumbersRoundLikePython() {
        let cases: [(Double, String)] = [
            (0.12345, "0.1235"), (0.00005, "0.0001"), (0.11115, "0.1111"),
            (1.0, "1.0"), (0.99995, "1.0"), (1e-7, "0.0"), (0.5, "0.5"), (0.25, "0.25"),
        ]
        for (value, want) in cases {
            XCTAssertEqual(LayaAnswerJSON.number(value), want, "\(value)")
        }
    }

    func testAnswerFieldsInLayaOrder() {
        let choice = LayaAnswer(
            answer: .choice(ChoiceAnswer(choice: "billing", confidence: 0.9,
                                         probabilities: [("billing", 0.93456), ("tech", 0.06544)])),
            confidence: 0.91234, actProbability: 0.99999, probabilities: [0.93456, 0.06544],
            labels: ["billing", "tech"], legend: [], inputTokens: 40)
        XCTAssertEqual(LayaAnswerJSON.render(choice),
                       #"{"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9346, "tech": 0.0654}, "#
                           + #""confidence": 0.9123, "action": {"act_probability": 1.0}}"#)

        // Die Legende trägt die Stufen, wie sie kamen: Zahl, null, Objekt, Text.
        let levels: [JevValue] = [.int(1), .double(2.5), .null, .object([("label", .string("great"))]), .string("five")]
        let score = LayaAnswer(
            answer: .score(ScoreAnswer(score: 2.00004, legend: ["1", "2.5", "null", "?", "five"],
                                       probabilities: [0.1, 0.2, 0.4, 0.2, 0.1], confidence: 0.3)),
            confidence: 0.3, actProbability: 0.5, probabilities: [0.1, 0.2, 0.4, 0.2, 0.1],
            labels: [], legend: levels, inputTokens: 40)
        XCTAssertEqual(LayaAnswerJSON.render(score),
                       #"{"type": "score", "score": 2.0, "legend": {"0": 1, "1": 2.5, "2": null, "3": {"label": "great"}, "4": "five"}, "#
                           + #""probabilities": {"0": 0.1, "1": 0.2, "2": 0.4, "3": 0.2, "4": 0.1}, "#
                           + #""confidence": 0.3, "action": {"act_probability": 0.5}}"#)

        let noul = LayaAnswer(answer: .noul(NoulAnswer(noul: 0.12694, probabilities: [0.87306, 0.12694])),
                              confidence: 0.87306, actProbability: 1.0, probabilities: [0.87306, 0.12694],
                              labels: [], legend: [], inputTokens: 30)
        XCTAssertEqual(LayaAnswerJSON.render(noul),
                       #"{"type": "noul", "noul": 0.1269, "confidence": 0.8731, "action": {"act_probability": 1.0}}"#)
    }

    /// Dieselben Schlüssel wie `RouteDecision`, die Anteile ungerundet wie in laya.
    func testRoutingRecord() {
        let detection = LayaRouter.analyse(.string("ab αβ вг"))
        let text = LayaAnswerJSON.routing(checkpoint: "multilingual", reason: "x",
                                          detection: detection, workflow: nil)
        XCTAssertEqual(text,
                       #"{"model": "multilingual", "repo": "convaiinnovations/laya/multilingual", "reason": "x", "#
                           + #""detection": {"script": "greek", "script_profile": {"latin": 0.3333333333333333, "#
                           + #""greek": 0.3333333333333333, "cyrillic": 0.3333333333333333}, "language": null, "#
                           + #""is_english": false, "non_latin_fraction": 0.6667}, "workflow": null}"#)
        XCTAssertEqual(LayaAnswerJSON.routing(checkpoint: "english", reason: "y", detection: nil, workflow: "customer_service"),
                       #"{"model": "english", "repo": "convaiinnovations/laya", "reason": "y", "detection": null, "workflow": "customer_service"}"#)
    }

    /// `task` und `lang` kommen aus der Anfrage, ein falscher Typ ist 422 und kein stilles nil.
    func testRoutingHintsAreParsed() throws {
        let body = #"{"state": "x", "task": "typed_decisions", "lang": "de", "questions": {"a": {"type": "noul", "instructions": "?"}}}"#
        let request = try JevRequest.parse(json: Data(body.utf8))
        XCTAssertEqual(request.task, "typed_decisions")
        XCTAssertEqual(request.language, "de")
        let wrong = #"{"state": "x", "lang": 5, "questions": {"a": {"type": "noul", "instructions": "?"}}}"#
        XCTAssertThrowsError(try JevRequest.parse(json: Data(wrong.utf8)))
    }

    // MARK: - Mit Modell

    private var port: UInt16 = 0

    private func makeServer() throws -> (JevServer, LayaRouted)? {
        let english = Fixtures.models.appendingPathComponent("Laya-EN-L512-K512-fp16.mlpackage")
        guard FileManager.default.fileExists(atPath: english.path) else {
            if Fixtures.requiresModel { XCTFail("JEV_REQUIRE_MODEL=1, aber kein laya-Modell vorhanden") }
            return nil
        }
        let routed = try LayaRouted.discovering(in: Fixtures.models)
        port = UInt16(20_000 + abs(name.hashValue % 900))
        let server = JevServer(engine: routed, configuration: .init(port: port),
                               modelIdentifier: "laya-rl-agent", modelDescription: "laya",
                               optionLimit: .max)
        return (server, routed)
    }

    private func post(_ path: String, _ body: String) throws -> (status: String, json: JevValue?) {
        let socket = try LayaSocket(port: port, timeout: 120)
        defer { socket.close() }
        socket.write("POST \(path) HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     + "Content-Length: \(body.utf8.count)\r\n\r\n\(body)")
        let raw = socket.readAll()
        let parts = raw.components(separatedBy: "\r\n\r\n")
        let status = String(parts.first?.prefix(12) ?? "")
        let json = parts.count > 1 ? try? JevValue.parse(json: Data(parts[1...].joined(separator: "\r\n\r\n").utf8)) : nil
        return (status, json)
    }

    private func keys(_ value: JevValue?) -> [String] {
        if case let .object(pairs)? = value { return pairs.map(\.key) }
        return []
    }

    private func field(_ value: JevValue?, _ path: String...) -> JevValue? {
        var current = value
        for key in path {
            guard case let .object(pairs)? = current else { return nil }
            current = pairs.first { $0.key == key }?.value
        }
        return current
    }

    func testAnswerHasLayaShape() throws {
        guard let (server, _) = try makeServer() else { return }
        try server.start()
        defer { server.stop() }
        let body = #"{"state": {"body": "We were billed twice for March, please refund us."}, "questions": {"#
            + #""team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": "bugs"}}, "#
            + #""stars": {"type": "score", "instructions": "How happy?", "criteria": [1, 2.5, null, "five"]}, "#
            + #""churn": {"type": "noul", "instructions": "Does the user threaten to leave?"}}}"#
        let (status, json) = try post("/v1/systemone", body)
        XCTAssertEqual(status, "HTTP/1.1 200")
        XCTAssertEqual(keys(json), ["model", "answers", "usage", "latency_ms", "passes", "routing"])
        XCTAssertEqual(field(json, "model"), .string("laya-rl-agent"))
        XCTAssertEqual(keys(field(json, "usage")), ["input_tokens", "output_tokens"])
        XCTAssertEqual(field(json, "usage", "output_tokens"), .int(0))
        XCTAssertEqual(keys(field(json, "answers")), ["team", "stars", "churn"])
        XCTAssertEqual(keys(field(json, "answers", "team")), ["type", "choice", "probabilities", "confidence", "action"])
        XCTAssertEqual(keys(field(json, "answers", "stars")),
                       ["type", "score", "legend", "probabilities", "confidence", "action"])
        XCTAssertEqual(field(json, "answers", "stars", "legend"),
                       .object([("0", .int(1)), ("1", .double(2.5)), ("2", .null), ("3", .string("five"))]))
        XCTAssertEqual(keys(field(json, "answers", "churn")), ["type", "noul", "confidence", "action"])
        XCTAssertEqual(keys(field(json, "answers", "churn", "action")), ["act_probability"])
        XCTAssertEqual(keys(field(json, "routing")), ["model", "repo", "reason", "detection", "workflow"])
        XCTAssertEqual(field(json, "routing", "model"), .string("english"))
        XCTAssertEqual(field(json, "routing", "detection", "script"), .string("latin"))
    }

    /// Die Hinweise der Anfrage wirken, und ein unbekannter Name ist ein Fehler der Anfrage.
    func testHintsSteerTheRouting() throws {
        guard let (server, routed) = try makeServer() else { return }
        try server.start()
        defer { server.stop() }
        let questions = #""questions": {"a": {"type": "noul", "instructions": "Is this about money?"}}"#
        let (status, json) = try post("/v1/systemone", #"{"state": "We were billed twice.", "lang": "de", "# + questions + "}")
        XCTAssertEqual(status, "HTTP/1.1 200")
        let expected = routed.available.contains(.multilingual) ? "multilingual" : "english"
        XCTAssertEqual(field(json, "routing", "model"), .string(expected))
        XCTAssertEqual(field(json, "routing", "detection"), .null, "ein Hinweis entscheidet ohne Erkennung")

        // kev-latest ist der Vorgabewert des TypeSafe-Clients und darf das Routing nicht festlegen.
        let (_, kev) = try post("/v1/systemone", #"{"state": "We were billed twice.", "model": "kev-latest", "# + questions + "}")
        XCTAssertEqual(field(kev, "routing", "detection", "script"), .string("latin"))

        let (bad, _) = try post("/v1/systemone", #"{"state": "x", "task": "poetry", "# + questions + "}")
        XCTAssertEqual(bad, "HTTP/1.1 422")
    }

    /// So viele Optionen wie laya: 150 nimmt der englische Checkpoint dort an, 200 passen nicht
    /// mehr vor max_len und laya bricht ab. Hier dasselbe, das zweite als 422 mit Grund.
    func testOptionCountLikeLaya() throws {
        guard let (server, _) = try makeServer() else { return }
        try server.start()
        defer { server.stop() }
        func body(_ count: Int) -> String {
            let criteria = (0 ..< count).map { #""o\#($0)": null"# }.joined(separator: ", ")
            return #"{"state": "We were billed twice.", "model": "english", "questions": {"c": {"type": "choice", "instructions": "pick", "criteria": {"#
                + criteria + "}}}}"
        }
        let (fits, json) = try post("/v1/systemone", body(150))
        XCTAssertEqual(fits, "HTTP/1.1 200")
        XCTAssertEqual(keys(field(json, "answers", "c", "probabilities")).count, 150)
        let (overflow, _) = try post("/v1/systemone", body(200))
        XCTAssertEqual(overflow, "HTTP/1.1 422")
    }
}

/// Wie der Client in ServerTests, dort ist er privat.
private final class LayaSocket {
    private let fd: Int32

    init(port: UInt16, timeout: TimeInterval) throws {
        fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw JevError.modelOutput("socket()") }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        var tv = timeval(tv_sec: Int(timeout), tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        let ok = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard ok == 0 else { Darwin.close(fd); throw JevError.modelOutput("connect()") }
    }

    func write(_ text: String) {
        let bytes = Array(text.utf8)
        _ = bytes.withUnsafeBytes { Darwin.send(fd, $0.baseAddress, $0.count, 0) }
    }

    func readAll() -> String {
        var out = Data()
        var chunk = [UInt8](repeating: 0, count: 8192)
        while true {
            let n = Darwin.recv(fd, &chunk, chunk.count, 0)
            if n <= 0 { break }
            out.append(contentsOf: chunk[0 ..< n])
        }
        return String(decoding: out, as: UTF8.self)
    }

    func close() { Darwin.close(fd) }
}
