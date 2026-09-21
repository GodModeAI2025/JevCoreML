import Foundation
import XCTest
@testable import JevDecisionKit

/// Der Server war der einzige Teil ohne Test. Zwei Eingaben haben den Prozess beendet, und ein
/// belegter Port fiel gar nicht auf. Diese Tests prüfen die Stellen, an denen das passiert ist.
final class ServerTests: XCTestCase {
    private var port: UInt16 = 0

    private func makeServer() throws -> (JevServer, SystemOne)? {
        guard let model = Fixtures.modelURL() else {
            if Fixtures.requiresModel { XCTFail("JEV_REQUIRE_MODEL=1, aber kein Modell vorhanden") }
            return nil
        }
        let systemOne = try SystemOne(modelURL: model, tokenizerURL: Fixtures.tokenizerURL)
        // Port aus dem Testnamen ableiten, damit parallele Läufe sich nicht ins Gehege kommen.
        port = UInt16(19_000 + abs(name.hashValue % 900))
        return (JevServer(systemOne: systemOne, configuration: .init(port: port)), systemOne)
    }

    private func send(_ raw: String, timeout: TimeInterval = 10) -> String? {
        guard let socket = try? Socket(host: "127.0.0.1", port: port, timeout: timeout) else { return nil }
        defer { socket.close() }
        socket.write(raw)
        return socket.readAll()
    }

    private func healthy() -> Bool {
        guard let response = send("GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n") else { return false }
        return response.contains("\"status\"")
    }

    /// /separate rechnet jede Frage für sich und darf deshalb mehr Fragen annehmen, als der Export
    /// in einem Durchlauf fasst. Vorher prüfte der Server die ganze Anfrage am Stück und lehnte
    /// fünf Fragen an einem Export für vier mit 422 ab; kev.serve beantwortet sie.
    func testSeparateAcceptsMoreQuestionsThanOnePass() throws {
        guard let (server, systemOne) = try makeServer() else { return }
        try XCTSkipUnless(systemOne.runtime.contract == .fanOut, "braucht einen Fan-Out-Export")
        try server.start()
        defer { server.stop() }
        let count = systemOne.runtime.maxQuestions + 1
        let questions = (0 ..< count).map { #""q\#($0)": {"type": "noul", "instructions": "Frage \#($0)?"}"# }
        let body = #"{"state": "Der Kunde wurde doppelt belastet.", "questions": {"# + questions.joined(separator: ", ") + "}}"
        func post(_ path: String) -> String? {
            send("POST \(path) HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                 + "Content-Length: \(body.utf8.count)\r\n\r\n\(body)", timeout: 60)
        }
        let separate = post("/v1/systemone/separate")
        XCTAssertEqual(separate?.hasPrefix("HTTP/1.1 200"), true, separate ?? "keine Antwort")
        XCTAssertEqual(separate?.contains("\"q\(count - 1)\""), true, "letzte Frage fehlt in der Antwort")
        let packed = post("/v1/systemone")
        XCTAssertEqual(packed?.hasPrefix("HTTP/1.1 422"), true, "am Stück passt es nicht in einen Durchlauf")
    }

    /// Ein einziges "Content-Length: -1" beendete den Prozess mit einem Trap.
    func testNegativeContentLengthDoesNotKillTheServer() throws {
        guard let (server, _) = try makeServer() else { return }
        try server.start()
        defer { server.stop() }
        XCTAssertTrue(healthy(), "Server antwortet vor dem Angriff")
        let response = send("POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: -1\r\n\r\n{}")
        XCTAssertEqual(response?.contains("400"), true, "negative Länge muss 400 ergeben, nicht Schweigen")
        XCTAssertTrue(healthy(), "Server muss eine negative Länge überleben")
    }

    /// Eine absurd große Länge darf keinen Puffer wachsen lassen, bis der Speicher ausgeht.
    func testOversizedContentLengthIsRejected() throws {
        guard let (server, _) = try makeServer() else { return }
        try server.start()
        defer { server.stop() }
        let response = send("POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: 99999999999\r\n\r\n{}")
        XCTAssertEqual(response?.contains("413"), true, "übergroße Länge muss 413 ergeben")
        XCTAssertTrue(healthy(), "Server muss eine übergroße Länge überleben")
    }

    /// Tief verschachteltes JSON lief in den Stapelüberlauf.
    func testDeeplyNestedJSONIsRejected() throws {
        guard let (server, _) = try makeServer() else { return }
        try server.start()
        defer { server.stop() }
        let deep = String(repeating: "[", count: 500) + String(repeating: "]", count: 500)
        let body = #"{"state":"# + deep + #","questions":{"q":{"type":"noul","instructions":"x"}}}"#
        let request = "POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: \(body.utf8.count)\r\n\r\n\(body)"
        let response = send(request)
        XCTAssertEqual(response?.contains("422"), true, "tiefe Verschachtelung muss 422 ergeben")
        XCTAssertTrue(healthy(), "Server muss tiefe Verschachtelung überleben")
    }

    /// Ein belegter Port wurde nicht gemeldet: start() kehrte erfolgreich zurück, und die
    /// Anfragen beantwortete der erste Prozess, also womöglich ein anderes Modell.
    func testSecondServerOnTheSamePortFails() throws {
        guard let (first, systemOne) = try makeServer() else { return }
        try first.start()
        defer { first.stop() }
        let second = JevServer(systemOne: systemOne, configuration: .init(port: port))
        XCTAssertThrowsError(try second.start(timeout: 3), "zweiter Server muss scheitern") { _ in }
        second.stop()
    }

    /// Chunking wird nicht dekodiert, aber auch nicht als leerer Körper missverstanden.
    func testChunkedIsRejectedExplicitly() throws {
        guard let (server, _) = try makeServer() else { return }
        try server.start()
        defer { server.stop() }
        let response = send("POST /v1/systemone HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n")
        XCTAssertEqual(response?.contains("411"), true, "chunked muss 411 ergeben, nicht 422 über einen leeren Körper")
        XCTAssertTrue(healthy())
    }

    /// Eine Verbindung, die nur Kopfzeilen schickt und nie eine Leerzeile, darf nicht ewig offen bleiben.
    func testHeaderFloodIsCapped() throws {
        guard let (server, _) = try makeServer() else { return }
        try server.start()
        defer { server.stop() }
        var flood = "POST /v1/systemone HTTP/1.1\r\n"
        for i in 0 ..< 3000 { flood += "X-Pad-\(i): \(String(repeating: "a", count: 40))\r\n" }
        let response = send(flood, timeout: 20)
        XCTAssertEqual(response?.contains("431"), true, "Kopfflut muss 431 ergeben")
        XCTAssertTrue(healthy())
    }

    /// Fehlendes state ist in der Referenz 422, nicht 200 über einen leeren Zustand.
    func testMissingStateIsRejected() throws {
        guard let (server, _) = try makeServer() else { return }
        try server.start()
        defer { server.stop() }
        let body = #"{"model":"m","questions":{"q":{"type":"noul","instructions":"x"}}}"#
        let response = send("POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: \(body.utf8.count)\r\n\r\n\(body)")
        XCTAssertEqual(response?.contains("422"), true, "fehlendes state muss 422 ergeben")
    }
}

/// Minimaler blockierender TCP-Client. Genug, um dem Server rohe Bytes zu schicken.
private final class Socket {
    private let fd: Int32

    init(host: String, port: UInt16, timeout: TimeInterval) throws {
        fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw JevError.modelOutput("socket()") }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = inet_addr(host)
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
            if out.count > 1 << 20 { break }
        }
        return String(decoding: out, as: UTF8.self)
    }

    func close() { Darwin.close(fd) }
}
