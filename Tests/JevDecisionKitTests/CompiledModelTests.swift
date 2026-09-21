import Foundation
import XCTest
@testable import JevDecisionKit

/// Der Compile-Cache räumt ältere Fassungen desselben Modells weg und sonst nichts.
///
/// Ohne das Aufräumen lagen nach zwei Tagen 44 Einträge mit 51 GB im Cache, und die Platte lief
/// voll. Mit einem zu großzügigen Aufräumen verschwände dagegen `Kev06B-fp16-palette8`, sobald
/// `Kev06B-fp16` neu kompiliert wird, denn der Name beginnt genauso.
final class CompiledModelTests: XCTestCase {
    func testEvictsOnlyOlderVersionsOfTheSameModel() throws {
        let cache = FileManager.default.temporaryDirectory
            .appendingPathComponent("jev-cache-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: cache) }

        let names = [
            "Kev06B-fp16-0000000000000001.mlmodelc",          // alt, soll weg
            "Kev06B-fp16-0000000000000002.mlmodelc",          // alt, soll weg
            "Kev06B-fp16-00000000000000ff.mlmodelc",          // der neue, bleibt
            "Kev06B-fp16-palette8-0000000000000003.mlmodelc", // anderes Modell, bleibt
            "Kev06B-fp16-lut4g16-0000000000000004.mlmodelc",  // anderes Modell, bleibt
            "Kev06B-fp16-staging.mlmodelc",                   // kein Hexschlüssel, bleibt
            "Kev06B-Q4-fp16-0000000000000005.mlmodelc",       // anderes Modell, bleibt
        ]
        for name in names {
            try FileManager.default.createDirectory(at: cache.appendingPathComponent(name),
                                                    withIntermediateDirectories: true)
        }
        let target = cache.appendingPathComponent("Kev06B-fp16-00000000000000ff.mlmodelc")
        CompiledModel.evictOlderEntries(of: URL(fileURLWithPath: "/x/Kev06B-fp16.mlpackage"),
                                        keeping: target, in: cache)

        let left = try FileManager.default.contentsOfDirectory(atPath: cache.path).sorted()
        XCTAssertEqual(left, [
            "Kev06B-Q4-fp16-0000000000000005.mlmodelc",
            "Kev06B-fp16-00000000000000ff.mlmodelc",
            "Kev06B-fp16-lut4g16-0000000000000004.mlmodelc",
            "Kev06B-fp16-palette8-0000000000000003.mlmodelc",
            "Kev06B-fp16-staging.mlmodelc",
        ])
    }
}
