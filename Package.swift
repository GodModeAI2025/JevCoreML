// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "JevDecisionKit",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "JevDecisionKit", targets: ["JevDecisionKit"]),
        .executable(name: "jev", targets: ["jev"]),
    ],
    targets: [
        .target(name: "JevDecisionKit"),
        .executableTarget(name: "jev", dependencies: ["JevDecisionKit"]),
        .testTarget(name: "JevDecisionKitTests", dependencies: ["JevDecisionKit"]),
    ]
)
