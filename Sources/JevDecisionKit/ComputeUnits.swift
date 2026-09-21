import CoreML

/// Dünne Hülle um `MLComputeUnits`, damit aufrufender Code CoreML nicht selbst importieren muss.
public enum MLComputeUnitsSelection: String, Sendable, CaseIterable {
    case cpuOnly, cpuAndGPU, cpuAndNeuralEngine, all

    public var value: MLComputeUnits {
        switch self {
        case .cpuOnly: return .cpuOnly
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        case .all: return .all
        }
    }
}
