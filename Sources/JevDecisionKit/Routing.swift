import Foundation

/// Schwellen, ab denen eine Entscheidung ohne Rückfrage benutzt werden darf.
///
/// Folgt dem, was TypeSafe unter „Confidence-gated routing" beschreibt: ein Boden, unter dem nie
/// automatisch gehandelt wird, und darüber je Ausgang eine eigene Schwelle, die sich am Risiko
/// bemisst. Für Noul gilt dasselbe zweiseitig, weil eine Noul-Antwort keine Konfidenz mitbringt:
/// ab `noulYes` ist es ein Ja, bis `noulNo` ein Nein, dazwischen entscheidet jemand anderes.
public struct RoutingPolicy: Sendable {
    /// Untere Grenze für Choice und Score. Darunter wird nie automatisch gehandelt.
    public var confidenceFloor: Double
    /// Höhere Schwelle für einzelne Ausgänge, etwa für Aktionen, deren Fehler teuer ist.
    public var perOutcome: [String: Double]
    public var noulYes: Double
    public var noulNo: Double
    /// Verlangt, dass eine Choice-Frage eine Auffangoption anbietet.
    public var requireNoneOption: Bool
    public var noneOptionNames: Set<String>

    public init(confidenceFloor: Double = 0.5,
                perOutcome: [String: Double] = [:],
                noulYes: Double = 0.7,
                noulNo: Double = 0.3,
                requireNoneOption: Bool = false,
                noneOptionNames: Set<String> = ["none", "other", "sonstiges", "unbekannt", "unknown"]) {
        self.confidenceFloor = confidenceFloor
        self.perOutcome = perOutcome
        self.noulYes = noulYes
        self.noulNo = noulNo
        self.requireNoneOption = requireNoneOption
        self.noneOptionNames = noneOptionNames
    }

    /// Konservativ: handelt nur bei sehr klarer Verteilung.
    public static let cautious = RoutingPolicy(confidenceFloor: 0.8, noulYes: 0.85, noulNo: 0.15)
}

public enum EscalationReason: Sendable, Equatable {
    case belowConfidenceFloor(confidence: Double, floor: Double)
    case belowOutcomeThreshold(outcome: String, confidence: Double, threshold: Double)
    /// Der Noul-Wert liegt zwischen den beiden Schwellen: weder Ja noch Nein.
    case noulUndecided(value: Double, no: Double, yes: Double)
    case missingNoneOption

    public var description: String {
        switch self {
        case let .belowConfidenceFloor(c, f):
            return String(format: "Konfidenz %.2f unter dem Boden %.2f", c, f)
        case let .belowOutcomeThreshold(o, c, t):
            return String(format: "Konfidenz %.2f unter der Schwelle %.2f für %@", c, t, o)
        case let .noulUndecided(v, no, yes):
            return String(format: "Noul %.2f liegt zwischen %.2f und %.2f", v, no, yes)
        case .missingNoneOption:
            return "die Frage bietet keine Auffangoption"
        }
    }
}

/// Ergebnis einer gefilterten Entscheidung.
public enum Routed<Value: Sendable>: Sendable {
    /// Die Schwellen sind erfüllt, der Wert darf benutzt werden.
    case act(Value)
    /// Der Wert liegt vor, aber die Schwellen sind nicht erfüllt.
    case escalate(Value, reason: EscalationReason)

    public var value: Value {
        switch self {
        case let .act(v), let .escalate(v, _): return v
        }
    }

    public var isEscalated: Bool {
        if case .escalate = self { return true }
        return false
    }

    public var reason: EscalationReason? {
        if case let .escalate(_, r) = self { return r }
        return nil
    }

    /// Löst eine Eskalation auf, etwa durch ein größeres Modell oder eine Warteschlange.
    public func resolve(
        _ escalate: (Value, EscalationReason) async throws -> Value
    ) async rethrows -> Value {
        switch self {
        case let .act(v): return v
        case let .escalate(v, reason): return try await escalate(v, reason)
        }
    }
}

extension RoutingPolicy {
    func route(_ answer: ChoiceAnswer) -> Routed<ChoiceAnswer> {
        if requireNoneOption,
           !answer.probabilities.contains(where: { candidate in
               noneOptionNames.contains { $0.lowercased() == candidate.name.lowercased() }
           }) {
            return .escalate(answer, reason: .missingNoneOption)
        }
        if answer.confidence < confidenceFloor {
            return .escalate(answer, reason: .belowConfidenceFloor(confidence: answer.confidence,
                                                                   floor: confidenceFloor))
        }
        if let threshold = perOutcome[answer.choice], answer.confidence < threshold {
            return .escalate(answer, reason: .belowOutcomeThreshold(outcome: answer.choice,
                                                                    confidence: answer.confidence,
                                                                    threshold: threshold))
        }
        return .act(answer)
    }

    func route(_ answer: ScoreAnswer) -> Routed<ScoreAnswer> {
        answer.confidence < confidenceFloor
            ? .escalate(answer, reason: .belowConfidenceFloor(confidence: answer.confidence,
                                                              floor: confidenceFloor))
            : .act(answer)
    }

    func route(_ answer: NoulAnswer) -> Routed<NoulAnswer> {
        (answer.noul >= noulYes || answer.noul <= noulNo)
            ? .act(answer)
            : .escalate(answer, reason: .noulUndecided(value: answer.noul, no: noulNo, yes: noulYes))
    }
}

/// Legt eine Eskalationsregel über SystemOne und zählt mit, wie oft sie greift.
///
/// Der Fallback-Anteil ist die Zahl, an der sich eine Kaskade rechnet oder eben nicht: sie sagt,
/// welcher Teil der Fälle das teurere zweite System erreicht.
public actor DecisionRouter {
    public struct Statistics: Sendable, Equatable {
        public var decisions: Int = 0
        public var escalations: Int = 0
        public var fallbackRate: Double { decisions == 0 ? 0 : Double(escalations) / Double(decisions) }
    }

    public let systemOne: SystemOne
    public var policy: RoutingPolicy
    public private(set) var statistics = Statistics()

    public init(systemOne: SystemOne, policy: RoutingPolicy = .init()) {
        self.systemOne = systemOne
        self.policy = policy
    }

    public func setPolicy(_ policy: RoutingPolicy) { self.policy = policy }
    public func resetStatistics() { statistics = Statistics() }

    public func choice(state: JevValue, instruction: JevValue,
                       options: [(name: String, description: JevValue)]) async throws -> Routed<ChoiceAnswer> {
        record(policy.route(try await systemOne.choice(state: state, instruction: instruction, options: options)))
    }

    public func choice(state: JevValue, instruction: JevValue,
                       options: [String]) async throws -> Routed<ChoiceAnswer> {
        try await choice(state: state, instruction: instruction,
                         options: options.map { (name: $0, description: JevValue.null) })
    }

    public func noul(state: JevValue, instruction: JevValue,
                     whenTrue: JevValue = .null, whenFalse: JevValue = .null) async throws -> Routed<NoulAnswer> {
        record(policy.route(try await systemOne.noul(state: state, instruction: instruction,
                                                     whenTrue: whenTrue, whenFalse: whenFalse)))
    }

    public func score(state: JevValue, instruction: JevValue,
                      levels: [JevValue]) async throws -> Routed<ScoreAnswer> {
        record(policy.route(try await systemOne.score(state: state, instruction: instruction, levels: levels)))
    }

    /// Eine vollständige Anfrage, jede Antwort einzeln durch die Regel geschickt.
    /// Unter Phase 2 kostet das trotzdem nur einen Modelldurchlauf.
    public func answer(_ request: JevRequest) async throws -> [(id: String, routed: Routed<JevAnswer>)] {
        let response = try await systemOne.answer(request)
        return response.answers.map { entry in
            let routed: Routed<JevAnswer>
            switch entry.answer {
            case let .choice(a):
                routed = map(policy.route(a), JevAnswer.choice)
            case let .score(a):
                routed = map(policy.route(a), JevAnswer.score)
            case let .noul(a):
                routed = map(policy.route(a), JevAnswer.noul)
            }
            statistics.decisions += 1
            if routed.isEscalated { statistics.escalations += 1 }
            return (id: entry.id, routed: routed)
        }
    }

    private func map<T>(_ routed: Routed<T>, _ wrap: (T) -> JevAnswer) -> Routed<JevAnswer> {
        switch routed {
        case let .act(v): return .act(wrap(v))
        case let .escalate(v, reason): return .escalate(wrap(v), reason: reason)
        }
    }

    private func record<T>(_ routed: Routed<T>) -> Routed<T> {
        statistics.decisions += 1
        if routed.isEscalated { statistics.escalations += 1 }
        return routed
    }
}
