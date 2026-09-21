import Foundation
@testable import JevDecisionKit

// Nur zum Übersetzen: das laya-Beispiel aus der README, unverändert. Läuft nie.
private func readmeLayaExample(modelsDirectory: URL, ticket: JevValue,
                               teams: [(name: String, description: JevValue)]) async throws {
    let laya = try LayaRouted.discovering(in: modelsDirectory)
    let result = try await laya.answer(state: ticket, questions: [
        ("team", .choice(instructions: "An welches Team geht das Ticket?", criteria: teams)),
    ])
    let triage = try await laya.answer(state: .object([("message", ticket)]),
                                       questions: LayaPresets.triage())
    print(result.checkpoint, result.reason)
    let answer = result.answers[0].answer
    print(answer.confidence, answer.actProbability, triage.checkpoint)
}

// Die Beispiele aus der README des Repos, unverändert. Auch sie laufen nie, sie müssen nur
// übersetzen, damit die README nicht still veraltet.
private func readmeRepoExamples(ticketText: String, modelsURL: URL, ticket: JevValue,
                                questions: [(id: String, question: JevQuestion)]) async throws {
    let laya = try LayaRouted.bundled()          // oder .discovering(in: ordnerMitModellen)

    let result = try await laya.answer(
        state: .object([("message", .string(ticketText))]),
        questions: [
            ("team", .choice(instructions: "An welches Team geht das Ticket?", criteria: [
                (name: "billing", description: "Rechnungen, Zahlungen, Erstattungen"),
                (name: "tech", description: "Fehler, Abstürze, Ausfälle"),
            ])),
            ("dringend", .noul(instructions: "Ist das dringend?")),
            ("frust", .score(instructions: "Wie verärgert klingt der Kunde?",
                             levels: ["ruhig", "besorgt", "verärgert", "wütend"])),
        ])

    print(result.checkpoint, result.reason)     // multilingual, die Sprache sieht nach de aus ...
    for (id, answer) in result.answers {
        print(id, answer.answer, answer.confidence, answer.actProbability)
    }

    let kev = try SystemOne(modelURL: modelsURL.appending(path: "JevCoreML.mlpackage"),
                            tokenizerURL: modelsURL.appending(path: "tokenizer.json"))
    let response = try await kev.answer(JevRequest(state: ticket, questions: questions))
    print(response.answers.count)
}
