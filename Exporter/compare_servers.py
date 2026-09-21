#!/usr/bin/env python3
"""Stellt dieselben Anfragen an zwei /v1/systemone-Server und vergleicht die Antworten.

Gedacht für den Vergleich der nativen Swift-Variante mit `kev.serve`, also mit der
PyTorch-Referenzimplementierung über dieselbe Schnittstelle. Verglichen werden die
Wahrscheinlichkeiten, nicht nur die gewählte Option: eine Kaskade hängt an den Zahlen.
"""
import argparse
import json
import statistics
import time

import httpx

DEPARTMENT = {"returns": "Exchanges, refunds, wrong or damaged items",
              "shipping": "Delivery status, delays, lost packages",
              "billing": "Charges, invoices, payment problems"}

REQUESTS = [
    {"state": "My running shoes arrived in the wrong size. Can I swap them for a size 10?",
     "model": "jev-latest",
     "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
                                  "criteria": DEPARTMENT}}},
    {"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
     "model": "jev-latest",
     "questions": {
         "department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": DEPARTMENT},
         "tone": {"type": "choice", "instructions": "What is the customer's tone?",
                  "criteria": {"calm": None, "frustrated": None, "angry": None}},
         "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                     "criteria": ["can wait", "this week", "today"]}}},
    {"state": {"document": "I was charged twice. Please fix this ASAP."}, "model": "jev-latest",
     "questions": {"billing": {"type": "noul", "instructions": "Is this ticket about billing?",
                               "criteria": {"true": "Explicitly about charges", "false": "Not about charges"}},
                   "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                               "criteria": ["can wait", "this week", "today"]}}},
    {"state": "Der Kunde meldet, dass die App beim Start abstuerzt. Er nutzt ein iPhone 15 mit iOS 18.",
     "model": "jev-latest",
     "questions": {"is_bug": {"type": "noul", "instructions": "Beschreibt der Zustand einen technischen Fehler?"},
                   "team": {"type": "choice", "instructions": "An welches Team geht das Ticket?",
                            "criteria": {"billing": None, "logistics": None, "technical": None, "none": None}}}},
    {"state": "Erstelle mir bitte eine Tabelle mit den Umsaetzen der letzten zwoelf Monate.",
     "model": "jev-latest",
     "questions": {"skill": {"type": "choice", "instructions": "Welcher Skill soll diese Aufgabe uebernehmen?",
                             "criteria": {"research": "Quellen recherchieren und zusammenfassen",
                                          "presentation": "Folien und Praesentationen bauen",
                                          "excel": "Tabellen, Kennzahlen und Berechnungen",
                                          "coding": "Software schreiben oder aendern",
                                          "none": "keiner dieser Skills passt"}}}},
    {"state": {"kunde": {"plan": "Enterprise", "seit": "2021"},
               "ticket": "Ihr Produkt ist voellig unbrauchbar, ich will mein Geld zurueck.",
               "historie": ["zwei identische Belastungen", "Rueckerstattung abgelehnt"]},
     "model": "jev-latest",
     "questions": {"refund": {"type": "noul", "instructions": "Ist eine Rueckerstattung angebracht?"},
                   "eskalation": {"type": "score", "instructions": "Wie stark ist die Eskalation?",
                                  "criteria": ["neutral", "leicht gereizt", "veraergert",
                                               "stark veraergert", "eskaliert"]}}},
]


def decision_of(answer):
    """Die Entscheidung, die ein Aufrufer treffen wuerde, je Fragetyp."""
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "noul":
        return answer["noul"] >= 0.5
    return max(answer["probabilities"], key=lambda k: answer["probabilities"][k])


def distribution(answer):
    if answer["type"] == "noul":
        return {"yes": answer["noul"]}
    return dict(answer["probabilities"])


def call(client, base, body, repeat):
    times = []
    data = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        response = client.post(f"{base}/v1/systemone", json=body, timeout=300)
        times.append((time.perf_counter() - t0) * 1000)
        response.raise_for_status()
        data = response.json()
    return data, statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", default="http://127.0.0.1:8008")
    ap.add_argument("--reference", default="http://127.0.0.1:8009")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default="../Benchmarks/server-vergleich.json")
    args = ap.parse_args()

    rows = []
    worst = 0.0
    flips = 0
    with httpx.Client() as client:
        # Ein Aufruf zum Warmlaufen, damit die Zeitmessung nicht die Kernel-Kompilierung enthält.
        for base in (args.native, args.reference):
            call(client, base, REQUESTS[0], 1)
        for body in REQUESTS:
            native, t_native = call(client, args.native, body, args.repeat)
            reference, t_reference = call(client, args.reference, body, args.repeat)
            for qid in reference["answers"]:
                a, b = distribution(native["answers"][qid]), distribution(reference["answers"][qid])
                delta = max(abs(a[k] - b[k]) for k in b)
                worst = max(worst, delta)
                # Ein Entscheidungswechsel gibt es in allen drei Typen, nicht nur bei Choice:
                # bei Noul der Seitenwechsel um 0,5, bei Score die Stufe mit der groessten Masse.
                same = decision_of(native["answers"][qid]) == decision_of(reference["answers"][qid])
                if not same:
                    flips += 1
                rows.append({"question": qid, "type": reference["answers"][qid]["type"],
                             "max_abs_delta": delta, "same_choice": same})
            rows[-1]["latency_native_ms"] = round(t_native, 1)
            rows[-1]["latency_reference_ms"] = round(t_reference, 1)
            print(f"{len(body['questions'])} Frage(n): max|dp|={max(r['max_abs_delta'] for r in rows[-len(body['questions']):]):.3e}  "
                  f"nativ {t_native:7.1f} ms  Referenz {t_reference:7.1f} ms")

    summary = {"requests": len(REQUESTS), "questions": len(rows),
               "max_abs_delta": worst, "choice_flips": flips, "rows": rows}
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"\n{len(rows)} Fragen, max|dp| = {worst:.3e}, abweichende Auswahl: {flips}")


if __name__ == "__main__":
    main()
