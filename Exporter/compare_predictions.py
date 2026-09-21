#!/usr/bin/env python3
"""Vergleicht zwei Vorhersage-Dumps Frage für Frage.

Aggregate können sich gegenseitig aufhebende Fehler verstecken: zwei Server mit 0,14
Prozentpunkten Abstand können in zwei Fragen abweichen oder in vierzig. Diese Datei zählt nach.

Zur Auflösung: die Wahrscheinlichkeiten stammen aus der HTTP-Antwort und sind dort wie in
kev/api.py auf zwei Nachkommastellen gerundet. Die kleinste messbare Abweichung ist damit 0,01.
`max_abs_delta` und `mean_abs_delta` sind untere Schranken, und `probabilities_over_1pct_apart`
zählt genau die Fälle, in denen der Unterschied diese Auflösung überhaupt erreicht. Für die
Frage, ob eine Entscheidung kippt, genügt das: dafür zählt `disagreements`, und der wird auf
den Auswahlwerten gebildet, nicht auf den gerundeten Zahlen.
"""
import argparse
import json


def load(path):
    return [json.loads(line) for line in open(path)]


def same_answer(row):
    """Vergleicht Vorhersage und Label typunabhaengig."""
    predicted, label = row["predicted"], row["label"]
    if label is None:
        return False
    if isinstance(predicted, bool) or isinstance(label, bool):
        want = label if isinstance(label, bool) else str(label).lower() in ("true", "yes", "1")
        return bool(predicted) == want
    return str(predicted) == str(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-a", default="nativ")
    ap.add_argument("--label-b", default="referenz")
    args = ap.parse_args()

    rows_a, rows_b = load(args.a), load(args.b)
    if len(rows_a) != len(rows_b):
        raise SystemExit(f"unterschiedlich viele Fragen: {len(rows_a)} gegen {len(rows_b)}")

    disagreements, deltas, both_right, a_only, b_only, neither = [], [], 0, 0, 0, 0
    over_1pct = 0
    for x, y in zip(rows_a, rows_b):
        if (x["index"], x["qid"]) != (y["index"], y["qid"]):
            raise SystemExit("Reihenfolge stimmt nicht überein")
        delta = max(abs(x["probabilities"][k] - y["probabilities"].get(k, 0.0))
                    for k in x["probabilities"]) if x["probabilities"] else 0.0
        deltas.append(delta)
        if delta > 0.01:
            over_1pct += 1
        if x["predicted"] != y["predicted"]:
            disagreements.append({"index": x["index"], "qid": x["qid"], "source": x["source"],
                                  "type": x["type"], args.label_a: x["predicted"],
                                  args.label_b: y["predicted"], "label": x["label"],
                                  "max_delta": delta})
        # Score-Vorhersagen kommen als Zeichenkette, das Label als Zahl. Ohne Angleichung
        # gilt jede Score-Frage als falsch, und both_right ist systematisch zu niedrig.
        ok_a, ok_b = same_answer(x), same_answer(y)
        both_right += ok_a and ok_b
        a_only += ok_a and not ok_b
        b_only += ok_b and not ok_a
        neither += not ok_a and not ok_b

    n = len(rows_a)
    summary = {
        "questions": n,
        "disagreements": len(disagreements),
        "disagreement_rate": len(disagreements) / n,
        "probabilities_over_1pct_apart": over_1pct,
        "max_abs_delta": max(deltas),
        "mean_abs_delta": sum(deltas) / n,
        "aufloesung": 0.01,
        "hinweis": "Wahrscheinlichkeiten sind serverseitig auf zwei Stellen gerundet; "
                   "Deltas sind untere Schranken",
        f"both_right": both_right,
        f"only_{args.label_a}_right": a_only,
        f"only_{args.label_b}_right": b_only,
        "neither_right": neither,
        "examples": disagreements[:20],
    }
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"{n} Fragen verglichen")
    print(f"  abweichende Antwort: {len(disagreements)} ({len(disagreements)/n*100:.2f} %)")
    print(f"  davon nur {args.label_a} richtig: {a_only}, nur {args.label_b} richtig: {b_only}")
    print(f"  Wahrscheinlichkeiten mehr als 0,01 auseinander: {over_1pct} ({over_1pct/n*100:.2f} %)")
    print(f"  max |dp| = {max(deltas):.4f}, im Mittel {sum(deltas)/n:.5f}")
    for d in disagreements[:5]:
        print(f"    {d['source']}/{d['qid']}: {d[args.label_a]} gegen {d[args.label_b]} "
              f"(richtig: {d['label']}, |dp| {d['max_delta']:.3f})")


if __name__ == "__main__":
    main()
