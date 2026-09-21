#!/usr/bin/env python3
"""Wertet eine kev-Eval-Suite gegen einen /v1/systemone-Server aus.

Gleicher Code für den nativen Swift-Server und für `kev.serve`, damit der Vergleich
apples-to-apples ist. Berechnet wird, was der Factsheet-Pilotschritt 4 verlangt:
Trefferquote, Kalibrierung und die Abdeckungskurve über der Konfidenzschwelle.
"""
import argparse
import json
import statistics
import time

import httpx


def ece(confidences, correct, bins=10):
    """Expected Calibration Error, wie in kev.evaluate."""
    if not confidences:
        return 0.0
    edges = [i / bins for i in range(bins + 1)]
    total = 0.0
    n = len(confidences)
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i, c in enumerate(confidences)
               if (lo <= c < hi) or (hi == 1.0 and c == 1.0)]
        if not idx:
            continue
        share = len(idx) / n
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        total += share * abs(acc - conf)
    return total


def evaluate_answer(question, answer):
    """-> (richtig, Konfidenz) oder None, wenn die Frage kein Label hat."""
    label = question.get("label")
    if label is None:
        return None
    if answer["type"] == "choice":
        return answer["choice"] == label, answer["confidence"]
    if answer["type"] == "noul":
        want = label if isinstance(label, bool) else str(label).lower() in ("true", "yes", "1")
        value = answer["noul"]
        # Ohne Konfidenzfeld ist der Abstand zur Mitte das, was die Sicherheit beschreibt.
        return (value >= 0.5) == want, abs(value - 0.5) * 2
    if answer["type"] == "score":
        want = int(label)
        got = max(answer["probabilities"], key=lambda k: answer["probabilities"][k])
        return int(got) == want, answer["confidence"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8008")
    ap.add_argument("--suite", default="../vendor/kev/evals/v7/decision-v7/development.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="nativ")
    ap.add_argument("--dump-predictions", default="",
                    help="JSONL je Frage: Aggregate koennen sich gegenseitig aufhebende Fehler verstecken")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.suite)]
    if args.limit:
        rows = rows[: args.limit]

    correct, confidence, sources, latencies = [], [], [], []
    predictions = []
    skipped, errors = 0, []
    per_source = {}

    with httpx.Client(timeout=600) as client:
        # Warmlauf, damit die erste Anfrage nicht die Kernel-Kompilierung in die Statistik trägt.
        client.post(f"{args.base}/v1/systemone",
                    json={"state": "warm", "questions": {"q": {"type": "noul", "instructions": "warm"}}})
        for row in rows:
            body = {"state": row["state"], "model": "jev-latest",
                    "questions": {k: {kk: vv for kk, vv in v.items() if kk in ("type", "instructions", "criteria")}
                                  for k, v in row["questions"].items()}}
            t0 = time.perf_counter()
            response = client.post(f"{args.base}/v1/systemone", json=body)
            latency = (time.perf_counter() - t0) * 1000
            if response.status_code != 200:
                skipped += 1
                if len(errors) < 5:
                    errors.append(f"{response.status_code}: {response.text[:160]}")
                continue
            latencies.append(latency)
            answers = response.json()["answers"]
            source = row.get("_meta", {}).get("source", "?")
            for qid, question in row["questions"].items():
                answer = answers[qid]
                if answer["type"] == "choice":
                    predicted, dist = answer["choice"], answer["probabilities"]
                elif answer["type"] == "noul":
                    predicted, dist = answer["noul"] >= 0.5, {"yes": answer["noul"]}
                else:
                    predicted = max(answer["probabilities"], key=lambda k: answer["probabilities"][k])
                    dist = answer["probabilities"]
                predictions.append({"row": row.get("_meta", {}).get("row"),
                                    "id": row.get("_meta", {}).get("id"),
                                    "index": len(predictions), "qid": qid, "source": source,
                                    "type": answer["type"], "predicted": predicted,
                                    "label": question.get("label"), "probabilities": dist})
                graded = evaluate_answer(question, answers[qid])
                if graded is None:
                    continue
                ok, conf = graded
                correct.append(1.0 if ok else 0.0)
                confidence.append(conf)
                sources.append(source)
                bucket = per_source.setdefault(source, {"n": 0, "ok": 0})
                bucket["n"] += 1
                bucket["ok"] += 1 if ok else 0

    n = len(correct)
    curve = []
    for threshold in [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        idx = [i for i, c in enumerate(confidence) if c >= threshold]
        curve.append({
            "threshold": threshold,
            "coverage": len(idx) / n if n else 0.0,
            "accuracy": (sum(correct[i] for i in idx) / len(idx)) if idx else 0.0,
            "fallback_rate": 1 - (len(idx) / n if n else 0.0),
        })

    summary = {
        "label": args.label,
        "base": args.base,
        "suite": args.suite,
        "records": len(rows),
        "records_answered": len(rows) - skipped,
        "records_rejected": skipped,
        "questions_scored": n,
        "accuracy": sum(correct) / n if n else 0.0,
        "ece": ece(confidence, correct),
        "mean_confidence": statistics.fmean(confidence) if confidence else 0.0,
        "latency_ms": {
            "median": statistics.median(latencies) if latencies else 0.0,
            "p95": sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0.0,
            "mean": statistics.fmean(latencies) if latencies else 0.0,
        },
        "coverage_curve": curve,
        "per_source": {k: {"n": v["n"], "accuracy": v["ok"] / v["n"]} for k, v in sorted(per_source.items())},
        "errors": errors,
    }
    if args.dump_predictions:
        with open(args.dump_predictions, "w") as fh:
            for row in predictions:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"{args.label}: {summary['records_answered']}/{summary['records']} Records beantwortet, "
          f"{n} Fragen bewertet")
    print(f"  Trefferquote {summary['accuracy']*100:.1f} %   ECE {summary['ece']:.3f}   "
          f"Median {summary['latency_ms']['median']:.0f} ms   p95 {summary['latency_ms']['p95']:.0f} ms")
    for point in curve:
        if point["threshold"] in (0.0, 0.8, 0.9):
            print(f"  Schwelle {point['threshold']:.2f}: Abdeckung {point['coverage']*100:5.1f} %  "
                  f"Trefferquote im akzeptierten Teil {point['accuracy']*100:5.1f} %")
    if errors:
        print("  abgelehnt, Beispiele:", errors[:2])


if __name__ == "__main__":
    main()
