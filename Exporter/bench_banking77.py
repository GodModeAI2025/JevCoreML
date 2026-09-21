#!/usr/bin/env python3
"""Banking77 gegen einen /v1/systemone-Server, in der Form des veröffentlichten Jev-Vergleichs.

Zwei Aufgaben, wie im AY-Benchmark vom 19.09.2026:
  intent77  alle 77 Kategorien
  intent8   acht Kategorien aus dem Karten-/Zahlungscluster

Wichtig zur Einordnung: kev wurde auf dem Trainings-Split von banking77 feinjustiert
(1000 Beispiele, siehe Modellkarte). Bewertet wird hier der Test-Split, also ungesehene
Zeilen, aber dieselbe Domäne. Jev kennt diese Domäne nicht als eigenes Trainingsziel.
Die Zahlen sind deshalb kein Beleg für ein besseres Modell, sondern für ein zugeschnittenes.
"""
import argparse
import json
import random
import statistics
import time

import httpx

INSTRUCTION = "Which banking intent best describes this customer message?"

# Die acht Kategorien, die in der Fehleranalyse des veröffentlichten Tests vorkommen.
# Der Originalsatz ist nicht veröffentlicht; dies ist ein Nachbau derselben Form, nicht
# derselben Zeilen.
INTENT8 = [
    "card_arrival", "card_delivery_estimate", "card_not_working",
    "card_payment_not_recognised", "declined_card_payment",
    "direct_debit_payment_not_recognised", "getting_spare_card", "order_physical_card",
]


def load_split(limit_per_class=None, categories=None, seed=0):
    from datasets import load_dataset
    ds = load_dataset("legacy-datasets/banking77", split="test",
                      revision="f54121560de48f2852f90be299010d1d6dc612ec")
    names = ds.features["label"].names
    keep = set(categories) if categories else set(names)
    rng = random.Random(seed)
    by_class = {}
    for row in ds:
        name = names[row["label"]]
        if name not in keep:
            continue
        by_class.setdefault(name, []).append(row["text"])
    items = []
    for name in sorted(by_class):
        texts = by_class[name]
        rng.shuffle(texts)
        for text in (texts[:limit_per_class] if limit_per_class else texts):
            items.append({"text": text, "label": name})
    rng.shuffle(items)
    return items, sorted(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8010")
    ap.add_argument("--task", choices=["intent77", "intent8"], default="intent77")
    ap.add_argument("--limit", type=int, default=0, help="Zeilen insgesamt, 0 = alle")
    ap.add_argument("--per-class", type=int, default=0)
    ap.add_argument("--label", default="nativ")
    ap.add_argument("--describe", action="store_true",
                    help="Kriterien mit Beschreibung statt nur Name, naeher an den 360 Eingabe-Tokens des veroeffentlichten Tests")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    categories = INTENT8 if args.task == "intent8" else None
    items, names = load_split(args.per_class or None, categories)
    if args.limit:
        items = items[: args.limit]
    criteria = ({name: f"Issue concerning {name.replace('_', ' ')}" for name in names}
                if args.describe else {name: None for name in names})

    correct, confidence, latencies = [], [], []
    confusion = {}
    input_tokens = []
    failures = []

    with httpx.Client(timeout=600) as client:
        client.post(f"{args.base}/v1/systemone",
                    json={"state": "warm", "questions": {"q": {"type": "noul", "instructions": "warm"}}})
        for item in items:
            body = {"state": item["text"], "model": "jev-latest",
                    "questions": {"intent": {"type": "choice", "instructions": INSTRUCTION,
                                             "criteria": criteria}}}
            t0 = time.perf_counter()
            response = client.post(f"{args.base}/v1/systemone", json=body)
            latencies.append((time.perf_counter() - t0) * 1000)
            if response.status_code != 200:
                failures.append(response.text[:160])
                continue
            payload = response.json()
            answer = payload["answers"]["intent"]
            ok = answer["choice"] == item["label"]
            correct.append(1.0 if ok else 0.0)
            confidence.append(answer["confidence"])
            input_tokens.append(payload.get("usage", {}).get("input_tokens", 0))
            if not ok:
                key = f"{item['label']} -> {answer['choice']}"
                confusion[key] = confusion.get(key, 0) + 1

    n = len(correct)
    gating = []
    for threshold in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]:
        idx = [i for i, c in enumerate(confidence) if c >= threshold]
        gating.append({"min_conf": threshold,
                       "coverage": len(idx) / n if n else 0.0,
                       "accuracy": (sum(correct[i] for i in idx) / len(idx)) if idx else 0.0})

    summary = {
        "label": args.label, "task": args.task, "base": args.base, "described_criteria": args.describe,
        "categories": len(names), "n": n, "failed": len(failures),
        "accuracy": sum(correct) / n if n else 0.0,
        "latency_ms": {"p50": statistics.median(latencies), "p95": sorted(latencies)[int(0.95 * len(latencies))],
                       "mean": statistics.fmean(latencies)},
        "input_tokens_mean": statistics.fmean(input_tokens) if input_tokens else 0,
        "gating": gating,
        "top_confusions": sorted(confusion.items(), key=lambda kv: -kv[1])[:10],
        "failures": failures[:3],
    }
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"{args.label} / {args.task}: n={n}, {len(names)} Kategorien, "
          f"Trefferquote {summary['accuracy']*100:.1f} %")
    print(f"  Latenz p50 {summary['latency_ms']['p50']:.0f} ms  p95 {summary['latency_ms']['p95']:.0f} ms  "
          f"Eingabe-Tokens im Mittel {summary['input_tokens_mean']:.0f}")
    for g in gating:
        if g["min_conf"] in (0.0, 0.8, 0.9):
            print(f"  Schwelle {g['min_conf']:.2f}: Abdeckung {g['coverage']*100:5.1f} %  "
                  f"Trefferquote {g['accuracy']*100:5.1f} %")


if __name__ == "__main__":
    main()
