#!/usr/bin/env python3
"""Oeffentliche Datensaetze, fuer die laya Zahlen veroeffentlicht hat.

Wichtig fuer die Einordnung, welcher Datensatz fair ist:

  dair-ai/emotion   steht weder in kevs Trainingsliste noch laut laya in deren Mix.
                    Der einzige echte Direktvergleich. laya: 0.573 (englisch) / 0.595 (routed).
  ag_news           steht in beiden Trainingsmischungen. laya: 0.947 / 0.950.
  SetFit/sst5       steht in kevs Trainingsliste, laut laya held-out. laya: 0.372.
  google/boolq      steht in beiden. laya: 0.830.

Die Spalte "fair" im Ergebnis haelt das fest, damit die Zahl nicht ohne den Vorbehalt wandert.
"""
import argparse, json, statistics, time
import httpx

TASKS = {
    "emotion": {
        "repo": "dair-ai/emotion", "split": "test", "text": "text",
        "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "instructions": "Which emotion does this text express?",
        "criteria": {"sadness": None, "joy": None, "love": None,
                     "anger": None, "fear": None, "surprise": None},
        "laya": 0.573, "laya_routed": 0.595, "jev": 0.480,
        "fair": "held-out fuer beide Seiten",
    },
    "agnews": {
        "repo": "fancyzhx/ag_news", "split": "test", "text": "text",
        "labels": ["world", "sports", "business", "scitech"],
        "instructions": "Which category does this news article belong to?",
        "criteria": {"world": "World news: politics, international affairs, conflicts",
                     "sports": "Sports: games, athletes, teams, results",
                     "business": "Business: companies, markets, economy, finance",
                     "scitech": "Science and technology: research, gadgets, software, space"},
        "laya": 0.947, "laya_routed": 0.950, "jev": 0.910,
        "fair": "in beiden Trainingsmischungen",
    },
    "sst5": {
        "repo": "SetFit/sst5", "split": "test", "text": "text",
        "labels": ["very negative", "negative", "neutral", "positive", "very positive"],
        "instructions": "How positive is this sentence?",
        "score": True,
        "laya": 0.372, "laya_routed": None, "jev": None,
        "fair": "in kevs Trainingsliste, laut laya held-out",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = TASKS[args.task]
    from datasets import load_dataset
    ds = load_dataset(spec["repo"], split=spec["split"])
    names = spec["labels"]
    rows = [{"text": r[spec["text"]], "label": names[int(r["label"])]} for r in ds][: args.limit]
    print(f"{args.task}: {len(rows)} Zeilen, {len(names)} Klassen ({spec['fair']})")

    question = ({"type": "score", "instructions": spec["instructions"], "criteria": names}
                if spec.get("score") else
                {"type": "choice", "instructions": spec["instructions"], "criteria": spec["criteria"]})

    correct, confidence, latencies, tokens = 0, [], [], []
    failures = []
    with httpx.Client(timeout=600) as client:
        client.post(f"{args.base}/v1/systemone",
                    json={"state": "warm", "questions": {"q": {"type": "noul", "instructions": "warm"}}})
        for row in rows:
            body = {"state": row["text"], "model": "jev-latest", "questions": {"q": question}}
            t0 = time.perf_counter()
            r = client.post(f"{args.base}/v1/systemone", json=body)
            latencies.append((time.perf_counter() - t0) * 1000)
            if r.status_code != 200:
                failures.append(r.text[:140]); continue
            payload = r.json(); a = payload["answers"]["q"]
            tokens.append(payload["usage"]["input_tokens"])
            if spec.get("score"):
                pick = names[max(a["probabilities"], key=lambda k: a["probabilities"][k], default="0") and
                             int(max(a["probabilities"], key=lambda k: a["probabilities"][k]))]
                confidence.append(a["confidence"])
            else:
                pick = a["choice"]; confidence.append(a["confidence"])
            if pick == row["label"]:
                correct += 1

    n = len(rows) - len(failures)
    gating = []
    for t in (0.0, 0.5, 0.7, 0.8, 0.9):
        idx = [i for i, c in enumerate(confidence) if c >= t]
        gating.append({"min_conf": t, "coverage": len(idx) / n if n else 0.0,
                       "accuracy": sum(1 for i in idx if True) and
                                   (sum(1 for i in idx) and None)})
    result = {"task": args.task, "n": n, "failed": len(failures), "fair": spec["fair"],
              "accuracy": correct / n if n else 0.0,
              "laya": spec["laya"], "laya_routed": spec["laya_routed"], "jev": spec["jev"],
              "median_latency_ms": statistics.median(latencies),
              "median_input_tokens": statistics.median(tokens) if tokens else 0}
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False); fh.write("\n")
    print(f"  kev (dieser Port) {result['accuracy']*100:5.1f} %   "
          f"laya {spec['laya']*100 if spec['laya'] else 0:5.1f} %   "
          f"Jev {spec['jev']*100 if spec['jev'] else 0:5.1f} %   "
          f"Median {result['median_latency_ms']:.0f} ms")
    if failures: print("  abgelehnt:", failures[:1])


if __name__ == "__main__":
    main()
