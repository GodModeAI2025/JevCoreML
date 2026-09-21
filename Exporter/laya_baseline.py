#!/usr/bin/env python3
"""laya auf dieser Maschine messen.

Die 33 ms aus deren README stammen von einer T4. Fuer die Frage, ob unser Port mithaelt,
zaehlt nur die Zahl auf derselben Hardware. Gemessen wird, was Router.predict tatsaechlich
zurueckgibt, mit preload, also ohne Nachladekosten.
"""
import argparse, json, statistics, time

STATE = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": ("Hi, we were billed twice for March. Please refund the duplicate today "
             "or we will cancel our plan."),
}
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "sales": "pricing, new contracts",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
}


def cases():
    """Die fünf Messfälle, gemeinsam mit compare_laya_port.py, damit beide dasselbe messen."""
    one = {"department": QUESTIONS["department"]}
    ten = {f"{k}_{i}": v for i in range(3) for k, v in QUESTIONS.items()}
    ten = dict(list(ten.items())[:10])
    return [
        ("1 Frage, englisch", STATE, one),
        ("4 Fragen, englisch", STATE, QUESTIONS),
        ("10 Fragen, englisch", STATE, ten),
        ("1 Frage, deutsch", {"body": "Der Kunde wurde zweimal belastet und will sein Geld zurueck."}, one),
        ("1 Frage, hindi", {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}, one),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import laya
    from laya import Router

    t0 = time.time()
    router = Router(preload=True, device=args.device)
    load_seconds = time.time() - t0
    print(f"Router(preload=True, device={args.device}) in {load_seconds:.1f}s")

    out = {"device": args.device, "load_seconds": load_seconds, "runs": []}

    def bench(label, state, questions, model=None):
        kw = {"model": model} if model else {}
        first = router.predict(state, questions, **kw)          # Warmlauf
        times = []
        for _ in range(args.repeat):
            t = time.perf_counter()
            res = router.predict(state, questions, **kw)
            times.append((time.perf_counter() - t) * 1000)
        entry = {"label": label, "questions": len(questions),
                 "routed_to": first.get("routing", {}).get("model"),
                 "median_ms": statistics.median(times),
                 "min_ms": min(times), "p95_ms": sorted(times)[int(0.95 * len(times))],
                 "per_question_ms": statistics.median(times) / len(questions),
                 "answers": {k: {kk: vv for kk, vv in v.items() if kk != "probabilities"}
                             for k, v in first["answers"].items()}}
        out["runs"].append(entry)
        print(f"  {label:34s} {len(questions):2d} Fragen -> {entry['routed_to']:14s} "
              f"Median {entry['median_ms']:7.1f} ms  ({entry['per_question_ms']:6.1f} ms/Frage)")
        return first

    for label, state, questions in cases():
        bench(label, state, questions)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False); fh.write("\n")


if __name__ == "__main__":
    main()
