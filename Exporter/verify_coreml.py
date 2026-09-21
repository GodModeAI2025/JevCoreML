#!/usr/bin/env python3
"""Vergleicht das .mlpackage gegen die PyTorch-Golden-Logits, pro Compute-Unit.

Ausgegeben werden maximale Logit- und Wahrscheinlichkeitsabweichung sowie Argmax-Flips.
Ein Argmax-Flip ist das einzige Kriterium, das eine Entscheidung tatsächlich verändert.
"""
import argparse
import json
import time

import numpy as np

import jevx

ORDER = ["input_ids", "position_ids", "attention_mask", "decide_index", "option_indices"]


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden", default=str(jevx.ROOT / "Golden" / "records.json"))
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--options", type=int, default=8)
    ap.add_argument("--units", nargs="*", default=["CPU_ONLY", "CPU_AND_GPU", "ALL"])
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    import coremltools as ct

    golden = jevx.read_json(args.golden)["records"]
    runtime = jevx.read_json(jevx.ROOT / "Golden" / "runtime.json")
    pad_id = runtime["pad_id"]

    feeds, refs = [], []
    for r in golden:
        e = r["encoding"]
        enc = {"ids": e["ids"], "pos": e["position_ids"],
               "decide_idx": [e["decide_index"]], "opt_idx": [e["option_indices"]]}
        t = jevx.model_inputs(enc, args.length, args.options, pad_id)
        feeds.append({k: t[k].numpy() for k in ORDER})
        refs.append((np.asarray(r["logits"], dtype=np.float64), np.asarray(r["probabilities"], dtype=np.float64)))

    report = {}
    for unit in args.units:
        try:
            model = ct.models.MLModel(args.model, compute_units=getattr(ct.ComputeUnit, unit))
        except Exception as exc:
            print(f"{unit:12s} nicht verfügbar: {exc}")
            continue
        dl, dp, flips, times = 0.0, 0.0, 0, []
        for feed, (ref_logits, ref_probs) in zip(feeds, refs):
            k = len(ref_logits)
            t0 = time.perf_counter()
            for _ in range(args.repeat):
                out = model.predict(feed)
            times.append((time.perf_counter() - t0) / args.repeat * 1000)
            got = np.asarray(out["logits"], dtype=np.float64).reshape(-1)[:k]
            dl = max(dl, float(np.max(np.abs(got - ref_logits))))
            p = softmax(got)
            dp = max(dp, float(np.max(np.abs(p - ref_probs))))
            flips += int(np.argmax(p) != np.argmax(ref_probs))
        report[unit] = {"max_abs_logit_delta": dl, "max_abs_prob_delta": dp,
                        "argmax_flips": flips, "median_latency_ms": float(np.median(times))}
        print(f"{unit:12s} max|dlogit|={dl:.3e}  max|dp|={dp:.3e}  Flips={flips}/{len(feeds)}  "
              f"Median {np.median(times):6.1f} ms")

    jevx.write_json(jevx.ROOT / "Benchmarks" / f"coreml-parity-{jevx.pathlib.Path(args.model).stem}.json",
                    {"model": args.model, "records": len(feeds), "units": report})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
