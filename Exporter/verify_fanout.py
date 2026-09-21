#!/usr/bin/env python3
"""Vergleicht ein Phase-2-Paket (Fan-Out-Vertrag) gegen die PyTorch-Golden-Daten.

Geprüft wird je Recheneinheit und je Eingabelänge des Pakets: die sechs Einzelanfragen aus
Golden/records.json und der gepackte Dreifragenlauf aus Golden/packed.json. Ausgegeben werden
maximale Logit- und Wahrscheinlichkeitsabweichung, Argmax-Flips und die Latenz je Länge, mit
einer eigenen Instanz je Länge, so wie die Laufzeit es hält.

  verify_fanout.py --model ../Models/JevCoreML.mlpackage
  verify_fanout.py --model ... --compare ../Models/Kev06B-Q4-fp16.mlpackage
"""
import argparse
import pathlib
import time

import numpy as np

import jevx

ORDER = ["input_ids", "position_ids", "segment_ids", "decide_index", "option_indices"]


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def contract(spec):
    """Längen, Q und K aus der Modellbeschreibung, aufgezählte Formen eingeschlossen."""
    by_name = {i.name: i.type.multiArrayType for i in spec.description.input}
    ids = by_name["input_ids"]
    lengths = sorted({int(s.shape[-1]) for s in ids.enumeratedShapes.shapes}) or [int(ids.shape[-1])]
    q = int(by_name["decide_index"].shape[0])
    k = int(by_name["option_indices"].shape[-1])
    return lengths, q, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--compare", default="", help="zweites Phase-2-Paket, gleiche Eingaben, Logits gegeneinander")
    ap.add_argument("--units", nargs="*", default=["CPU_ONLY", "CPU_AND_GPU"])
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    import coremltools as ct

    runtime = jevx.read_json(jevx.ROOT / "Golden" / "runtime.json")
    pad_id = runtime["pad_id"]
    records = jevx.read_json(jevx.ROOT / "Golden" / "records.json")["records"]
    packed = jevx.read_json(jevx.ROOT / "Golden" / "packed.json")["records"]

    q_id = jevx.q_token_id(packed[0])
    for p in packed:
        if jevx.segments_from_ids(p["encoding"]["ids"], q_id) != p["encoding"]["segment_ids"]:
            raise SystemExit("Segmentregel passt nicht zu packed.json")

    cases = []
    for r in records:
        cases.append((jevx.single_enc(r, q_id),
                      [(np.asarray(r["logits"], dtype=np.float64),
                        np.asarray(r["probabilities"], dtype=np.float64))]))
    for p in packed:
        cases.append((jevx.packed_enc(p),
                      [(np.asarray(q["logits"], dtype=np.float64),
                        np.asarray(q["probabilities"], dtype=np.float64)) for q in p["questions"]]))

    spec = ct.models.MLModel(args.model, skip_model_load=True).get_spec()
    lengths, Q, K = contract(spec)
    print(f"{pathlib.Path(args.model).name}: L={lengths} Q={Q} K={K}, {len(cases)} Fälle")
    compiled = ct.models.utils.compile_model(args.model)
    other = None
    if args.compare:
        other_spec = ct.models.MLModel(args.compare, skip_model_load=True).get_spec()
        other = (ct.models.utils.compile_model(args.compare), *contract(other_spec))
        print(f"Vergleich mit {pathlib.Path(args.compare).name}: L={other[1]} Q={other[2]} K={other[3]}")

    report = {}
    for unit in args.units:
        cu = getattr(ct.ComputeUnit, unit)
        per_length = {}
        for n in lengths:
            fitting = [c for c in cases if len(c[0]["ids"]) <= n and len(c[0]["decide_idx"]) <= Q]
            if not fitting:
                continue
            model = ct.models.CompiledMLModel(compiled, compute_units=cu)
            dl, dp, flips, questions = 0.0, 0.0, 0, 0
            # None, solange kein Fall gegen das Vergleichspaket gerechnet wurde: das passt nur in
            # Längen, die beide Pakete annehmen. Eine 0 hieße sonst "identisch" statt "nicht geprüft".
            cross = None
            for enc, refs in fitting:
                feed = {k: v.numpy() for k, v in jevx.multi_inputs(enc, n, Q, K, pad_id).items()}
                out = np.asarray(model.predict(feed)["logits"], dtype=np.float64).reshape(Q, K)
                for qi, (ref_logits, ref_probs) in enumerate(refs):
                    k = len(ref_logits)
                    got = out[qi, :k]
                    dl = max(dl, float(np.max(np.abs(got - ref_logits))))
                    p = softmax(got)
                    dp = max(dp, float(np.max(np.abs(p - ref_probs))))
                    flips += int(np.argmax(p) != np.argmax(ref_probs))
                    questions += 1
                if other and n in other[1] and len(enc["decide_idx"]) <= other[2]:
                    ko = min(K, other[3])
                    if all(len(x) <= ko for x in enc["opt_idx"]):
                        omodel = ct.models.CompiledMLModel(other[0], compute_units=cu)
                        ofeed = {k: v.numpy() for k, v in
                                 jevx.multi_inputs(enc, n, other[2], other[3], pad_id).items()}
                        oout = np.asarray(omodel.predict(ofeed)["logits"], dtype=np.float64).reshape(other[2], other[3])
                        nq = len(enc["decide_idx"])
                        delta = float(np.max(np.abs(out[:nq, :ko] - oout[:nq, :ko])))
                        cross = delta if cross is None else max(cross, delta)
                        del omodel
            # Latenz: der letzte gepackte Lauf (zwei Fragen, 45 Token), auf n gepolstert, nach einem Warmlauf.
            enc = cases[-1][0]
            feed = {k: v.numpy() for k, v in jevx.multi_inputs(enc, n, Q, K, pad_id).items()}
            model.predict(feed)
            times = []
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                model.predict(feed)
                times.append((time.perf_counter() - t0) * 1000)
            per_length[n] = {"cases": len(fitting), "questions": questions,
                             "max_abs_logit_delta": dl, "max_abs_prob_delta": dp,
                             "argmax_flips": flips, "median_latency_ms": float(np.median(times)),
                             **({"max_abs_logit_delta_vs_compare": cross} if other else {})}
            line = (f"{unit:12s} L={n:5d}  {len(fitting)} Fälle/{questions} Fragen  max|dlogit|={dl:.3e}  "
                    f"max|dp|={dp:.3e}  Flips={flips}  Median {np.median(times):6.1f} ms")
            if other:
                line += "  |Δ| zu Vergleich " + (f"{cross:.3e}" if cross is not None else "n. v.")
            print(line)
            del model
        report[unit] = per_length

    stem = pathlib.Path(args.model).stem
    jevx.write_json(jevx.ROOT / "Benchmarks" / f"fanout-parity-{stem}.json",
                    {"model": args.model, "compare": args.compare or None,
                     "lengths": lengths, "questions": Q, "options": K, "units": report})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
