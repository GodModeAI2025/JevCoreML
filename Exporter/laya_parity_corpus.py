#!/usr/bin/env python3
"""Parität laya <-> Core-ML-Port auf einem grossen Korpus.

Sechs Referenzfaelle belegen nichts. Diese Pruefung faehrt beide Seiten ueber mehrere hundert
echte Beispiele und zaehlt, wie oft die gewaehlte Antwort auseinandergeht. Verglichen wird
gegen laya selbst, nicht gegen den eigenen Erwartungswert, und zusaetzlich wird die
Treffergenauigkeit beider Seiten gegen die Goldlabels gemessen.
"""
import argparse
import math
import statistics
import time

import numpy as np
import torch

import layax
from bench_public import TASKS
from verify_laya_coreml import confidence_from_probs, softmax_with_temperature, temp_bucket


def main():
    layax.quiet()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", default="english")
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--units", default="gpu", choices=["cpu", "gpu", "ane"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import coremltools as ct
    from datasets import load_dataset
    from laya.common import QTYPES, build_sequence

    spec = TASKS[args.task]
    ds = load_dataset(spec["repo"], split=spec["split"])
    names = spec["labels"]
    rows = [{"text": r[spec["text"]], "label": names[int(r["label"])]} for r in ds][: args.limit]

    qdef = ({"type": "score", "instructions": spec["instructions"], "criteria": names}
            if spec.get("score") else
            {"type": "choice", "instructions": spec["instructions"], "criteria": spec["criteria"]})

    agent = layax.load_laya(args.checkpoint)
    tok, model = agent.tok, agent.model
    cfg = layax.CHECKPOINTS[args.checkpoint]
    max_len = agent.cfg.get("max_len", cfg["max_len"])
    head_max_len = agent.cfg.get("head_max_len", cfg["head_max_len"])
    q = agent._to_internal(qdef)
    qtype = QTYPES[q["t"]]

    unit = {"cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU,
            "ane": ct.ComputeUnit.CPU_AND_NE}[args.units]
    mspec = ct.models.MLModel(args.model, compute_units=ct.ComputeUnit.CPU_ONLY).get_spec()
    ids_type = [d for d in mspec.description.input if d.name == "input_ids"][0].type.multiArrayType
    length = ids_type.shape[1]
    options = [d for d in mspec.description.input if d.name == "marker_pos"][0].type.multiArrayType.shape[0]
    # Wie die Swift-Laufzeit: je Frage die kürzeste Länge des Pakets, je Länge eine Instanz.
    lengths = sorted({int(x.shape[1]) for x in ids_type.enumeratedShapes.shapes}) or [int(length)]
    instances = {}
    used = {}
    by_temp = agent.cfg.get("temperature_by_options", {})
    base_temp = [float(x) for x in agent.temperature]

    same = 0
    dp_max = dl_max = 0.0
    conf_max = 0.0
    hit_laya = hit_port = 0
    t_laya, t_port = [], []
    disagree = []
    skipped = 0

    for row in rows:
        ids, markers = build_sequence(tok, row["text"], q, max_len, head_max_len)
        k = len(markers)
        if len(ids) > length:
            skipped += 1
            continue
        n_len = next(x for x in lengths if x >= len(ids))
        used[n_len] = used.get(n_len, 0) + 1
        if n_len not in instances:
            instances[n_len] = ct.models.MLModel(args.model, compute_units=unit)
            warm = layax.model_inputs(ids, markers, qtype, n_len, int(options), tok.pad_token_id)
            instances[n_len].predict({key: v.numpy() for key, v in warm.items()})
        ml = instances[n_len]
        inputs = layax.model_inputs(ids, markers, qtype, n_len, int(options), tok.pad_token_id)

        # laya auf der ungepolsterten Sequenz, so wie system_one eine einzelne Frage rechnet: es
        # polstert nur auf die längste Sequenz im Batch, bei einer Frage also gar nicht. Vorher
        # lief laya hier auf 512 aufgefüllt, und die Zeit maß vor allem das Auffüllen.
        unpadded = layax.model_inputs(ids, markers, qtype, len(ids), k, tok.pad_token_id)
        t0 = time.perf_counter()
        with torch.no_grad():
            ref_logits, _ = model(unpadded["input_ids"].long(), unpadded["attention_mask"].long(),
                                  unpadded["marker_pos"].long()[None, :],
                                  unpadded["marker_mask"].bool()[None, :], unpadded["qtype"].long())
        t_laya.append((time.perf_counter() - t0) * 1000)
        ref = ref_logits.reshape(-1).numpy().astype(np.float64)

        t0 = time.perf_counter()
        out = ml.predict({key: v.numpy() for key, v in inputs.items()})
        t_port.append((time.perf_counter() - t0) * 1000)
        got = np.asarray(out["logits"]).reshape(-1).astype(np.float64)

        dl_max = max(dl_max, float(np.max(np.abs(got[:k] - ref[:k]))))
        scale = by_temp.get(temp_bucket(qtype, k), base_temp[qtype])
        p_ref = softmax_with_temperature(ref, k, scale)
        p_got = softmax_with_temperature(got, k, scale)
        dp_max = max(dp_max, float(np.max(np.abs(p_got - p_ref))))
        conf_max = max(conf_max, abs(confidence_from_probs(p_got, k) - confidence_from_probs(p_ref, k)))

        if spec.get("score"):
            pick_ref = names[int(round(float((np.arange(k) * p_ref).sum())))]
            pick_got = names[int(round(float((np.arange(k) * p_got).sum())))]
        else:
            keys = list(spec["criteria"])
            pick_ref, pick_got = keys[int(p_ref.argmax())], keys[int(p_got.argmax())]
        if pick_ref == pick_got:
            same += 1
        else:
            disagree.append({"text": row["text"][:90], "laya": pick_ref, "port": pick_got,
                             "p_laya": round(float(p_ref.max()), 4), "p_port": round(float(p_got.max()), 4)})
        hit_laya += pick_ref == row["label"]
        hit_port += pick_got == row["label"]

    n = len(rows) - skipped
    result = {
        "task": args.task, "checkpoint": args.checkpoint, "modell": str(args.model),
        "backend": args.units, "n": n, "uebersprungen_zu_lang": skipped,
        "gleiche_antwort": same, "gleiche_antwort_quote": same / n if n else 0.0,
        "max_dlogit": dl_max, "max_dp": dp_max, "max_dkonfidenz": conf_max,
        "treffer_laya": hit_laya / n if n else 0.0, "treffer_port": hit_port / n if n else 0.0,
        "median_ms_laya": statistics.median(t_laya) if t_laya else 0.0,
        "zeitmessung": "laya ungepolstert wie system_one mit einer Frage, der Port in der kürzesten "
                       "passenden Länge des Pakets; beide im selben Prozess abwechselnd, daher "
                       "langsamer als in laya_baseline.py",
        "zeilen_je_laenge": {str(k): v for k, v in sorted(used.items())},
        "median_ms_port": statistics.median(t_port) if t_port else 0.0,
        "abweichungen": disagree[:20],
    }
    print(f"{args.task}: {n} Beispiele, Backend {args.units}")
    print(f"  gleiche Antwort   {same}/{n} = {result['gleiche_antwort_quote']*100:.2f} %")
    print(f"  max|dlogit|       {dl_max:.3e}")
    print(f"  max|dp|           {dp_max:.3e}")
    print(f"  max|dKonfidenz|   {conf_max:.4f}")
    print(f"  Treffer laya      {result['treffer_laya']*100:.1f} %")
    print(f"  Treffer Port      {result['treffer_port']*100:.1f} %")
    print(f"  Median laya/Port  {result['median_ms_laya']:.1f} ms / {result['median_ms_port']:.1f} ms")
    for d in disagree[:5]:
        print(f"    laya={d['laya']!r} ({d['p_laya']}) vs Port={d['port']!r} ({d['p_port']}): {d['text']}")
    if args.out:
        layax.write_json(args.out, result)
        print(f"  Bericht: {args.out}")


if __name__ == "__main__":
    main()
