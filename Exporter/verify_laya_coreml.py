#!/usr/bin/env python3
"""Prueft ein laya-.mlpackage gegen die Referenzdaten aus dump_laya_golden.py.

Verglichen wird auf drei Ebenen, weil nur die dritte zaehlt:
  Logits          roher Abstand, sagt etwas ueber die Rechengenauigkeit
  Wahrscheinlichkeiten  nach Temperatur und Softmax, das ist die Ausgabe
  Antwort         gewaehlte Option, Punktwert, Konfidenz: kippt hier etwas, ist der Port kaputt

Jedes Backend einzeln, denn die Rechenwege unterscheiden sich und eines davon liefert
still falsche Zahlen, wenn man es laesst.
"""
import argparse
import json
import math
import pathlib

import numpy as np

import layax


def softmax_with_temperature(logits, k, scale):
    z = np.asarray(logits[:k], dtype=np.float64) / max(1e-3, float(scale))
    p = np.exp(z - z.max())
    return p / p.sum()


def confidence_from_probs(p, k):
    if k < 2:
        return 1.0
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def temp_bucket(qtype, k):
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % ({0: "choice", 1: "score", 2: "noul"}[int(qtype)], size)


def answer_from(probs, record, runtime, act_probability):
    """Baut die Antwort aus den Wahrscheinlichkeiten, Zeile fuer Zeile wie laya.agent."""
    k = len(probs)
    conf = round(confidence_from_probs(probs, k), 4)
    ext = {"act_probability": round(float(act_probability), 4)}
    qtype = record["qtype"]
    if qtype == 0:
        keys = list(record["request"]["questions"][record["question_id"]]["criteria"])
        return {"type": "choice", "choice": keys[int(probs.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, probs)},
                "confidence": conf, "action": ext}
    if qtype == 1:
        crit = record["request"]["questions"][record["question_id"]]["criteria"]
        return {"type": "score", "score": round(float((np.arange(k) * probs).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(crit)},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(probs)},
                "confidence": conf, "action": ext}
    return {"type": "noul", "noul": round(float(probs[1]), 4),
            "confidence": round(max(float(probs[1]), 1.0 - float(probs[1])), 4), "action": ext}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden", default="")
    ap.add_argument("--checkpoint", default="english")
    ap.add_argument("--units", default="all", choices=["all", "cpu", "gpu", "ane"])
    ap.add_argument("--report", default="")
    ap.add_argument("--timing", type=int, default=0,
                    help="je Rechenwerk so viele Vorhersagen auf dem ersten Record messen, 0 = aus")
    args = ap.parse_args()

    import coremltools as ct

    golden = pathlib.Path(args.golden or (layax.ROOT / "Golden" / "laya" / args.checkpoint))
    runtime = layax.read_json(golden / "runtime.json")
    records = layax.read_json(golden / "records.json")["records"]
    by_temp = runtime["temperature_by_options"]
    base_temp = runtime["temperature"]

    units = {"cpu": [("CPU", ct.ComputeUnit.CPU_ONLY)], "gpu": [("GPU", ct.ComputeUnit.CPU_AND_GPU)],
             "ane": [("ANE", ct.ComputeUnit.CPU_AND_NE)]}
    units["all"] = units["cpu"] + units["gpu"] + units["ane"]
    spec = ct.models.MLModel(args.model, compute_units=ct.ComputeUnit.CPU_ONLY).get_spec()
    ids_type = [d for d in spec.description.input if d.name == "input_ids"][0].type.multiArrayType
    length = ids_type.shape[1]
    options = [d for d in spec.description.input if d.name == "marker_pos"][0].type.multiArrayType.shape[0]
    # Mehrere Längen im Paket: jede Frage läuft wie in der Swift-Laufzeit in der kürzesten, in
    # die sie passt, und jede Länge hat eine eigene Instanz. Eine Instanz, die zwischen Längen
    # wechselt, bereitet den Graphen jedes Mal neu vor und braucht dann 450 ms statt 10.
    lengths = sorted({int(s.shape[1]) for s in ids_type.enumeratedShapes.shapes}) or [int(length)]

    report = {"model": str(args.model), "checkpoint": args.checkpoint,
              "sequence_length": int(length), "sequence_lengths": lengths, "max_options": int(options),
              "records": len(records), "backends": {}}

    for label, unit in units[args.units]:
        instances = {}

        def instance(n):
            if n not in instances:
                instances[n] = ct.models.MLModel(args.model, compute_units=unit)
            return instances[n]

        used = {}
        dl = dp = 0.0
        conf_max = value_max = 0.0
        # Der Handlungskopf, getrennt geführt. Seine rohen Logits sind groß und gesättigt; was
        # zählt, ist act_probability, die laya ausgibt.
        dact_logit = dact_p = 0.0
        flipped = []
        for rec in records:
            ids, markers = rec["encoding"]["ids"], rec["encoding"]["markers"]
            k = len(markers)
            n = next(x for x in lengths if x >= len(ids))
            used[n] = used.get(n, 0) + 1
            inputs = layax.model_inputs(ids, markers, rec["qtype"], n, int(options),
                                        runtime["pad_id"])
            out = instance(n).predict({key: v.numpy() for key, v in inputs.items()})
            got = np.asarray(out["logits"]).reshape(-1)
            act = np.asarray(out["act_logits"]).reshape(-1)
            ref = np.asarray(rec["logits"], dtype=np.float64)
            dl = max(dl, float(np.max(np.abs(got[:k] - ref[:k]))))

            scale = by_temp.get(temp_bucket(rec["qtype"], k), base_temp[rec["qtype"]])
            p_got = softmax_with_temperature(got, k, scale)
            p_ref = softmax_with_temperature(ref, k, scale)
            dp = max(dp, float(np.max(np.abs(p_got - p_ref))))

            e_act = np.exp(act - act.max())
            answer = answer_from(p_got, rec, runtime, (e_act / e_act.sum())[0])
            ref_act = np.asarray(rec["act_logits"], dtype=np.float64)
            e_ref = np.exp(ref_act - ref_act.max())
            dact_logit = max(dact_logit, float(np.max(np.abs(act - ref_act))))
            dact_p = max(dact_p, abs(float((e_act / e_act.sum())[0]) - float((e_ref / e_ref.sum())[0])))
            want = rec["answer"]
            key = want["type"]
            # Zwei Fragen, nicht eine: kippt die Entscheidung, und wie weit liegt der Zahlenwert
            # daneben. Eine Entscheidung kippt bei choice, wenn eine andere Option gewinnt, bei
            # noul, wenn p die 0,5 ueberquert, bei score, wenn die gerundete Stufe wechselt.
            if key == "choice":
                kipp = answer["choice"] != want["choice"]
            elif key == "noul":
                kipp = (float(answer["noul"]) >= 0.5) != (float(want["noul"]) >= 0.5)
            else:
                kipp = round(float(answer["score"])) != round(float(want["score"]))
                value_max = max(value_max, abs(float(answer["score"]) - float(want["score"])))
            if key == "noul":
                value_max = max(value_max, abs(float(answer["noul"]) - float(want["noul"])))
            if kipp:
                flipped.append({"id": rec["question_id"], "erwartet": want[key], "erhalten": answer[key]})
            conf_max = max(conf_max, abs(answer["confidence"] - want["confidence"]))

        timing = {}
        if args.timing:
            # Dieselbe Eingabe, derselbe Prozess, direkt nach der Genauigkeitsprüfung. So stehen
            # Abstand und Zeit eines Rechenwerks aus einem Lauf nebeneinander.
            import statistics
            import time
            first = records[0]
            n = next(x for x in lengths if x >= len(first["encoding"]["ids"]))
            ml = instance(n)
            feed = {key: v.numpy() for key, v in layax.model_inputs(
                first["encoding"]["ids"], first["encoding"]["markers"], first["qtype"],
                n, int(options), runtime["pad_id"]).items()}
            for _ in range(3):
                ml.predict(feed)
            times = []
            for _ in range(args.timing):
                t0 = time.perf_counter()
                ml.predict(feed)
                times.append((time.perf_counter() - t0) * 1000)
            timing = {"median_ms": statistics.median(times), "min_ms": min(times), "runs": len(times),
                      "timing_length": n}
        report["backends"][label] = {"max_dlogit": dl, "max_dp": dp, "max_dwert": value_max, **timing,
                                     "max_dconfidence": conf_max, "max_dact_logit": dact_logit,
                                     "max_dact_probability": dact_p,
                                     "records_per_length": {str(k): v for k, v in sorted(used.items())},
                                     "gekippte_entscheidungen": len(flipped), "details": flipped}
        print(f"{label:4s} max|dlogit|={dl:.3e}  max|dp|={dp:.3e}  max|dWert|={value_max:.4f}  "
              f"max|dKonfidenz|={conf_max:.4f}  max|dact|={dact_logit:.3f} (p {dact_p:.1e})  "
              f"gekippt={len(flipped)}/{len(records)}  Längen {dict(sorted(used.items()))}"
              + (f"  Median {timing['median_ms']:.1f} ms" if timing else ""))
        for f in flipped:
            print(f"     {f['id']}: erwartet {f['erwartet']!r}, erhalten {f['erhalten']!r}")

    if args.report:
        layax.write_json(args.report, report)
        print(f"Bericht: {args.report}")


if __name__ == "__main__":
    main()
