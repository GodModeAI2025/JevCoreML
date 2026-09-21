#!/usr/bin/env python3
"""laya-Checkpoint -> Core-ML-.mlpackage.

Vertrag, Batch 1, feste Shapes:
  input_ids      [1, L]  int32
  attention_mask [1, L]  int32
  marker_pos     [K]     int32   Position je Option, ungenutzte Plaetze zeigen auf 0
  marker_mask    [K]     int32   1 fuer belegte Optionsplaetze, 0 sonst
  qtype          [1]     int32   0 choice, 1 score, 2 noul
  -> logits      [K]     float32  unbelegte Plaetze auf -1e4, wie in laya
  -> act_logits  [2]     float32  Handlungskopf, roh

K ist in der Vorgabe gleich L. laya kennt keine Obergrenze für Optionen: es kürzt jede Option
bis auf vier Token und behält jede Marke, die noch vor max_len liegt. Gemessen nimmt laya 169
Optionen auf dem englischen Checkpoint, 257 auf dem mehrsprachigen und 340 auf typed-decisions.
Mehr Marken als Positionen kann es nicht geben, also deckt K = L alles ab. Der Scorer auf K
Positionen kostet rechnerisch weniger als ein Prozent des Encoders.

L ist nicht eine Länge, sondern mehrere: 128, 256 und so weiter bis max_len, alle in einem Paket
(EnumeratedShapes auf input_ids und attention_mask). laya rechnet jede Frage nur so lang, wie
sie ist; der erste Port füllte jede auf 512 oder 1024 auf. Gemessen rechnet L=128 doppelt so
schnell wie L=512, und eine typische Frage hat 50 bis 100 Token. Getrennte Pakete je Länge
hätten jede Länge eine volle Kopie der Gewichte gekostet, auf der Platte und im Speicher. K
bleibt fest beim größten L; der Scorer auf K Plätzen ist auch bei L=128 ein kleiner Posten.

Draussen bleibt nur Konfiguration: die Temperatur je Fragetyp und Optionszahl sowie die
Entropiekonfidenz. Beides steht in den Metadaten des Pakets, damit Swift keine Beidatei braucht.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

import layax

ORDER = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]


def main():
    layax.quiet()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", choices=list(layax.CHECKPOINTS), default="english")
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=0, help="0 = max_len des Checkpoints")
    ap.add_argument("--options", type=int, default=0, help="0 = L, jede Marke, die laya setzen kann")
    ap.add_argument("--lengths", default="",
                    help="Eingabelängen im Paket, etwa 128,256,512; leer = verdoppelnd ab 128 bis L")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    args = ap.parse_args()

    import coremltools as ct

    agent = layax.load_laya(args.checkpoint)
    spec = layax.CHECKPOINTS[args.checkpoint]
    max_len = int(agent.cfg.get("max_len", spec["max_len"]))
    length = args.length or max_len
    args.options = args.options or length
    if args.lengths:
        lengths = sorted({int(x) for x in args.lengths.split(",")})
    else:
        lengths = [n for n in (128, 256, 512, 1024, 2048, 4096) if n < length] + [length]
    if lengths[-1] != length or lengths[0] < 16:
        raise SystemExit(f"--lengths {lengths} muss bei L={length} enden und darf nicht unter 16 beginnen")
    if length < max_len:
        # Kürzer als laya ließe sich exportieren, aber dann schnitte der Port Zustände ab, die
        # laya noch liest. Länger geht: gekürzt wird auf max_len, gepolstert auf L.
        raise SystemExit(f"--length {length} ist kleiner als laya's max_len {max_len}")
    model = agent.model if hasattr(agent, "model") else agent
    model.eval()
    tok = agent.tok if hasattr(agent, "tok") else agent.tokenizer

    module = layax.LayaExport(model, args.options, length).eval()

    # Traceeingabe aus einer echten Frage, nicht aus Nullen.
    from laya.common import build_sequence
    q = {"t": "choice", "ins": "Which department should handle this request?",
         "crit": {"billing": "invoices, payments, refunds", "technical": "bugs and outages",
                  "sales": "pricing", "other": "everything else"}}
    ids, markers = build_sequence(tok, "We were billed twice for March, please refund.", q,
                                  max_len=length, head_max_len=spec["head_max_len"])
    example = layax.model_inputs(ids, markers, layax.QTYPES["choice"], length, args.options,
                                 tok.pad_token_id)

    print(f"torch.export: {args.checkpoint}  L={lengths} K={args.options} ...")
    t0 = time.time()
    # Eine gemeinsame symbolische Länge für input_ids und attention_mask. Alles, was im Graphen
    # davon abhängt (Rotary-Positionen, das gleitende Fenster, die Maske samt Diagonale), wird
    # damit aus der tatsächlichen Länge gerechnet und nicht als Konstante für L eingebacken.
    dynamic = None
    if len(lengths) > 1:
        seq = torch.export.Dim("seq", min=lengths[0], max=length)
        dynamic = {"input_ids": {1: seq}, "attention_mask": {1: seq},
                   "marker_pos": None, "marker_mask": None, "qtype": None}
    with torch.no_grad():
        program = torch.export.export(module, tuple(example[k] for k in ORDER),
                                      dynamic_shapes=dynamic).run_decompositions({})
    print(f"export ok in {time.time() - t0:.1f}s")

    # Referenz aus dem unveraenderten laya: Maskenpatch zuruecknehmen und dessen eigenes
    # forward rechnen lassen. Nur so belegt der Vergleich, dass der Patch nichts verschiebt.
    module.restore_mask()
    with torch.no_grad():
        ref_logits, ref_act = model(
            example["input_ids"].long(), example["attention_mask"].long(),
            example["marker_pos"].long()[None, :], example["marker_mask"].bool()[None, :],
            example["qtype"].long())
    reference = ref_logits.reshape(-1).numpy()
    reference_act = ref_act.reshape(-1).numpy()

    # Warum nicht einfach FLOAT16: ModernBERT-large traegt im Residualstrom Ausreisser von
    # rund 33000. Der Wert selbst passt noch in fp16, sein Quadrat nicht, und genau das rechnet
    # jede LayerNorm fuer die Varianz. Ergebnis war inf, dann NaN in jeder Ausgabe. Die
    # Normalisierungen bleiben deshalb in fp32, alles Rechenintensive wird fp16.
    if args.precision == "fp16":
        precision = ct.transform.FP16ComputePrecision(
            op_selector=lambda op: op.op_type not in ("layer_norm", "rsqrt", "reduce_mean"))
    else:
        precision = ct.precision.FLOAT32
    print(f"convert ({args.precision}) ...")
    t0 = time.time()
    # Namen direkt beim Konvertieren setzen statt hinterher mit ct.utils.rename_feature:
    # das arbeitet auf der Protobuf-Beschreibung, und bei zwei Ausgaben kann die zweite
    # Umbenennung eine treffen, die die erste gerade erzeugt hat.
    inputs = None
    if len(lengths) > 1:
        # Seit macOS 15 dürfen mehrere Eingaben aufgezählte Formen tragen, je gleich viele.
        def enumerated():
            return ct.EnumeratedShapes(shapes=[[1, n] for n in lengths], default=[1, length])
        inputs = [ct.TensorType(name="input_ids", shape=enumerated(), dtype=np.int32),
                  ct.TensorType(name="attention_mask", shape=enumerated(), dtype=np.int32),
                  ct.TensorType(name="marker_pos", shape=(args.options,), dtype=np.int32),
                  ct.TensorType(name="marker_mask", shape=(args.options,), dtype=np.int32),
                  ct.TensorType(name="qtype", shape=(1,), dtype=np.int32)]
    ml = ct.convert(program, convert_to="mlprogram", compute_precision=precision, inputs=inputs,
                    outputs=[ct.TensorType(name="logits"), ct.TensorType(name="act_logits")],
                    minimum_deployment_target=ct.target.macOS15)
    print(f"convert ok in {time.time() - t0:.1f}s")

    cfg = agent.cfg if hasattr(agent, "cfg") else {}
    ml.author = "JevCoreML"
    ml.short_description = (f"laya {args.checkpoint}, L={'/'.join(map(str, lengths))}, "
                            f"K={args.options}, {args.precision}")
    ml.user_defined_metadata.update({
        "laya.checkpoint": args.checkpoint,
        "laya.encoder": str(cfg.get("encoder", "")),
        "laya.sequence_length": str(length),
        # Alle Längen, die das Paket annimmt, zum Nachlesen. Die Laufzeit liest sie aus den
        # aufgezählten Formen der Eingabe selbst und nimmt die kleinste, in die eine Frage passt.
        "laya.sequence_lengths": json.dumps(lengths),
        "laya.max_options": str(args.options),
        "laya.head_max_len": str(spec["head_max_len"]),
        "laya.pad_id": str(tok.pad_token_id),
        "laya.mask_id": str(tok.mask_token_id),
        # Der Text, nicht nur die ID: build_sequence ersetzt ihn in Anweisung, Optionen und
        # Zustand durch ein Leerzeichen. "[MASK]" bei ModernBERT, "<mask>" bei mmBERT. Wer das
        # fest verdrahtet, schleust auf dem falschen Checkpoint ein Markertoken in den Zustand.
        "laya.mask_token": str(tok.mask_token),
        "laya.cls_id": str(tok.cls_token_id),
        "laya.sep_id": str(tok.sep_token_id),
        "laya.precision": args.precision,
        # agent.temperature, nicht model.temperature: laya nimmt den Rückfallwert aus
        # cfg["temperature"] (agent.py). Der gleichnamige Puffer im Modell steht bei allen drei
        # Checkpoints auf [1, 1, 1] und wird von laya nie gelesen. Beim englischen Checkpoint
        # sind es [1,64, 1,25, 1,98], und jede score-Frage mit 2 oder mehr als 5 Stufen fällt
        # auf diesen Wert zurück.
        "laya.temperature": json.dumps([round(float(x), 6) for x in agent.temperature]),
        # Die Länge, auf die laya kürzt. Sie kann kleiner sein als L, auf das gepolstert wird.
        "laya.max_len": str(int(agent.cfg.get("max_len", spec["max_len"]))),
        "laya.temperature_by_options": json.dumps(cfg.get("temperature_by_options", {})),
    })

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(out))
    print(f"gespeichert: {out}")

    # Gegen alle drei Backends pruefen, nicht nur gegen eines. Die Rechenwege unterscheiden
    # sich, und ein Backend, das still falsche Zahlen liefert, faellt sonst niemandem auf.
    feed = {k: v.numpy() for k, v in example.items()}
    k = len(markers)
    for label, unit in [("CPU", ct.ComputeUnit.CPU_ONLY), ("GPU", ct.ComputeUnit.CPU_AND_GPU),
                        ("ANE", ct.ComputeUnit.CPU_AND_NE)]:
        got = ct.models.MLModel(str(out), compute_units=unit).predict(feed)
        dl = float(np.max(np.abs(np.asarray(got["logits"]).reshape(-1)[:k] - reference[:k])))
        da = float(np.max(np.abs(np.asarray(got["act_logits"]).reshape(-1) - reference_act)))
        print(f"Core ML vs laya ({label}) max|dlogit| = {dl:.3e}  max|dact| = {da:.3e}")

    # Jede kürzere Länge auf der GPU gegen dieselbe Referenz. Die Frage ist 50 Token lang und
    # passt in jede; weicht eine Länge ab, stimmt die symbolische Länge im Graphen nicht.
    gpu = ct.models.MLModel(str(out), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    for n in lengths[:-1]:
        short = layax.model_inputs(ids, markers, layax.QTYPES["choice"], n, args.options,
                                   tok.pad_token_id)
        got = gpu.predict({key: v.numpy() for key, v in short.items()})
        dl = float(np.max(np.abs(np.asarray(got["logits"]).reshape(-1)[:k] - reference[:k])))
        print(f"Core ML vs laya (GPU, L={n}) max|dlogit| = {dl:.3e}")


if __name__ == "__main__":
    main()
