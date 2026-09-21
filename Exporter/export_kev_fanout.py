#!/usr/bin/env python3
"""Phase 2: Kev mit mehreren Fragen in einer Sequenz nach Core ML.

Vertrag, Batch 1:
  input_ids      [1, L]   int32
  position_ids   [1, L]   int32
  segment_ids    [1, L]   int32   0 = Zustand, k = Frage k, -1 = Padding
  decide_index   [Q]      int32
  option_indices [Q, K]   int32
  -> logits      [Q, K]   float32

Die block-kausale Maske entsteht im Graphen aus `segment_ids`. Gegenüber Phase 1 fällt damit
eine Eingabe von einem Megabyte je Aufruf weg, und die Maskenregel steht nur noch an einer Stelle.

Mit --lengths trägt ein Paket mehrere aufgezählte Längen (EnumeratedShapes auf den drei
Sequenzeingaben); Q und K bleiben fest. Die Laufzeit rechnet jede Anfrage in der kürzesten
Länge, in die sie passt, mit einer Instanz je Länge, so wie beim laya-Port. Fragen und Optionen
wirken nur im Pointer-Kopf, deshalb dürfen sie großzügig sein, ohne die Latenz zu berühren.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

import jevx

ORDER = ["input_ids", "position_ids", "segment_ids", "decide_index", "option_indices"]


def main():
    jevx.quiet()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=jevx.DEFAULT_RUN)
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--lengths", default="",
                    help="aufgezählte Längen, etwa 128,256,512,1024,2048,3072; die größte ist L")
    ap.add_argument("--default-length", type=int, default=0,
                    help="Vorgabeform des Pakets bei --lengths (0 = 512, falls dabei, sonst die größte)")
    ap.add_argument("--questions", type=int, default=4)
    ap.add_argument("--options", type=int, default=8)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--name", default="",
                    help="eigener Name des Pakets in Beschreibung und Metadaten, etwa JevCoreML; "
                         "die Herkunft (kev-Run, Basis) steht daneben")
    args = ap.parse_args()

    import coremltools as ct

    if args.lengths:
        lengths = sorted({int(x) for x in args.lengths.split(",")})
        if lengths[0] < 16:
            raise SystemExit(f"--lengths {lengths} darf nicht unter 16 beginnen")
        L = lengths[-1]
    else:
        lengths = [args.length]
        L = args.length
    default_length = args.default_length or (512 if 512 in lengths else L)
    if default_length not in lengths:
        raise SystemExit(f"--default-length {default_length} ist nicht unter {lengths}")
    Q, K = args.questions, args.options

    tok, kev = jevx.load_kev(args.run)
    module = jevx.MultiQuestionKev(kev, L, Q, K).eval()
    pad_id = jevx.pad_id_of(tok)

    # Traceeingabe ist der gepackte Golden-Lauf mit drei Fragen (192 Token), nicht ein
    # Nullvektor. Für Längen, in die er nicht passt, prüft die kürzeste Einzelanfrage (26 Token).
    packed = jevx.read_json(jevx.ROOT / "Golden" / "packed.json")["records"][0]
    records = jevx.read_json(jevx.ROOT / "Golden" / "records.json")["records"]
    enc = jevx.packed_enc(packed)
    shortest = jevx.single_enc(min(records, key=lambda r: len(r["encoding"]["ids"])), jevx.q_token_id(packed))
    if len(enc["ids"]) > default_length:
        raise SystemExit(f"die Traceeingabe hat {len(enc['ids'])} Token und passt nicht in die "
                         f"Vorgabelänge {default_length}")
    example = jevx.multi_inputs(enc, default_length, Q, K, pad_id)

    print(f"torch.export: L={lengths} Q={Q} K={K} ...")
    t0 = time.time()
    # Eine gemeinsame symbolische Länge für die drei Sequenzeingaben. Maske, Kausalität und
    # Diagonale entstehen im Graphen aus dieser Länge, nicht aus Konstanten für L.
    dynamic = None
    if len(lengths) > 1:
        seq = torch.export.Dim("seq", min=lengths[0], max=L)
        dynamic = {"input_ids": {1: seq}, "position_ids": {1: seq}, "segment_ids": {1: seq},
                   "decide_index": None, "option_indices": None}
    with torch.no_grad():
        reference = module(*(example[k] for k in ORDER)).numpy()
        short_example = jevx.multi_inputs(shortest, default_length, Q, K, pad_id)
        short_reference = module(*(short_example[k] for k in ORDER)).numpy()
        program = torch.export.export(module, tuple(example[k] for k in ORDER),
                                      dynamic_shapes=dynamic).run_decompositions({})
    print(f"export ok in {time.time() - t0:.1f}s")

    precision = {"fp32": ct.precision.FLOAT32, "fp16": ct.precision.FLOAT16}[args.precision]
    print(f"convert ({args.precision}) ...")
    t0 = time.time()
    inputs = None
    if len(lengths) > 1:
        # Seit macOS 15 dürfen mehrere Eingaben aufgezählte Formen tragen, je gleich viele.
        def enumerated():
            return ct.EnumeratedShapes(shapes=[[1, n] for n in lengths], default=[1, default_length])
        inputs = [ct.TensorType(name="input_ids", shape=enumerated(), dtype=np.int32),
                  ct.TensorType(name="position_ids", shape=enumerated(), dtype=np.int32),
                  ct.TensorType(name="segment_ids", shape=enumerated(), dtype=np.int32),
                  ct.TensorType(name="decide_index", shape=(Q,), dtype=np.int32),
                  ct.TensorType(name="option_indices", shape=(Q, K), dtype=np.int32)]
    ml = ct.convert(program, convert_to="mlprogram", compute_precision=precision, inputs=inputs,
                    outputs=[ct.TensorType(name="logits")],
                    minimum_deployment_target=ct.target.macOS15)
    print(f"convert ok in {time.time() - t0:.1f}s")

    ml.author = "JevCoreML"
    shape = f"{Q} Fragen pro Inferenz, L={'/'.join(map(str, lengths))}, K={K}, {args.precision}"
    # Urheber und Basis nur für den bekannten Run; ein anderer --run bekommt nur seinen Namen.
    known = args.run == jevx.DEFAULT_RUN
    origin = (f"{args.run} (Jared Palmer, Apache 2.0) auf Qwen/Qwen3-0.6B-Base (Apache 2.0)"
              if known else args.run)
    described = f"kev-0.6b ({args.run}, Basis Qwen3-0.6B-Base)" if known else args.run
    ml.short_description = (
        f"{args.name}: Entscheidungsmodell auf {described}, {shape}"
        if args.name else f"Kev-Entscheidungsmodell ({args.run}), {shape}")
    ml.user_defined_metadata.update({
        # Eigener Name des Pakets und seine Herkunft, damit beides im Paket selbst steht.
        **({"jev.name": args.name, "jev.based_on": origin} if args.name else {}),
        "kev.run": args.run, "kev.phase": "2", "kev.sequence_length": str(L),
        # Alle Längen, die das Paket annimmt, zum Nachlesen. Die Laufzeit liest sie aus den
        # aufgezählten Formen der Eingabe selbst.
        "kev.sequence_lengths": json.dumps(lengths),
        "kev.default_length": str(default_length),
        "kev.max_questions": str(Q), "kev.max_options": str(K),
        "kev.pad_id": str(pad_id), "kev.mask_neg": str(jevx.MASK_NEG),
        "kev.precision": args.precision,
        **{f"kev.env.{k}": v for k, v in getattr(kev, "kev_environment", {}).items() if v},
    })

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(out))
    print(f"gespeichert: {out}")

    # Jede Länge gegen dieselbe PyTorch-Referenz, auf CPU und GPU. Die Auffüllung ist
    # ausmaskiert, also muss jede Länge dasselbe ergeben; weicht eine ab, stimmt die
    # symbolische Länge im Graphen nicht. Eine frische Instanz je Länge, wie in der Laufzeit.
    compiled = ct.models.utils.compile_model(str(out))
    for n in lengths:
        if len(enc["ids"]) <= n:
            case, expected, label_case = enc, reference, "3 Fragen"
        else:
            case, expected, label_case = shortest, short_reference, "1 Frage"
        feed = {k: v.numpy() for k, v in jevx.multi_inputs(case, n, Q, K, pad_id).items()}
        for label, unit in [("CPU", ct.ComputeUnit.CPU_ONLY), ("GPU", ct.ComputeUnit.CPU_AND_GPU)]:
            model = ct.models.CompiledMLModel(compiled, compute_units=unit)
            got = np.asarray(model.predict(feed)["logits"])
            delta = float(np.max(np.abs(got - expected)))
            print(f"Core ML vs PyTorch ({label}, L={n}, {label_case}) max|dlogit| = {delta:.3e}")
            del model


if __name__ == "__main__":
    main()
