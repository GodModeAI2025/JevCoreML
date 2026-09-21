#!/usr/bin/env python3
"""Phase 2: Kev mit mehreren Fragen in einer Sequenz nach Core ML.

Vertrag, Batch 1, feste Shapes:
  input_ids      [1, L]   int32
  position_ids   [1, L]   int32
  segment_ids    [1, L]   int32   0 = Zustand, k = Frage k, -1 = Padding
  decide_index   [Q]      int32
  option_indices [Q, K]   int32
  -> logits      [Q, K]   float32

Die block-kausale Maske entsteht im Graphen aus `segment_ids`. Gegenüber Phase 1 fällt damit
eine Eingabe von einem Megabyte je Aufruf weg, und die Maskenregel steht nur noch an einer Stelle.
"""
import argparse
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
    ap.add_argument("--questions", type=int, default=4)
    ap.add_argument("--options", type=int, default=8)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    args = ap.parse_args()

    import coremltools as ct

    tok, kev = jevx.load_kev(args.run)
    L, Q, K = args.length, args.questions, args.options
    module = jevx.MultiQuestionKev(kev, L, Q, K).eval()

    # Traceeingabe ist der gepackte Golden-Lauf mit drei Fragen, nicht ein Nullvektor.
    packed = jevx.read_json(jevx.ROOT / "Golden" / "packed.json")["records"][0]
    enc = {
        "ids": packed["encoding"]["ids"],
        "pos": packed["encoding"]["position_ids"],
        "seg": packed["encoding"]["segment_ids"],
        "decide_idx": [q["decide_index"] for q in packed["questions"]],
        "opt_idx": [q["option_indices"] for q in packed["questions"]],
    }
    example = jevx.multi_inputs(enc, L, Q, K, jevx.pad_id_of(tok))

    print(f"torch.export: L={L} Q={Q} K={K} ...")
    t0 = time.time()
    with torch.no_grad():
        reference = module(*(example[k] for k in ORDER)).numpy()
        program = torch.export.export(module, tuple(example[k] for k in ORDER)).run_decompositions({})
    print(f"export ok in {time.time() - t0:.1f}s")

    precision = {"fp32": ct.precision.FLOAT32, "fp16": ct.precision.FLOAT16}[args.precision]
    print(f"convert ({args.precision}) ...")
    t0 = time.time()
    ml = ct.convert(program, convert_to="mlprogram", compute_precision=precision,
                    minimum_deployment_target=ct.target.macOS15)
    current = ml.get_spec().description.output[0].name
    if current != "logits":
        spec = ml.get_spec()
        ct.utils.rename_feature(spec, current, "logits")
        ml = ct.models.MLModel(spec, weights_dir=ml.weights_dir)
    print(f"convert ok in {time.time() - t0:.1f}s")

    ml.author = "JevCoreML"
    ml.short_description = (
        f"Kev-Entscheidungsmodell ({args.run}), {Q} Fragen pro Inferenz, L={L}, K={K}, {args.precision}")
    ml.user_defined_metadata.update({
        "kev.run": args.run, "kev.phase": "2", "kev.sequence_length": str(L),
        "kev.max_questions": str(Q), "kev.max_options": str(K),
        "kev.pad_id": str(jevx.pad_id_of(tok)), "kev.mask_neg": str(jevx.MASK_NEG),
        "kev.precision": args.precision,
        **{f"kev.env.{k}": v for k, v in getattr(kev, "kev_environment", {}).items() if v},
    })

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(out))
    print(f"gespeichert: {out}")

    loaded = ct.models.MLModel(str(out), compute_units=ct.ComputeUnit.CPU_ONLY)
    got = np.asarray(loaded.predict({k: v.numpy() for k, v in example.items()})["logits"])
    print(f"Core ML vs PyTorch (CPU_ONLY) max|dlogit| = {float(np.max(np.abs(got - reference))):.3e}")


if __name__ == "__main__":
    main()
