#!/usr/bin/env python3
"""Kev (LoRA in die Basis gefaltet) -> Core-ML-.mlpackage.

Vertrag des exportierten Modells, Batch 1, feste Shapes:
  input_ids      [1, L]        int32
  position_ids   [1, L]        int32
  attention_mask [1, 1, L, L]  float32, additiv
  decide_index   [1]           int32
  option_indices [K]           int32
  -> logits      [K]           float32

Erst fp32 exportieren und gegen PyTorch prüfen, danach fp16 als zweites Artefakt.
Sonst vermischt sich Konvertierungsfehler mit Präzisionsverlust.
"""
import argparse
import pathlib
import time

import numpy as np
import torch

import jevx


ORDER = ["input_ids", "position_ids", "attention_mask", "decide_index", "option_indices"]


def exported_program(module, example):
    """torch.export statt torch.jit.trace.

    jit.trace scheitert bei transformers 4.57 an aten::Int auf mehrdimensionalen Shape-Werten
    in Qwen3Attention. torch.export bildet Shapes symbolisch ab und kommt sauber durch;
    run_decompositions() senkt den TRAINING-Dialekt auf ATEN, den coremltools erwartet.
    """
    args = tuple(example[k] for k in ORDER)
    with torch.no_grad():
        return torch.export.export(module, args).run_decompositions({}), args


def rename_output(ml, name):
    """torch.export vergibt generierte Ausgabenamen; der Swift-Vertrag erwartet "logits"."""
    import coremltools as ct

    current = ml.get_spec().description.output[0].name
    if current == name:
        return ml
    spec = ml.get_spec()
    ct.utils.rename_feature(spec, current, name)
    renamed = ct.models.MLModel(spec, weights_dir=ml.weights_dir)
    print(f'Ausgabe "{current}" -> "{name}"')
    return renamed


def main():
    jevx.quiet()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=jevx.DEFAULT_RUN)
    ap.add_argument("--out", default=str(jevx.ROOT / "Models" / "Kev06B.mlpackage"))
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--options", type=int, default=8)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    args = ap.parse_args()

    import coremltools as ct

    tok, kev = jevx.load_kev(args.run)
    module = jevx.OneQuestionKev(kev).eval()
    L, K = args.length, args.options

    # Traceeingabe: echtes Encoding aus dem Golden-Satz, nicht Nullen. Shapes bestimmen den
    # Graphen, aber echte Werte machen den anschliessenden Vergleich aussagekräftig.
    golden = jevx.read_json(jevx.ROOT / "Golden" / "records.json")["records"][0]
    enc = {
        "ids": golden["encoding"]["ids"],
        "pos": golden["encoding"]["position_ids"],
        "decide_idx": [golden["encoding"]["decide_index"]],
        "opt_idx": [golden["encoding"]["option_indices"]],
    }
    example = jevx.model_inputs(enc, L, K, jevx.pad_id_of(tok))

    print(f"torch.export: L={L} K={K} ...")
    t0 = time.time()
    with torch.no_grad():
        reference = module(*(example[k] for k in ORDER)).numpy()
    program, _ = exported_program(module, example)
    print(f"export ok in {time.time() - t0:.1f}s")

    precision = {"fp32": ct.precision.FLOAT32, "fp16": ct.precision.FLOAT16}[args.precision]
    print(f"convert ({args.precision}) ...")
    t0 = time.time()
    ml = ct.convert(
        program,
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS15,
    )
    ml = rename_output(ml, "logits")
    print(f"convert ok in {time.time() - t0:.1f}s")

    ml.author = "JevCoreML"
    ml.short_description = (
        f"Kev-Entscheidungsmodell ({args.run}), eine Frage pro Inferenz, L={L}, K={K}, {args.precision}"
    )
    ml.user_defined_metadata.update({
        "kev.run": args.run,
        "kev.sequence_length": str(L),
        "kev.max_options": str(K),
        "kev.pad_id": str(jevx.pad_id_of(tok)),
        "kev.mask_neg": str(jevx.MASK_NEG),
        "kev.precision": args.precision,
        **{f"kev.env.{k}": v for k, v in getattr(kev, "kev_environment", {}).items() if v},
    })

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(out))
    print(f"gespeichert: {out}")

    # Sofortkontrolle auf der CPU, bevor irgendein Swift-Code im Spiel ist.
    try:
        loaded = ct.models.MLModel(str(out), compute_units=ct.ComputeUnit.CPU_ONLY)
        got = np.asarray(loaded.predict({k: v.numpy() for k, v in example.items()})["logits"]).reshape(-1)
        d = float(np.max(np.abs(got - reference)))
        print(f"Core ML vs PyTorch (CPU_ONLY) max|dlogit| = {d:.3e}")
    except Exception as exc:
        print(f"Direkte Prüfung nicht möglich: {exc}")


if __name__ == "__main__":
    main()
