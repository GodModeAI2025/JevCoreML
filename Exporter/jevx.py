"""Gemeinsame Bausteine für Export, Golden-Dump und Paritätsprüfung.

Phase 1: genau eine Frage pro Inferenz. Damit fällt Kevs block-kausale Branch-Maske
auf eine gewöhnliche kausale Maske zusammen (in tests/test_phase1_assumptions nachgemessen),
und die Positionen sind schlicht 0..L-1. Mehrere Fragen laufen in Swift als getrennte Aufrufe.
"""
import json
import os
import pathlib

import torch
import torch.nn as nn

# Wert für maskierte Attention-Felder. fp16-sicher (finfo(float32).min würde auf der
# ANE zu -inf und damit zu NaN werden); exp(-1e4) unterläuft in fp32 wie in fp16 exakt zu 0,
# das Ergebnis ist also identisch zu PyTorchs finfo.min.
MASK_NEG = -1.0e4

DEFAULT_RUN = "jaredpalmer/kev-0.6b"
ROOT = pathlib.Path(__file__).resolve().parent.parent


class OneQuestionKev(nn.Module):
    """Backbone + Pointer-Head als ein einziger tracebarer Graph.

    Eingaben sind bewusst alle explizit: die Maske und die Readout-Indizes kommen von aussen,
    damit der Swift-Encoder und der Python-Encoder denselben Vertrag erfüllen und Phase 2
    (mehrere Fragen in einer Sequenz) ohne Re-Export möglich bleibt.
    """

    def __init__(self, kev):
        super().__init__()
        self.lm = kev.lm
        self.head = kev.head

    def forward(self, input_ids, position_ids, attention_mask, decide_index, option_indices):
        h = self.lm(
            input_ids=input_ids.long(),
            position_ids=position_ids.long(),
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state.float()[0]
        h_decide = torch.index_select(h, 0, decide_index.long())     # [1, d]
        h_opts = torch.index_select(h, 0, option_indices.long())     # [K, d]
        # Rechnerisch identisch zu PointerHead.forward, aber beide Matmul-Operanden bleiben
        # zweidimensional: Core MLs matmul akzeptiert keine 1-D-Operanden.
        q = self.head.q(h_decide)                                    # [1, dp]
        k = self.head.k(h_opts)                                      # [K, dp]
        return torch.matmul(k, q.transpose(0, 1)).squeeze(-1) * self.head.scale


def load_kev(run=DEFAULT_RUN, device="cpu"):
    """LoRA in fp32 in die Basisgewichte gefaltet, laut kev-Modellkarte exakt.

    dtype und attn werden ausdruecklich uebergeben und neutralisieren damit KEV_DTYPE und
    KEV_ATTN. KEV_MERGE und KEV_LORA_SCALE wirken dagegen weiter in kev.evaluate.load hinein
    und wuerden andere Gewichte exportieren, ohne dass man es dem .mlpackage ansieht.
    Deshalb werden sie hier gelesen und zurueckgegeben, damit der Export sie vermerkt.
    """
    from kev.evaluate import load

    environment = {
        "KEV_MERGE": os.environ.get("KEV_MERGE", "1"),
        "KEV_LORA_SCALE": os.environ.get("KEV_LORA_SCALE", "1"),
        "KEV_DTYPE": os.environ.get("KEV_DTYPE", ""),
        "KEV_ATTN": os.environ.get("KEV_ATTN", ""),
    }
    tok, m = load(run, device, dtype=torch.float32, merge=True, attn="eager")
    m.eval()
    m.kev_environment = environment
    if environment["KEV_MERGE"] != "1" or environment["KEV_LORA_SCALE"] != "1":
        print(f"ACHTUNG: abweichende Umgebung wirkt auf die Gewichte: {environment}")
    return tok, m


def pad_id_of(tok):
    return tok.pad_token_id if tok.pad_token_id is not None else 0


def causal_mask(n, length, dtype=torch.float32):
    """Additive [1,1,L,L] Maske: j<=i innerhalb der n echten Token, sonst gesperrt.

    Padding-Zeilen behalten ihre Diagonale, sonst wäre eine Softmax-Zeile vollständig
    maskiert und produzierte NaN, das über die Keys in echte Token zurückliefe.
    """
    idx = torch.arange(length)
    q, k = idx[:, None], idx[None, :]
    allow = (k <= q) & (k < n) & (q < n)
    allow = allow | torch.eye(length, dtype=torch.bool)
    return torch.zeros(1, 1, length, length, dtype=dtype).masked_fill(~allow, MASK_NEG)


def model_inputs(enc, length, options, pad_id, dtype=torch.float32):
    """Encoding -> die fünf Tensoren des Core-ML-Vertrags (Batch 1, feste Shapes)."""
    ids_list = enc["ids"]
    n = len(ids_list)
    if n > length:
        raise ValueError(f"Sequenz länger als das Shape-Budget: {n} > {length}")
    if len(enc["decide_idx"]) != 1:
        raise ValueError("Phase 1 verarbeitet genau eine Frage pro Inferenz")
    ends = enc["opt_idx"][0]
    if len(ends) > options:
        raise ValueError(f"mehr Optionen als exportiert: {len(ends)} > {options}")

    ids = torch.full((1, length), pad_id, dtype=torch.int32)
    ids[0, :n] = torch.tensor(ids_list, dtype=torch.int32)
    pos = torch.zeros((1, length), dtype=torch.int32)
    pos[0, :n] = torch.tensor(enc["pos"], dtype=torch.int32)
    mask = causal_mask(n, length, dtype=dtype)
    decide = torch.tensor([enc["decide_idx"][0]], dtype=torch.int32)
    opt = torch.full((options,), ends[-1], dtype=torch.int32)
    opt[: len(ends)] = torch.tensor(ends, dtype=torch.int32)
    return {
        "input_ids": ids,
        "position_ids": pos,
        "attention_mask": mask,
        "decide_index": decide,
        "option_indices": opt,
    }


def softmax_list(logits, k):
    z = torch.tensor(logits[:k], dtype=torch.float64)
    return torch.softmax(z, 0).tolist()


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def write_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def quiet():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_grad_enabled(False)


class MultiQuestionKev(nn.Module):
    """Phase 2: mehrere Fragen in einer Sequenz, ein Durchlauf durch den Backbone.

    Gegenüber Phase 1 ändern sich zwei Dinge. Die Maske kommt nicht mehr als Tensor herein,
    sondern wird im Graphen aus `segment_ids` gebaut: bei L=512 spart das eine Million Byte
    Eingabe je Aufruf, und es ist die einzige Stelle, an der die Maskenregel dann noch steht.
    Der Pointer-Head liest Q Entscheidungspositionen statt einer.

    segment_ids: 0 für den Zustand, k für Frage k, -1 für Padding.
    """

    def __init__(self, kev, length, questions, options):
        super().__init__()
        self.lm = kev.lm
        self.head = kev.head
        self.L, self.Q, self.K = length, questions, options

    def build_mask(self, segment_ids, dtype):
        """Entspricht `kev.model.branch_mask_batch` ohne Optionsisolation.

        erlaubt(i,j) = j<=i und seg[j] gültig und (seg[j]==0 oder seg[j]==seg[i]).
        Die Diagonale bleibt immer offen, sonst ergäbe eine vollständig maskierte Zeile NaN.

        Gerechnet wird mit 0/1 statt mit Wahrheitswerten: der Core-ML-Frontend übersetzt
        `__or__` auf Bool-Tensoren nicht.

        Kausalität und Diagonale entstehen aus der tatsächlichen Länge der Eingabe, nicht aus
        Puffern der Form [L,L]: so trägt derselbe Graph mehrere aufgezählte Längen, und bei
        fester Länge faltet der Konverter die Vergleiche zu Konstanten.
        """
        seg = segment_ids[0].to(dtype)
        idx = torch.arange(seg.shape[0], device=seg.device)
        causal = torch.ge(idx[:, None], idx[None, :]).to(dtype)
        eye = torch.eq(idx[:, None], idx[None, :]).to(dtype)
        key, query = seg[None, :], seg[:, None]
        same = torch.eq(key, query).to(dtype)
        is_state = torch.eq(key, torch.zeros((), dtype=dtype)).to(dtype)
        valid = torch.ge(key, torch.zeros((), dtype=dtype)).to(dtype)
        visible = torch.clamp(same + is_state, max=1.0) * valid * causal
        allow = torch.clamp(visible + eye, max=1.0)
        return ((allow - 1.0) * (-MASK_NEG))[None, None]

    def forward(self, input_ids, position_ids, segment_ids, decide_index, option_indices):
        mask = self.build_mask(segment_ids, torch.float32)
        h = self.lm(
            input_ids=input_ids.long(),
            position_ids=position_ids.long(),
            attention_mask=mask,
            use_cache=False,
        ).last_hidden_state.float()[0]
        d = h.shape[-1]
        h_decide = torch.index_select(h, 0, decide_index.long())                       # [Q, d]
        h_opts = torch.index_select(h, 0, option_indices.reshape(-1).long())           # [Q*K, d]
        h_opts = h_opts.reshape(self.Q, self.K, d)
        q = self.head.q(h_decide).unsqueeze(-1)                                        # [Q, dp, 1]
        k = self.head.k(h_opts)                                                        # [Q, K, dp]
        return torch.matmul(k, q).squeeze(-1) * self.head.scale                        # [Q, K]


def q_token_id(packed_record):
    """Die ID des <q>-Tokens: dort, wo im gepackten Golden-Lauf das erste Segment 1 beginnt."""
    enc = packed_record["encoding"]
    return enc["ids"][enc["segment_ids"].index(1)]


def segments_from_ids(ids, q_id):
    """0 für den Zustand, k ab dem k-ten <q>-Token. Ersatz für fehlende segment_ids in den
    Einzelanfragen aus records.json; gegen packed.json geprüft."""
    seg, k = [], 0
    for t in ids:
        if t == q_id:
            k += 1
        seg.append(k)
    return seg


def single_enc(record, q_id):
    """Encoding einer Einzelanfrage aus records.json im Phase-2-Format."""
    e = record["encoding"]
    return {"ids": e["ids"], "pos": e["position_ids"], "seg": segments_from_ids(e["ids"], q_id),
            "decide_idx": [e["decide_index"]], "opt_idx": [e["option_indices"]]}


def packed_enc(record):
    """Encoding eines gepackten Mehrfragenlaufs aus packed.json im Phase-2-Format."""
    e = record["encoding"]
    return {"ids": e["ids"], "pos": e["position_ids"], "seg": e["segment_ids"],
            "decide_idx": [q["decide_index"] for q in record["questions"]],
            "opt_idx": [q["option_indices"] for q in record["questions"]]}


def segment_ids_of(enc, length):
    """seg-Vektor aus einem kev-Encoding, rechts mit -1 aufgefüllt."""
    seg = torch.full((1, length), -1, dtype=torch.int32)
    seg[0, : len(enc["seg"])] = torch.tensor(enc["seg"], dtype=torch.int32)
    return seg


def multi_inputs(enc, length, questions, options, pad_id):
    """Encoding mit beliebig vielen Fragen -> die fünf Tensoren des Phase-2-Vertrags."""
    n = len(enc["ids"])
    if n > length:
        raise ValueError(f"Sequenz länger als das Shape-Budget: {n} > {length}")
    q_count = len(enc["decide_idx"])
    if q_count > questions:
        raise ValueError(f"mehr Fragen als exportiert: {q_count} > {questions}")
    for ends in enc["opt_idx"]:
        if len(ends) > options:
            raise ValueError(f"mehr Optionen als exportiert: {len(ends)} > {options}")

    ids = torch.full((1, length), pad_id, dtype=torch.int32)
    ids[0, :n] = torch.tensor(enc["ids"], dtype=torch.int32)
    pos = torch.zeros((1, length), dtype=torch.int32)
    pos[0, :n] = torch.tensor(enc["pos"], dtype=torch.int32)

    decide = torch.zeros((questions,), dtype=torch.int32)
    opt = torch.zeros((questions, options), dtype=torch.int32)
    for qi in range(questions):
        # Ungenutzte Frageplätze wiederholen Frage 0; ihre Logits werden verworfen.
        src = qi if qi < q_count else 0
        decide[qi] = enc["decide_idx"][src]
        ends = enc["opt_idx"][src]
        for ki in range(options):
            opt[qi, ki] = ends[ki] if ki < len(ends) else ends[-1]

    return {
        "input_ids": ids,
        "position_ids": pos,
        "segment_ids": segment_ids_of(enc, length),
        "decide_index": decide,
        "option_indices": opt,
    }
