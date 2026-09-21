"""Gemeinsame Bausteine fuer den laya-Port nach Core ML.

laya ist ein Encoder, kein Decoder: ein bidirektionaler Durchlauf, keine kausale Maske, keine
Generierung. Der Aufbau der Sequenz ist

    [CLS] "<typ> question: <anweisung>" [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] zustand [SEP]

und gelesen wird an den [MASK]-Positionen. Das entspricht Kevs Pointer-Readout, nur dass hier
ein kleiner MLP je Markerposition einen Logit erzeugt statt eines Skalarprodukts.
"""
import json
import os
import pathlib

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parent.parent
MASK_NEG = -1.0e4
QTYPES = {"choice": 0, "score": 1, "noul": 2}

CHECKPOINTS = {
    "english": {"repo": "convaiinnovations/laya", "subfolder": "", "max_len": 512, "head_max_len": 192},
    "typed-decisions": {"repo": "convaiinnovations/laya", "subfolder": "typed-decisions",
                        "max_len": 1024, "head_max_len": 256},
    "multilingual": {"repo": "convaiinnovations/laya", "subfolder": "multilingual",
                     "max_len": 1024, "head_max_len": 256},
}


def patch_mask_constant(encoder):
    """ModernBERT fuellt seine Aufmerksamkeitsmaske mit `torch.finfo(float32).min`.

    In fp32 ist das harmlos, in fp16 wird daraus `-inf`. Eine Zeile, in der jeder Schluessel
    maskiert ist (jede Auffuellposition als Anfrage), softmaxt dann zu NaN, und weil die
    Aufmerksamkeit anschliessend `0 * NaN` rechnet, wandert das NaN in jede Position. Genau das
    war der Befund: max|dlogit| = nan bei beiden fp16-Exporten.

    Mit einem endlichen Wert rechnet dasselbe Modell, denn exp(-1e4) unterlaeuft exakt zu 0.
    Gibt die Funktion zum Zuruecksetzen zurueck, damit der Vergleich gegen das unveraenderte
    laya laufen kann.
    """
    import types

    original = encoder._update_attention_mask
    half = encoder.config.local_attention // 2

    def replacement(self, attention_mask, output_attentions=False):
        length = attention_mask.shape[1]
        valid = attention_mask[:, None, None, :].to(torch.float32)
        glob = (1.0 - valid) * MASK_NEG                            # [B, 1, 1, L]
        glob = glob.expand(-1, -1, length, -1)                     # [B, 1, L, L]
        rows = torch.arange(length).unsqueeze(0)
        outside = (torch.abs(rows - rows.T) > half).unsqueeze(0).unsqueeze(0)
        sliding = glob + outside.to(torch.float32) * MASK_NEG
        # Die Diagonale bleibt offen. Sonst gibt es Zeilen, in denen jeder Schluessel
        # maskiert ist: eine Auffuellposition, deren gleitendes Fenster nur aus Auffuellung
        # besteht. Softmax rechnet dort 0/0, und das NaN wandert ueber den Residualstrom in
        # jede Ausgabe. Fuer gueltige Anfragen aendert das nichts, deren Diagonale steht
        # ohnehin auf 0; fuer Auffuellzeilen zaehlt das Ergebnis nirgends, denn als Schluessel
        # sind sie ueberall maskiert. Erste betroffene Schicht war 1, die erste mit Fenster.
        # Die Einheitsmatrix aus einem Vergleich statt aus torch.eye: eye braucht in Core ML
        # eine feste Größe, und das Paket nimmt mehrere Längen an.
        keep = 1.0 - (rows == rows.T).to(torch.float32)[None, None, :, :]
        return glob * keep, sliding * keep

    encoder._update_attention_mask = types.MethodType(replacement, encoder)
    return lambda: setattr(encoder, "_update_attention_mask", original)


def patch_rotary_tables(encoder, length: int):
    """Rotary-Tabellen einmal in fp32 vorrechnen, im Graphen nur noch die ersten L Zeilen nehmen.

    ModernBERT rechnet cos und sin der Positionswinkel in jeder Aufmerksamkeitsschicht neu aus
    den Positionen. Bei fester Länge sind die Positionen eine Konstante, coremltools faltet die
    ganze Rechnung beim Übersetzen in fp32 und legt nur das Ergebnis ab. Bei mehreren Längen
    geht das nicht mehr, dann läuft der Winkel Position mal Frequenz zur Laufzeit in fp16, und
    fp16 löst einen Winkel um 1000 nur auf 0,5 genau auf. Gemessen: typed-decisions wich danach
    um 7,9e-02 in den Wahrscheinlichkeiten ab statt um 3,9e-03, multilingual um 0,98 in den
    Logits statt um 0,10.

    Die Tabelle gilt für position_ids = arange(L), und genau die setzt ModernBERT, wenn keine
    übergeben werden; der Export übergibt keine. Gibt die Funktion zum Zurücksetzen zurück.
    """
    import types
    from transformers.models.modernbert.modeling_modernbert import ModernBertRotaryEmbedding

    restores = []
    for module in encoder.modules():
        if not isinstance(module, ModernBertRotaryEmbedding):
            continue
        with torch.no_grad():
            positions = torch.arange(length, dtype=torch.float32)[None, :]
            probe = torch.zeros(1, dtype=torch.float32)
            cos, sin = module.forward(probe, positions)               # [1, L, dim], fp32
        module.register_buffer("cos_table", cos.float(), persistent=False)
        module.register_buffer("sin_table", sin.float(), persistent=False)
        original = module.forward

        def forward(self, x, position_ids):
            n = position_ids.shape[-1]
            return self.cos_table[:, :n].to(x.dtype), self.sin_table[:, :n].to(x.dtype)

        module.forward = types.MethodType(forward, module)
        restores.append((module, original))

    def restore():
        for module, original in restores:
            module.forward = original
    return restore


class LayaExport(nn.Module):
    """Encoder, Kopf, Bewerter und Handlungskopf als ein tracebarer Graph.

    Draussen bleibt nur, was Konfiguration ist und keine Gewichte hat: die Temperatur je
    Fragetyp und Optionszahl, die Entropiekonfidenz und das Auspacken in die Antwortstruktur.
    Alles andere rechnet der Graph, damit Swift kein Stueck des Modells nachbaut.
    """

    def __init__(self, model, options: int, length: int = 0):
        super().__init__()
        self.encoder = model.encoder
        self.type_emb = model.type_emb
        self.head = model.head
        self.scorer = model.scorer
        self.act_head = model.act_head
        self.options = options
        undo_mask = patch_mask_constant(self.encoder)
        # Nur nötig, wenn das Paket mehrere Längen annimmt; bei fester Länge faltet coremltools
        # die Rechnung ohnehin. Schaden tut es dort nicht, die Tabelle ist dieselbe Zahl.
        undo_rotary = patch_rotary_tables(self.encoder, length) if length else (lambda: None)

        def restore():
            undo_rotary()
            undo_mask()
        self.restore_mask = restore

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        mask = attention_mask.long()
        h = self.encoder(input_ids=input_ids.long(), attention_mask=mask).last_hidden_state
        h = h + torch.index_select(self.type_emb.weight, 0, qtype.long())[:, None, :]
        if self.head is not None:
            # Nicht src_key_padding_mask: die boolesche Variante wird beim Uebersetzen nach
            # Core ML zu -inf, und das Ergebnis weicht dann um zehn Prozent ab (gemessen:
            # Encoder allein relativ 5,7e-06, mit Kopf 1,0e-01). Eine additive Maske mit
            # endlichem Wert rechnet dasselbe, denn exp(-1e4) unterlaeuft exakt zu 0.
            valid = mask[0].to(h.dtype)
            additive = (1.0 - valid)[None, :] * MASK_NEG          # [1, L], je Schluessel
            additive = additive.expand(mask.shape[1], -1)         # [L, L]
            for layer in self.head.layers:
                h = layer(h, src_mask=additive)

        picked = torch.index_select(h[0], 0, marker_pos.long())   # [K, d]
        raw = self.scorer(picked).reshape(1, self.options).float()
        keep = marker_mask.reshape(1, self.options).to(raw.dtype)
        logits = torch.where(keep > 0, raw, torch.full_like(raw, MASK_NEG))

        # Merkmale des Handlungskopfs, Zeile fuer Zeile wie in laya.common.DecisionModel
        p = torch.softmax(logits, -1)
        k = keep.sum(-1).clamp(min=2.0)
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = torch.topk(p, 2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        act_logits = self.act_head(torch.cat([h[:, 0].float(), feats], -1))
        return logits.reshape(self.options), act_logits.reshape(-1)


def load_laya(name: str, device: str = "cpu"):
    """Laedt einen laya-Checkpoint ueber dessen eigene Bibliothek, damit nichts nachgebaut wird."""
    import laya

    spec = CHECKPOINTS[name]
    agent = laya.load(spec["repo"], device=device, subfolder=spec["subfolder"] or None)
    return agent


def model_inputs(ids, markers, qtype: int, length: int, options: int, pad_id: int):
    """Encoding -> die vier Tensoren des Core-ML-Vertrags."""
    n = len(ids)
    if n > length:
        raise ValueError(f"Sequenz laenger als das Shape-Budget: {n} > {length}")
    if len(markers) > options:
        raise ValueError(f"mehr Optionen als exportiert: {len(markers)} > {options}")
    input_ids = torch.full((1, length), pad_id, dtype=torch.int32)
    input_ids[0, :n] = torch.tensor(ids, dtype=torch.int32)
    attention = torch.zeros((1, length), dtype=torch.int32)
    attention[0, :n] = 1
    pos = torch.zeros((options,), dtype=torch.int32)
    pos[: len(markers)] = torch.tensor(markers, dtype=torch.int32)
    keep = torch.zeros((options,), dtype=torch.int32)
    keep[: len(markers)] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "marker_pos": pos,
        "marker_mask": keep,
        "qtype": torch.tensor([qtype], dtype=torch.int32),
    }


def write_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def quiet():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_grad_enabled(False)
