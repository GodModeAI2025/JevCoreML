#!/usr/bin/env python3
"""Erzeugt die Referenzdaten, gegen die die Swift-Seite geprüft wird.

Drei Dateien in Golden/:
  tokenizer.json  Text -> Token-IDs, inklusive der Fälle, an denen naive BPE-Nachbauten scheitern
  records.json    vollständige Encodings plus PyTorch-Logits/Wahrscheinlichkeiten
  runtime.json    Shape-Budget, Padding-ID, Delimiter-IDs, Maskenwert

Ohne diese Dateien ist jede Swift-Implementierung nur eine Behauptung.
"""
import argparse
import hashlib
import json
import pathlib

import torch

import jevx
from kev.api import SystemOneRequest, to_answers, to_record
from kev.model import SPECIAL, user_tokens

# Fälle, die Parität tatsächlich brechen können: NFC-Zerlegung, exotische Leerzeichen,
# ZWJ-Emoji, CJK, Ziffernketten, CRLF, fuehrende Leerzeichen, Apostroph-Kontraktionen und
# die Added-Token-Falle (<think> ist ein Token, obwohl es nicht dem <|name|>-Muster folgt).
EDGE_CASES = [
    "",
    " ",
    "  ",
    "\n",
    "\r\n",
    "\t",
    "a b",
    "Grüße aus München",
    "Grüße aus München",
    "Straße – Gasse — Weg",
    "ÄÖÜ äöü ß",
    "café café",
    "\U0001f468‍\U0001f469‍\U0001f467",
    "\U0001f600 \U0001f64f \U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
    "你好世界",
    "こんにちは世界",
    "مرحبا بالعالم",
    "Привет мир",
    "0123456789",
    "1234567890123456789",
    "don't you're we've I'll he'd she's it",
    "DON'T YOU'RE WE'VE",
    "   fuehrende Leerzeichen",
    "trailing   ",
    "a\n\n\nb",
    "line1\r\nline2\r\n",
    "<think>",
    "</think>",
    "<tool_call>",
    "<|endoftext|>",
    "<|im_start|>user",
    "<|fim_prefix|>",
    "<|box_start|>x<|box_end|>",
    "<¦fim_prefix¦>",
    "<|not_a_real_token|>",
    "<|fim_prefix|> mitten im Satz <|fim_suffix|>",
    "{\"json\": [1, 2, {\"nested\": true}]}",
    "def f(x): return x ** 2  # Kommentar",
    "https://example.com/path?query=1&x=2#frag",
    "E-Mail: kunde@example.com, Tel. +49 170 1234567",
    "Der Kunde wurde zweimal für dieselbe Bestellung belastet.",
    "The customer received two identical charges for the same order.",
    "Rechnung 2024-11 über 1.234,56 € wurde storniert.",
    "a" * 200,
    "Wort " * 60,
    "​‌‍",
    "﻿BOM am Anfang",
    "\U0001d539\U0001d554\U0001d55c\U0001d55b",
    "½ ⅓ ² Ω µ",
    # Whitespace-Klassen, NFC gegen NFKC und Komposition an Added-Token-Grenzen
    "a\x0bb",
    "a\x85b",
    "a\u2028b\u2029c",
    "a\u1680b",
    "a\u3000b",
    "a\u205fb",
    "a\u202fb",
    "\u212b \u2126",
    "\ufb01n",
    "\u1100\u1161",
    "a\u0301\u0327b",
    "<think>\u0301",
    "e\u0301<think>e\u0301",
    "<|box_start|>\u0301x<|box_end|>",
    "\U0001f1e9\U0001f1ea \U0001f3f3\ufe0f",
    # Nicht-BMP-Starter mit folgender Marke: hier kürzt Foundations NFC auf 16 Bit
    '\U000200c5\u0301',
    '\U00010415\u0301',
    '\U00020000\u0300',
    'Zeichen \U000200c5\u0301 Ende',
    '\U0001d160',
    '\U0001d158\U0001d165',
    '\U0001f600\u0301',
    '\u1100\u1161\u11a8',
]

# Vollständige Anfragen über alle drei Fragetypen.
REQUESTS = [
    {
        "state": "The customer received two identical charges for the same order.",
        "questions": {"billing_problem": {"type": "noul", "instructions": "Is there a billing problem?"}},
    },
    {
        "state": "Der Kunde meldet, dass die App beim Start abstuerzt. Er nutzt ein iPhone 15 mit iOS 18.",
        "questions": {"is_bug": {"type": "noul", "instructions": "Beschreibt der Zustand einen technischen Fehler?"}},
    },
    {
        "state": "Erstelle mir bitte eine Tabelle mit den Umsaetzen der letzten zwoelf Monate inklusive Quartalssummen.",
        "questions": {
            "skill": {
                "type": "choice",
                "instructions": "Welcher Skill soll diese Aufgabe uebernehmen?",
                "criteria": {
                    "research": "Quellen recherchieren und zusammenfassen",
                    "presentation": "Folien und Praesentationen bauen",
                    "excel": "Tabellen, Kennzahlen und Berechnungen",
                    "coding": "Software schreiben oder aendern",
                    "none": "keiner dieser Skills passt",
                },
            }
        },
    },
    {
        "state": "Ticket: 'Ihr Produkt ist voellig unbrauchbar, ich will mein Geld zurueck und zwar sofort.'",
        "questions": {
            "severity": {
                "type": "score",
                "instructions": "Wie stark ist die Eskalation?",
                "criteria": ["neutral", "leicht gereizt", "veraergert", "stark veraergert", "eskaliert"],
            }
        },
    },
    {
        "state": "Support-Chat: Kunde fragt nach der Lieferzeit einer bereits versandten Bestellung.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "An welches Team geht das Ticket?",
                "criteria": {"billing": None, "logistics": None, "technical": None, "none": None},
            }
        },
    },
    {
        "state": {
            "kunde": {"plan": "Enterprise", "seit": "2021"},
            "vorfall": ["Doppelbuchung", "zwei identische Belastungen"],
        },
        "questions": {"refund": {"type": "noul", "instructions": "Ist eine Rueckerstattung angebracht?",
                                  "criteria": {"true": "Rueckerstattung ist angebracht", "false": "keine Rueckerstattung"}}},
    },
]


# Gepackte Anfragen: mehrere Fragen in einer Sequenz, so wie kev sie im Original fährt.
# Die Swift-Seite beantwortet sie als getrennte Aufrufe; dieser Record belegt, dass dabei
# dasselbe herauskommt, statt es nur aus der Maskenform herzuleiten.
PACKED_REQUESTS = [
    {
        "state": {
            "kunde": {"plan": "Enterprise", "seit": "2021"},
            "ticket": "Ihr Produkt ist voellig unbrauchbar, ich will mein Geld zurueck und zwar sofort. "
                      "Die App stuerzt seit dem Update beim Start ab.",
            "historie": ["zwei identische Belastungen im Maerz", "Rueckerstattung damals abgelehnt"],
        },
        "questions": {
            "is_bug": {"type": "noul", "instructions": "Beschreibt der Zustand einen technischen Fehler?"},
            "team": {
                "type": "choice",
                "instructions": "An welches Team geht das Ticket?",
                "criteria": {
                    "billing": "Rechnungen, Zahlungen, Rueckerstattungen",
                    "logistics": "Versand und Lieferung",
                    "technical": "Fehler in Produkt oder App",
                    "none": "keines dieser Teams",
                },
            },
            "eskalation": {
                "type": "score",
                "instructions": "Wie stark ist die Eskalation?",
                "criteria": ["neutral", "leicht gereizt", "veraergert", "stark veraergert", "eskaliert"],
            },
        },
    },
    {
        "state": "The customer received two identical charges for the same order and has not been refunded.",
        "questions": {
            "billing_problem": {"type": "noul", "instructions": "Is there a billing problem?"},
            "urgent": {"type": "noul", "instructions": "Does this need attention today?"},
        },
    },
]


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    jevx.quiet()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=jevx.DEFAULT_RUN)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--options", type=int, default=8)
    ap.add_argument("--questions", type=int, default=4)
    ap.add_argument("--out", default=str(jevx.ROOT / "Golden"))
    args = ap.parse_args()

    tok, model = jevx.load_kev(args.run)
    from kev.evaluate import resolve_run

    run_dir = pathlib.Path(resolve_run(args.run))
    out = pathlib.Path(args.out)

    special_ids = {t: int(tok.convert_tokens_to_ids(t)) for t in SPECIAL}
    added = {a.content: int(i) for i, a in tok.added_tokens_decoder.items()}

    jevx.write_json(out / "runtime.json", {
        "run": args.run,
        "base": "Qwen/Qwen3-0.6B-Base",
        "sequence_length": args.length,
        "max_options": args.options,
        "max_state_tokens": 384,
        "pad_id": jevx.pad_id_of(tok),
        "mask_neg": jevx.MASK_NEG,
        "delimiters": {name: special_ids[t] for name, t in
                       zip(["state", "question", "option", "option_end", "decide"], SPECIAL)},
        "added_tokens": added,
        "tokenizer_sha256": sha256(run_dir / "tokenizer.json"),
    })

    cases = []
    for text in EDGE_CASES:
        cases.append({
            "text": text,
            "ids": [int(i) for i in tok(text, add_special_tokens=False).input_ids],
            "user_ids": [int(i) for i in user_tokens(tok, text)],
        })
    jevx.write_json(out / "tokenizer.json", {"count": len(cases), "cases": cases})

    records = []
    for raw in REQUESTS:
        req = SystemOneRequest.model_validate(raw)
        rec, meta = to_record(req)
        if len(rec["questions"]) != 1:
            raise SystemExit("Phase 1 erlaubt genau eine Frage pro Anfrage")
        enc = model.encode(tok, rec, strict=True)
        probs = [p.tolist() for p in model.probs(enc)]
        logits = model.forward(enc)[0].tolist()
        inputs = jevx.model_inputs(enc, args.length, args.options, jevx.pad_id_of(tok))
        padded = model.head(
            *(lambda h: (h[enc["decide_idx"][0]], h[torch.tensor(enc["opt_idx"][0])]))(
                model.lm(input_ids=inputs["input_ids"].long(), position_ids=inputs["position_ids"].long(),
                         attention_mask=inputs["attention_mask"], use_cache=False).last_hidden_state.float()[0]
            )
        ).tolist()
        q = rec["questions"][0]
        records.append({
            "request": raw,
            "rendered": {"state": rec["state"], "instructions": q["instr"], "options": q["options"]},
            "encoding": {
                "ids": [int(i) for i in enc["ids"]],
                "position_ids": [int(p) for p in enc["pos"]],
                "decide_index": int(enc["decide_idx"][0]),
                "option_indices": [int(i) for i in enc["opt_idx"][0]],
                "length": len(enc["ids"]),
                "state_truncated": bool(enc["state_truncated"]),
            },
            "logits": logits,
            "logits_padded": padded,
            "probabilities": probs[0],
            "answer": to_answers(probs, meta),
        })
        print(f"{list(raw['questions'])[0]:16s} L={len(enc['ids']):4d} "
              f"max|dlogit| padded-vs-tight={max(abs(a - b) for a, b in zip(logits, padded)):.3e}")

    jevx.write_json(out / "records.json", {"count": len(records), "records": records})

    packed = []
    for raw in PACKED_REQUESTS:
        req = SystemOneRequest.model_validate(raw)
        rec, meta = to_record(req)
        enc = model.encode(tok, rec, strict=True)
        probs = [p.tolist() for p in model.probs(enc)]
        logits = [z.tolist() for z in model.forward(enc)]
        questions = []
        for i, (q, m, p) in enumerate(zip(rec["questions"], meta, probs)):
            questions.append({"id": m["id"], "instructions": q["instr"], "options": q["options"],
                              "decide_index": int(enc["decide_idx"][i]),
                              "option_indices": [int(x) for x in enc["opt_idx"][i]],
                              "logits": logits[i], "probabilities": p})
        # Derselbe Lauf, aber auf das Shape-Budget gepolstert und mit der im Graphen gebauten
        # Maske. Weicht das ab, stimmt die Maskenspezifikation nicht.
        multi = jevx.multi_inputs(enc, args.length, args.questions, args.options, jevx.pad_id_of(tok))
        module = jevx.MultiQuestionKev(model, args.length, args.questions, args.options).eval()
        padded = module(*(multi[k] for k in ["input_ids", "position_ids", "segment_ids",
                                             "decide_index", "option_indices"]))
        drift = max(max(abs(a - b) for a, b in zip(logits[i], padded[i][: len(logits[i])].tolist()))
                    for i in range(len(logits)))
        packed.append({"request": raw, "state": rec["state"], "length": len(enc["ids"]),
                       "encoding": {"ids": [int(i) for i in enc["ids"]],
                                    "position_ids": [int(p) for p in enc["pos"]],
                                    "segment_ids": [int(s) for s in enc["seg"]]},
                       "questions": questions, "answer": to_answers(probs, meta)})
        print(f"gepackt: {len(questions)} Fragen, L={len(enc['ids'])}, "
              f"max|dlogit| gepolstert-vs-eng={drift:.3e}")
    jevx.write_json(out / "packed.json", {"count": len(packed), "records": packed})

    print(f"\nGolden geschrieben nach {out}")


if __name__ == "__main__":
    main()
