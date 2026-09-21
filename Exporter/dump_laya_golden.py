#!/usr/bin/env python3
"""Referenzdaten fuer den laya-Port, erzeugt aus dem Originalmodell.

Drei Dateien je Checkpoint unter Golden/laya/<checkpoint>/:
  runtime.json    Token-IDs, Budgets, Temperaturtabelle, Aufbau der Sequenz
  tokenizer.json  Text -> Token-IDs, dieselben Grenzfaelle wie beim kev-Tokenizer
  records.json    vollstaendige Encodings, Logits beider Koepfe und die fertige Antwort

Die Antworten kommen aus laya.Agent.system_one selbst, nicht aus einem Nachbau. Was hier
steht, ist damit das, was laya ausgibt, und nicht das, was wir glauben, dass es ausgibt.
"""
import argparse
import json
import pathlib

import torch

import layax
from dump_golden import EDGE_CASES, REQUESTS


# Zustaende und Anweisungen, die den Maskentext selbst enthalten. build_sequence ersetzt ihn
# ueberall durch ein Leerzeichen, und welcher Text das ist, haengt am Checkpoint: "[MASK]" bei
# den ModernBERT-Ablegern, "<mask>" bei mmBERT. Wer einen davon fest verdrahtet, schleust auf
# dem anderen Checkpoint ein Markertoken in den Zustand und verschiebt damit jede Ablesestelle.
MASK_REQUESTS = [
    {
        "state": "Der Nutzer schrieb woertlich: [MASK] und danach <mask>, beides als Text.",
        "questions": {"mask_in_state": {"type": "noul",
                                        "instructions": "Enthaelt der Zustand Steuerzeichen?"}},
    },
    {
        "state": {"log": ["<mask>", "[MASK]"], "note": "a<mask>b [MASK] c"},
        "questions": {"mask_in_options": {
            "type": "choice",
            "instructions": "Welche Marke steht im Zustand? [MASK] oder <mask>?",
            "criteria": {"eckig": "die Form [MASK]", "spitz": "die Form <mask>",
                         "beide": "beide Formen", "keine": "keine davon"}}},
    },
]


# Fragen, die in Temperatureimer fallen, die laya's Tabelle nicht kennt. Dort gilt der
# Rückfallwert aus cfg["temperature"]. Keine der übrigen Referenzfragen trifft so einen Eimer,
# und genau deshalb blieb unbemerkt, dass der Port den falschen Rückfallwert las.
BUCKET_REQUESTS = [
    {"state": "The server has been down for two hours and customers cannot pay.",
     "questions": {"score_two": {"type": "score", "instructions": "Is this an incident?",
                                 "criteria": ["no", "yes"]}}},
    {"state": "The server has been down for two hours and customers cannot pay.",
     "questions": {"score_six": {"type": "score", "instructions": "How severe is this incident?",
                                 "criteria": ["none", "cosmetic", "minor", "major", "critical", "catastrophic"]}}},
    {"state": "The server has been down for two hours and customers cannot pay.",
     "questions": {"score_eleven": {"type": "score", "instructions": "Rate the urgency from 0 to 10.",
                                    "criteria": [str(i) for i in range(11)]}}},
    {"state": "Please move my meeting with the sales team to Thursday afternoon.",
     "questions": {"choice_eleven": {"type": "choice", "instructions": "Which tool handles this?",
                                     "criteria": {f"tool_{i}": None for i in range(10)} | {"calendar": "meetings and scheduling"}}}},
]

# Der Maskentext mit einer kombinierenden Marke direkt dahinter. Python ersetzt Code-Punkt für
# Code-Punkt; Foundation ohne .literal findet den Text in so einer Folge nicht, und dann landet
# ein echtes Markertoken im Zustand.
MASK_REQUESTS.append({
    "state": "Kunde schrieb [MASK]" + chr(0x301) + " heute und <mask>" + chr(0xFE0F) + " gestern",
    "questions": {"mask_with_mark": {"type": "noul", "instructions": "Ist das eine Beschwerde?"}},
})


# Viele Optionen. laya hat keine feste Obergrenze, es kürzt jede Option bis auf vier Token und
# nimmt, was vor max_len passt: gemessen 169 auf english, 257 auf multilingual, 340 auf
# typed-decisions. Der erste Export hatte K=32 und lehnte alles darüber ab. Die Größen hier
# liegen knapp unter diesen Grenzen, die Überlaufanfrage darüber: dort bricht laya ab, und der
# Port muss es ebenso tun, statt eine Frage mit weniger Optionen zu beantworten.
# Namen wie o17 und keine Beschreibung, so wurden die Grenzen gemessen.
MANY = {"english": (150, 100, 200), "multilingual": (240, 200, 300), "typed-decisions": (300, 200, 400)}


def many_requests(checkpoint):
    choices, levels, _ = MANY[checkpoint]
    names = [f"o{i}" for i in range(choices - 1)]
    criteria = {name: None for name in names[: choices // 2]}
    criteria["billing"] = "invoices, payments, refunds"
    criteria.update({name: None for name in names[choices // 2:]})
    return [
        {"state": "We were billed twice for March. Please refund the duplicate charge.",
         "questions": {"choice_many": {"type": "choice", "instructions": "Which team handles this?",
                                       "criteria": criteria}}},
        {"state": "The server has been down for two hours and customers cannot pay.",
         "questions": {"score_many": {"type": "score", "instructions": "Rate the urgency.",
                                      "criteria": [str(i) for i in range(levels)]}}},
    ]


def overflow_request(checkpoint):
    count = MANY[checkpoint][2]
    return {"state": "We were billed twice for March.",
            "questions": {"choice_overflow": {"type": "choice", "instructions": "Which team?",
                                              "criteria": {f"o{i}": None for i in range(count)}}}}


def one_question_requests(checkpoint):
    """laya beantwortet mehrere Fragen in einem Durchlauf, unser Vertrag eine pro Aufruf.

    Beides muss dasselbe ergeben, sonst stimmt der Vertrag nicht. Deshalb wird jede Anfrage
    aus dem kev-Korpus hier einzeln je Frage gefahren und zusaetzlich einmal komplett.
    """
    out = []
    for raw in REQUESTS + MASK_REQUESTS + BUCKET_REQUESTS + many_requests(checkpoint):
        for qid, qdef in raw["questions"].items():
            out.append({"state": raw["state"], "questions": {qid: qdef}})
    return out


def main():
    layax.quiet()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", choices=list(layax.CHECKPOINTS), default="english")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from laya.common import QTYPES, build_sequence, render_options, serialize_state

    agent = layax.load_laya(args.checkpoint)
    spec = layax.CHECKPOINTS[args.checkpoint]
    tok, model = agent.tok, agent.model
    max_len = agent.cfg.get("max_len", spec["max_len"])
    head_max_len = agent.cfg.get("head_max_len", spec["head_max_len"])
    out = pathlib.Path(args.out or (layax.ROOT / "Golden" / "laya" / args.checkpoint))

    layax.write_json(out / "runtime.json", {
        "checkpoint": args.checkpoint,
        "encoder": str(agent.cfg.get("encoder", "")),
        "max_len": int(max_len),
        "head_max_len": int(head_max_len),
        "option_token_limit": 48,
        "head_min_tokens": 8,
        "option_budget_floor": 16,
        "pad_id": int(tok.pad_token_id),
        "cls_id": int(tok.cls_token_id),
        "sep_id": int(tok.sep_token_id),
        "mask_id": int(tok.mask_token_id),
        "mask_token": tok.mask_token,
        "qtypes": dict(QTYPES),
        # agent.temperature: der Rückfallwert, den laya tatsächlich benutzt (cfg), nicht der
        # nie gelesene Puffer im Modell.
        "temperature": [float(x) for x in agent.temperature],
        "temperature_by_options": dict(agent.cfg.get("temperature_by_options", {})),
        "masked_logit": layax.MASK_NEG,
    })

    cases = []
    for text in EDGE_CASES:
        cases.append({"text": text, "ids": [int(i) for i in tok(text, add_special_tokens=False)["input_ids"]]})
    # Zusaetzlich das, was in einer laya-Sequenz wirklich vorkommt: Optionstexte mit
    # fuehrendem Leerzeichen und die Kopfzeile mit dem Fragetyp.
    # lstrip: der Maskentreffer schluckt allen Unicode-Leerraum davor, nicht nur ASCII. Beide
    # Formen für jeden Checkpoint; beim jeweils anderen sind sie gewöhnlicher Text.
    for mask in ["[MASK]", "<mask>"]:
        for space in [" ", "\t", "\r ", chr(0xA0), chr(0x3000), chr(0x2028), chr(0x85), chr(0x0B), "  " + chr(0x2003)]:
            for text in ["a" + space + mask, "Hello" + space + mask + " world", space + mask + chr(0xB2) + " "]:
                cases.append({"text": text, "ids": [int(i) for i in tok(text, add_special_tokens=False)["input_ids"]]})
    for text in [" billing: invoices, payments, refunds", " level 0: neutral",
                 " true: yes, the statement holds", "choice question: Which team?",
                 "score question: Wie stark ist die Eskalation?", " none", "  doppelt"]:
        cases.append({"text": text, "ids": [int(i) for i in tok(text, add_special_tokens=False)["input_ids"]]})
    layax.write_json(out / "tokenizer.json", {"count": len(cases), "cases": cases})

    records = []
    for raw in one_question_requests(args.checkpoint):
        qid = next(iter(raw["questions"]))
        q = agent._to_internal(raw["questions"][qid])
        ids, markers = build_sequence(tok, raw["state"], q, max_len, head_max_len)
        qtype = QTYPES[q["t"]]
        inputs = layax.model_inputs(ids, markers, qtype, max_len, len(markers), tok.pad_token_id)
        with torch.no_grad():
            logits, act = model(inputs["input_ids"].long(), inputs["attention_mask"].long(),
                                inputs["marker_pos"].long()[None, :],
                                inputs["marker_mask"].bool()[None, :], inputs["qtype"].long())
        answer = agent.system_one(raw["state"], raw["questions"])
        records.append({
            "request": raw,
            "question_id": qid,
            "qtype": int(qtype),
            "rendered": {"state": serialize_state(raw["state"]),
                         "instructions": q["ins"], "options": render_options(q)},
            "encoding": {"ids": [int(i) for i in ids], "markers": [int(m) for m in markers],
                         "length": len(ids)},
            "logits": [float(x) for x in logits.reshape(-1)],
            "act_logits": [float(x) for x in act.reshape(-1)],
            "answer": answer["answers"][qid],
            "usage": answer["usage"],
        })
        print(f"{qid:16s} {q['t']:6s} L={len(ids):4d} K={len(markers):3d} "
              f"conf={records[-1]['answer']['confidence']:.4f}")

    # Die Anfrage, an der laya abbricht. Festgehalten wird, wie viele Marken noch vor max_len
    # liegen; der Port muss mit genau dieser Zahl ablehnen.
    raw = overflow_request(args.checkpoint)
    qid = next(iter(raw["questions"]))
    q = agent._to_internal(raw["questions"][qid])
    ids, markers = build_sequence(tok, raw["state"], q, max_len, head_max_len)
    try:
        agent.system_one(raw["state"], raw["questions"])
        raise SystemExit(f"{qid}: laya hat {len(render_options(q))} Optionen angenommen, erwartet war ein Abbruch")
    except ValueError as error:
        overflow = {"request": raw, "question_id": qid, "options": len(render_options(q)),
                    "fitting": len(markers), "error": str(error)}
    print(f"{qid:16s} {len(render_options(q))} Optionen, {len(markers)} passen, laya: {overflow['error']}")

    layax.write_json(out / "records.json", {"count": len(records), "records": records,
                                            "overflow": overflow})
    print(f"geschrieben nach {out}")


if __name__ == "__main__":
    main()
