#!/usr/bin/env python3
"""Referenz fuer die Schrifterkennung und die Routingentscheidung.

Ohne diese Datei ist der Swift-Router eine Nacherzaehlung. Mit ihr ist er geprueft: dieselben
Zustaende, dieselbe erkannte Schrift, derselbe Sprachtipp, derselbe Checkpoint.

Die Faelle decken ab, woran eine Nacherzaehlung scheitert: die Grenze zwischen "lateinisch" und
"fremd", die Schwelle der Funktionswoerter, Zustaende ohne Buchstaben, verschachtelte Zustaende
(Schluessel zaehlen nicht mit) und die vier typed-decisions-Ablaeufe.
"""
import argparse

import layax

STATES = [
    "The customer received two identical charges for the same order.",
    "Der Kunde meldet, dass die App beim Start abstuerzt.",
    "Der Kunde meldet, dass die Anwendung beim Start abstürzt und nicht mehr reagiert.",
    "Le client a reçu deux prélèvements identiques pour la même commande.",
    "El cliente recibió dos cargos idénticos por el mismo pedido.",
    "Il cliente ha ricevuto due addebiti identici per lo stesso ordine.",
    "De klant heeft twee identieke afschrijvingen ontvangen voor dezelfde bestelling.",
    "O cliente recebeu duas cobranças idênticas para o mesmo pedido.",
    "顧客は同じ注文で同一の請求を二度受けました。",
    "고객이 동일한 주문에 대해 두 번 청구되었습니다.",
    "客户为同一订单收到了两笔相同的费用。",
    "العميل تلقى رسمين متطابقين لنفس الطلب.",
    "ग्राहक को एक ही ऑर्डर के लिए दो समान शुल्क मिले।",
    "Ο πελάτης χρεώθηκε δύο φορές για την ίδια παραγγελία.",
    "Клиент получил два одинаковых списания за один заказ.",
    "ลูกค้าได้รับการเรียกเก็บเงินสองครั้งสำหรับคำสั่งซื้อเดียวกัน",
    "",
    "   ",
    "12345 67890 !!! ??? ---",
    "2024-05-01T12:00:00Z",
    "ok",
    "the and is are",
    "der die das und",
    "Mixed: der Kunde and the customer sind beide betroffen",
    "Réservation confirmée",
    "naïve café résumé",
    {"kunde": {"plan": "Enterprise", "seit": "2021"},
     "vorfall": ["Doppelbuchung", "zwei identische Belastungen wurden abgebucht"]},
    {"customer": {"plan": "Enterprise"}, "incident": ["duplicate charge", "two identical debits"]},
    {"ticket": "客户为同一订单收到了两笔相同的费用", "id": 4711},
    [{"role": "user", "content": "Bitte storniere die doppelte Buchung."},
     {"role": "agent", "content": "Ich sehe mir das an."}],
    # Gleichstände: in detect_script verliert "latin" jeden, unter den anderen Schriften gewinnt
    # die zuerst gesehene. Der Serververgleich hat den ersten Fall gefunden, keiner der obigen
    # Zustände traf ihn.
    "Order 4411: заказ пришёл дважды, refund please",
    "abc абв",
    "ab αβ вг",
    "вг αβ ab",
    "漢字かな",
    "かな漢字",
    "한글漢字",
    "ab 漢字 한글",
]

QUESTION_SETS = [
    [],
    ["route"],
    ["action", "category", "churn_risk", "needs_human", "urgency"],
    ["action", "needs_review", "outcome", "risk", "urgency"],
    ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"],
    ["credential_compromise", "disposition", "severity", "true_positive", "urgency"],
    ["action", "category", "churn_risk", "needs_human"],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(layax.ROOT / "Golden" / "laya" / "routing.json"))
    args = ap.parse_args()

    from laya.lang import analyse, detect_script, guess_latin_language, state_text
    from laya.router import Router, match_typed_decisions_workflow

    router = Router()
    auto = Router(auto_task_detection=True)

    detections = []
    for state in STATES:
        det = analyse(state)
        text = state_text(state)
        detections.append({
            "state": state,
            "text": text,
            "script": det["script"],
            # Ungerundet: der Server gibt die Erkennung aus, also muss sie bis aufs Bit stimmen.
            "script_profile": det["script_profile"],
            "language": det["language"],
            "is_english": det["is_english"],
            "non_latin_fraction": det["non_latin_fraction"],
            "detect_script": detect_script(text),
            "guess_latin_language": guess_latin_language(text),
            "model": router.route(state)["model"],
        })

    routes = []
    for state in STATES[:6] + STATES[8:12]:
        for ids in QUESTION_SETS:
            questions = {qid: {"type": "noul", "instructions": qid} for qid in ids}
            routes.append({
                "state": state,
                "question_ids": ids,
                "workflow": match_typed_decisions_workflow(questions),
                "model": router.route(state, questions)["model"],
                "model_auto_task": auto.route(state, questions)["model"],
            })
    # Und die Vorrangregeln
    overrides = []
    for kwargs in [{"model": "multi"}, {"model": "typed"}, {"model": "en"},
                   {"task": "typed_decisions"}, {"task": "multilingual"},
                   {"lang": "de"}, {"lang": "en-GB"}, {"lang": "eng"}]:
        overrides.append({"kwargs": kwargs,
                          "model": auto.route(STATES[0], {"urgency": {"type": "noul", "instructions": "x"}},
                                              **kwargs)["model"]})

    layax.write_json(args.out, {"detections": detections, "routes": routes, "overrides": overrides})
    print(f"{len(detections)} Erkennungen, {len(routes)} Routen, {len(overrides)} Vorrangfaelle -> {args.out}")


if __name__ == "__main__":
    main()
