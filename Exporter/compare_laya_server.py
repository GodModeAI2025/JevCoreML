#!/usr/bin/env python3
"""`jev --engine laya --serve` gegen `laya.Router.predict`, Antwort für Antwort.

laya hat keinen Server. Die Antwort des Ports soll deshalb genau das sein, was `Router.predict`
zurückgibt, nur als HTTP-Körper: dieselben Schlüssel in derselben Reihenfolge, dieselben
gerundeten Zahlen bis auf fp16, dasselbe `routing`. Dazu kommen `latency_ms` und `passes`, die
der Server auch für kev ausweist.

Geprüft wird, nicht nur ausgegeben:
  Schlüssel und Reihenfolge auf jeder Ebene, `model`, `usage`, die Namen der Optionen, die
  Legende bei score, im `routing` Modell, Repo, Erkennung und Ablauf.
Gemessen und berichtet:
  gleiche Entscheidung und die größten Abstände der Zahlen. `reason` ist im Port deutsch und
  wird nicht verglichen. Für die fünf Messfälle dazu die Zeit: laya im eigenen Prozess, der
  Port von außen über HTTP, also samt Verbindungsaufbau und JSON. Das benachteiligt den Port.

Voraussetzung: ein laufender Server, etwa
  jev --engine laya --models Models --serve --port 8131
"""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request

import layax
from laya_baseline import QUESTIONS, cases


def extra_cases():
    """Was die Messfälle nicht abdecken: Hinweise fürs Routing und Legenden, die keine Texte sind."""
    levels = {"stars": {"type": "score", "instructions": "How satisfied is the customer?",
                        "criteria": [1, 2.5, None, {"label": "great"}, "five"]}}
    noul = {"churn_risk": QUESTIONS["churn_risk"]}
    german = {"body": "Der Kunde wurde zweimal belastet und will sein Geld zurueck."}
    english = {"body": "We were billed twice for March, please refund us."}
    return [
        ("Legende aus Zahlen und Objekten", english, levels, {}),
        ("model=multilingual auf Englisch", english, noul, {"model": "multilingual"}),
        ("task=typed_decisions", english, noul, {"task": "typed_decisions"}),
        ("lang=de auf Englisch", english, noul, {"lang": "de"}),
        ("lang=en auf Deutsch", german, noul, {"lang": "en"}),
        ("ohne Buchstaben", {"body": "12345 !!! 67"}, noul, {}),
        ("gemischte Schrift", {"body": "Order 4411: заказ пришёл дважды, refund please"}, noul, {}),
        # Mehr Optionen, als kev zulässt und als der erste Export fasste. laya nimmt sie an.
        ("150 Optionen", english, many(150), {}),
        ("300 Optionen, typed-decisions", english, many(300), {"task": "typed_decisions"}),
    ]


def many(count):
    criteria = {f"o{i}": None for i in range(count)}
    criteria["billing"] = "invoices, payments, refunds"
    return {"team": {"type": "choice", "instructions": "Which team handles this?", "criteria": criteria}}


def post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8131/v1/systemone")
    ap.add_argument("--out", default=str(layax.ROOT / "Benchmarks" / "laya-server-vergleich.json"))
    ap.add_argument("--repeat", type=int, default=25)
    args = ap.parse_args()

    from laya import Router
    router = Router(preload=True, device="cpu")

    runs = [(label, state, questions, {}) for label, state, questions in cases()] + extra_cases()
    timed_labels = {label for label, _, _ in cases()}
    # Beide Seiten einmal ungemessen, damit kein Laden in der ersten Messung steckt.
    for label, state, questions in cases():
        router.predict(state, questions)
        post(args.url, {"state": state, "questions": questions})
    worst = {"konfidenz": 0.0, "wert": 0.0, "act": 0.0, "wahrscheinlichkeit": 0.0}
    decisions = same = 0
    out = {"laya": "laya.Router(preload=True, device='cpu').predict, im Prozess",
           "port": "jev --engine laya --serve (Release), POST /v1/systemone, gemessen beim Client",
           "wiederholungen": args.repeat, "runs": []}

    for label, state, questions, hints in runs:
        want = router.predict(state, questions, **hints)
        body = {"state": state, "questions": questions}
        if "model" in hints:
            body["model"] = hints["model"]
        if "task" in hints:
            body["task"] = hints["task"]
        if "lang" in hints:
            body["lang"] = hints["lang"]
        status, got = post(args.url, body)
        assert status == 200, (label, status, got)

        # Oben: laya-Schlüssel in laya-Reihenfolge, dazwischen die beiden des Servers.
        assert [k for k in got if k in want] == list(want), (label, list(got), list(want))
        assert got["model"] == want["model"], label
        assert got["usage"] == want["usage"], (label, got["usage"], want["usage"])

        r_want, r_got = dict(want["routing"]), got["routing"]
        assert list(r_got) == list(r_want), (label, list(r_got), list(r_want))
        repo = r_want["repo"]
        if isinstance(repo, (list, tuple)):  # laya legt beim erkannten Ablauf das Tupel ab
            repo = "/".join(x for x in repo if x)
        for key, value in [("model", r_want["model"]), ("repo", repo),
                           ("detection", r_want["detection"]), ("workflow", r_want["workflow"])]:
            assert r_got[key] == value, (label, key, r_got[key], value)

        rows = []
        assert list(got["answers"]) == list(want["answers"]), label
        for qid, w in want["answers"].items():
            g = got["answers"][qid]
            assert list(g) == list(w), (label, qid, list(g), list(w))
            assert list(g["action"]) == list(w["action"]), (label, qid)
            kind = w["type"]
            assert g["type"] == kind
            dprob = 0.0
            if kind == "choice":
                assert list(g["probabilities"]) == list(w["probabilities"]), (label, qid)
                agrees = g["choice"] == w["choice"]
                dvalue = 0.0
                dprob = max(abs(g["probabilities"][k] - w["probabilities"][k]) for k in w["probabilities"])
            elif kind == "score":
                assert g["legend"] == w["legend"], (label, qid, g["legend"], w["legend"])
                assert list(g["probabilities"]) == list(w["probabilities"]), (label, qid)
                agrees = round(g["score"]) == round(w["score"])
                dvalue = abs(g["score"] - w["score"])
                dprob = max(abs(g["probabilities"][k] - w["probabilities"][k]) for k in w["probabilities"])
            else:
                agrees = (g["noul"] >= 0.5) == (w["noul"] >= 0.5)
                dvalue = abs(g["noul"] - w["noul"])
            dconf = abs(g["confidence"] - w["confidence"])
            dact = abs(g["action"]["act_probability"] - w["action"]["act_probability"])
            for key, value in [("konfidenz", dconf), ("wert", dvalue), ("act", dact), ("wahrscheinlichkeit", dprob)]:
                worst[key] = max(worst[key], value)
            decisions += 1
            same += agrees
            rows.append({"frage": qid, "gleiche_entscheidung": agrees, "d_konfidenz": round(dconf, 6),
                         "d_wert": round(dvalue, 6), "d_act": round(dact, 6)})
        run = {"label": label, "hinweise": hints, "checkpoint": r_got["model"],
               "input_tokens": got["usage"]["input_tokens"], "antworten": rows}
        timing = ""
        if not hints and args.repeat > 0 and label in timed_labels:
            laya_ms, port_ms = [], []
            for _ in range(args.repeat):
                t = time.perf_counter()
                router.predict(state, questions)
                laya_ms.append((time.perf_counter() - t) * 1000)
            for _ in range(args.repeat):
                t = time.perf_counter()
                post(args.url, body)
                port_ms.append((time.perf_counter() - t) * 1000)
            run["laya_median_ms"] = round(statistics.median(laya_ms), 2)
            run["port_http_median_ms"] = round(statistics.median(port_ms), 2)
            run["faktor"] = round(run["laya_median_ms"] / run["port_http_median_ms"], 2)
            timing = (f"  laya {run['laya_median_ms']:6.1f} ms, Port über HTTP "
                      f"{run['port_http_median_ms']:6.1f} ms, {run['faktor']:.1f}x")
        out["runs"].append(run)
        print(f"{label:34s} {r_got['model']:16s} Form gleich, "
              f"{sum(r['gleiche_entscheidung'] for r in rows)}/{len(rows)} gleiche Entscheidung{timing}")

    out["gleiche_form"] = True
    out["gleiche_entscheidungen"] = f"{same}/{decisions}"
    out["groesste_abstaende"] = {k: round(v, 6) for k, v in worst.items()}
    layax.write_json(args.out, out)
    print(f"Form in allen {len(runs)} Fällen gleich, gleiche Entscheidung {same}/{decisions}, "
          f"größte Abstände {out['groesste_abstaende']}")
    print(f"geschrieben: {args.out}")


if __name__ == "__main__":
    main()
