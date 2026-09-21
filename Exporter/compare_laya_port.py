#!/usr/bin/env python3
"""Der Swift-Port gegen laya, auf den fünf Fällen aus laya_baseline.py.

Fährt `jev --engine laya` über dieselben Anfragen, die laya_baseline.py an laya schickt, und
vergleicht Antworten und Zeiten. Vorher ist die Tabelle dazu von Hand entstanden; jetzt kommt
sie aus diesem Skript.

Was verglichen wird, steht einzeln da, statt in einem Wort wie "identisch" zu verschwinden:
dieselbe Auswahl, bei noul dieselbe Seite von 0,5, bei score dieselbe gerundete Stufe, dazu die
größten Abstände in Konfidenz, Wert und act_probability. laya rundet selbst auf vier Stellen,
der Port nicht; Abstände unterhalb von 1e-3 sind fp16 und keine abweichende Antwort.

Voraussetzung: laya_baseline.py hat Benchmarks/laya-baseline-cpu.json geschrieben, und das
Release-Binary liegt unter .build/release/jev.
"""
import argparse
import json
import pathlib
import subprocess
import tempfile

import layax
from laya_baseline import cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(layax.ROOT / "Benchmarks" / "laya-baseline-cpu.json"))
    ap.add_argument("--repeat", type=int, default=25)
    ap.add_argument("--out", default=str(layax.ROOT / "Benchmarks" / "laya-vergleich.json"))
    args = ap.parse_args()

    binary = layax.ROOT / ".build" / "release" / "jev"
    reference = {r["label"]: r for r in json.load(open(args.baseline))["runs"]}
    out = {"laya": "laya_baseline.py: PyTorch, CPU, Router(preload=True)",
           "port": f"jev --engine laya, Core ML fp16, cpuAndGPU, Release-Build, {args.repeat} Läufe",
           "runs": []}
    worst = {"konfidenz": 0.0, "wert": 0.0, "act": 0.0}
    decisions = same = 0

    with tempfile.TemporaryDirectory() as tmp:
        for i, (label, state, questions) in enumerate(cases()):
            request = pathlib.Path(tmp) / f"fall{i}.json"
            result = pathlib.Path(tmp) / f"port{i}.json"
            request.write_text(json.dumps({"state": state, "questions": questions}, ensure_ascii=False))
            subprocess.run([str(binary), "--engine", "laya", "--models", str(layax.ROOT / "Models"),
                            "--request", str(request), "--repeat", str(args.repeat),
                            "--json", str(result), "--label", label],
                           check=True, capture_output=True)
            mine = json.load(open(result))
            theirs = reference[label]
            rows = []
            for qid, want in theirs["answers"].items():
                got = mine["answers"][qid]
                kind = want["type"]
                if kind == "choice":
                    agrees = got["choice"] == want["choice"]
                    dvalue = 0.0
                elif kind == "noul":
                    agrees = (got["noul"] >= 0.5) == (want["noul"] >= 0.5)
                    dvalue = abs(got["noul"] - want["noul"])
                else:
                    agrees = round(got["score"]) == round(want["score"])
                    dvalue = abs(got["score"] - want["score"])
                dconf = abs(got["confidence"] - want["confidence"])
                dact = abs(got["action"]["act_probability"] - want["action"]["act_probability"])
                worst["konfidenz"] = max(worst["konfidenz"], dconf)
                worst["wert"] = max(worst["wert"], dvalue)
                worst["act"] = max(worst["act"], dact)
                decisions += 1
                same += agrees
                rows.append({"frage": qid, "gleiche_entscheidung": agrees, "d_konfidenz": round(dconf, 6),
                             "d_wert": round(dvalue, 6), "d_act": round(dact, 6)})
            factor = theirs["median_ms"] / mine["median_ms"]
            out["runs"].append({
                "label": label, "fragen": len(questions),
                "routing_laya": theirs["routed_to"], "routing_port": mine["routed_to"],
                "laya_median_ms": round(theirs["median_ms"], 2),
                "port_median_ms": round(mine["median_ms"], 2),
                "port_min_ms": round(mine["min_ms"], 2),
                "faktor_median": round(factor, 2),
                "antworten": rows,
            })
            print(f"{label:22s} laya {theirs['median_ms']:7.1f} ms  Port {mine['median_ms']:7.1f} ms  "
                  f"{factor:4.1f}x  Routing {theirs['routed_to']}/{mine['routed_to']}")

    out["gleiche_entscheidungen"] = f"{same}/{decisions}"
    out["gleiches_routing"] = all(r["routing_laya"] == r["routing_port"] for r in out["runs"])
    out["groesste_abstaende"] = {k: round(v, 6) for k, v in worst.items()}
    layax.write_json(args.out, out)
    print(f"gleiche Entscheidung {same}/{decisions}, größte Abstände "
          f"Konfidenz {worst['konfidenz']:.4f}, Wert {worst['wert']:.4f}, act {worst['act']:.4f}")
    print(f"geschrieben: {args.out}")


if __name__ == "__main__":
    main()
