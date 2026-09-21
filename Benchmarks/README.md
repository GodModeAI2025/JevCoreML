# Benchmarks

Die Messungen hinter den Zahlen in der README, als JSON so, wie die Skripte sie geschrieben
haben. Gemessen auf einem MacBook Pro mit M5 Max unter macOS 27.2. Die laya-Läufe vom
21.09.2026 liefen auf Akku im Energiesparmodus; unter Dauerlast drückt der den Takt, am Netzteil
dürften die Mediane etwas niedriger liegen. Ausführlich erklärt ist alles in
[docs/benchmark.md](../docs/benchmark.md) und [docs/laya-port.md](../docs/laya-port.md).

## laya

### Gleiche Antworten

| Prüfung | gleich | max ∣Δp∣ | Datei |
|---|---|---|---|
| 15 Referenzanfragen, english | 15/15 | 2,7e−03 | `laya-parity-english-golden.json` |
| 15 Referenzanfragen, multilingual | 15/15 | 2,3e−03 | `laya-parity-multilingual-golden.json` |
| 15 Referenzanfragen, typed-decisions | 15/15 | 1,6e−03 | `laya-parity-typed-decisions-golden.json` |
| AG News, 1000 Zeilen | 1000/1000 | 3,6e−03 | `laya-parity-agnews-gpu.json` |
| DAIR Emotion, 1000 Zeilen | 999/1000 | 2,3e−02 | `laya-parity-emotion-gpu.json` |
| fünf Messfälle, 17 Antworten | 17/17 | 1e−03 in der Konfidenz | `laya-vergleich.json` |
| HTTP gegen `Router.predict`, 14 Anfragen | 26/26, Form gleich | 2,1e−03 | `laya-server-vergleich.json` |

Die eine abweichende Emotion-Zeile ist ein Gleichstand: laya wählt `sadness` mit 0,4978, der
Port `fear` mit 0,4977.

### Tempo

| Fall | laya, PyTorch CPU | Core ML | Faktor |
|---|---|---|---|
| 1 Frage, englisch | 112,9 ms | **9,0 ms** | 12,5× |
| 4 Fragen, englisch | 314,8 ms | **33,6 ms** | 9,4× |
| 10 Fragen, englisch | 709,2 ms | **83,4 ms** | 8,5× |
| 1 Frage, deutsch | 37,4 ms | **6,0 ms** | 6,2× |
| 1 Frage, hindi | 43,9 ms | **6,3 ms** | 7,0× |

Quelle: `laya-vergleich.json` (Kommandozeile, Median aus 25 Läufen) gegen `laya-baseline-cpu.json`
(laya, Median aus 20). Über HTTP und mit beiden Seiten im selben Lauf steht dasselbe in
`laya-server-vergleich.json`, dort liegt der Faktor zwischen 5,7 und 11,8.

Rechenwerke im Vergleich, `laya-backends-EN.json`: GPU 6,3 ms, CPU 85,7 ms, Neural Engine
268,4 ms, alle drei mit richtigen Antworten.

## kev

| Prüfung | Ergebnis | Datei |
|---|---|---|
| Entwicklungssuite, 1468 Fragen, gegen `kev.serve` | 2 abweichende Antworten, 80,79 % gegen 80,93 % | `eval-nativ-dev.json`, `eval-referenz-dev.json`, `vergleich-je-frage.json` |
| zurückgehaltener Test-Split, 1440 Fragen | 82,01 % | `eval-nativ-test.json` |
| mit `--buckets`, drei Längen | dieselben Antworten, Median 22,7 statt 70,3 ms | `eval-pool-dev.json`, `eval-single-L1024-dev.json`, `vergleich-pool.json` |
| Banking77, 77 Kategorien, 3075 Fragen | 78,8 %, Median 70 ms | `bench-intent77-nativ.json` |
| Core ML gegen PyTorch, Logits | fp16 auf der GPU max ∣Δp∣ 3,3e−03 | `coreml-parity-Kev06B-fp16.json` |

## Selbst messen

Die Skripte liegen unter `Exporter/`, eingerichtet von `scripts/reproduce.sh`:

```bash
cd Exporter
./.venv/bin/python laya_baseline.py --device cpu --repeat 20 --out ../Benchmarks/laya-baseline-cpu.json
./.venv/bin/python compare_laya_port.py                       # Kommandozeile gegen laya
../.build/release/jev --engine laya --models ../Models --serve --port 8131 &
./.venv/bin/python compare_laya_server.py                     # HTTP gegen Router.predict
./.venv/bin/python laya_parity_corpus.py --task agnews --limit 1000 --units gpu \
    --model ../Models/Laya-EN-L512-K512-fp16.mlpackage --out ../Benchmarks/laya-parity-agnews-gpu.json
```

Ohne Python geht nur die Seite des Ports:

```bash
.build/release/jev --engine laya --models Models --request Examples/support-triage.json --repeat 25
```

Vor dem Messen `pmset -g | grep powermode` prüfen; `1` heißt Energiesparmodus.
