# Benchmark-Bericht JevCoreML

Stand 21.09.2026. Alle Messungen auf einem Mac mit M5 Max und 128 GB, macOS 27.2, Xcode 27,
Swift 6.4, coremltools 9.0, transformers 4.57.6, torch 2.8, laya 0.3.4.

Die kev-Zahlen unten stammen aus den festen Exporten `Kev06B-Q4-fp16` (L=512) und
`Kev06B-L1024-Q4K96-fp16` sowie dem Pool aus drei Buckets. Seit dem Abend desselben Tages
ersetzt sie ein Paket, `JevCoreML.mlpackage`, mit sechs aufgezählten Längen von 128 bis 3072,
acht Fragen und 256 Optionen; bei 512 Token liefert es auf der GPU bitgleiche Logits mit
`Kev06B-Q4-fp16`, über alle Längen 0 Argmax-Flips gegen PyTorch
(`Benchmarks/fanout-parity-JevCoreML.json`). Die Paritäts- und Trefferquotenmessungen hier
bleiben damit gültig. Anders sind der Dateiname, das Rechenwerk (`JevRuntime` setzt `.all`
jetzt selbst auf `.cpuAndGPU`) und die Einmalkosten, beides in der README.

Gemessen wird ein Port: kev-0.6b, ein Qwen3-0.6B-Base mit LoRA und Pointer-Head, läuft statt in
PyTorch nativ über Core ML. Die Frage ist zweigeteilt. Kommen dieselben Zahlen heraus? Und ist
das Ergebnis mit dem vergleichbar, was für Jev veröffentlicht ist?

## 1. Methode

Drei Referenzen, absichtlich unterschiedlich streng:

1. **PyTorch direkt** (`kev.evaluate.load`, fp32, CPU). Liefert die Golden-Logits.
2. **PyTorch als Server** (`kev.serve` auf MPS, fp32). Dieselbe HTTP-Schnittstelle wie der
   native Server, also derselbe Code auf beiden Seiten des Vergleichs.
3. **Veröffentlichte Zahlen** für Jev 1.13 aus dem Benchmark von AY Automate vom 19.09.2026
   (`summary.json`, `analysis.txt`) und aus der kev-Modellkarte.

Was nicht gemessen wurde: Jev selbst. Es gibt hier keinen API-Schlüssel und keinen eigenen
Nachtest der veröffentlichten Läufe. Alle Aussagen über Jev stammen aus deren Dateien.

## 2. Parität: kommt dasselbe heraus?

### 2.1 Tokenizer und Encoding

| Prüfung | Umfang | Ergebnis |
|---|---|---|
| Token-IDs Swift gegen Python | 72 handverlesene plus 14 960 aus einem Suchlauf | byte-identisch |
| Encoding einer Frage | 6 Anfragen | identisch in IDs, Positionen, `decide_index`, `option_indices` |
| Encoding mehrerer Fragen | 2 Anfragen, 2 und 3 Fragen | identisch, auch in den Segment-IDs |

Der Tokenizer-Korpus enthält die Fälle, an denen Nachbauten scheitern: zerlegte Umlaute, acht
Leerzeichenarten, ZWJ-Emoji, CJK, Arabisch, BOM, CRLF, Apostroph-Kontraktionen, NFC gegen NFKC
und Added-Token wie `<think>`, die nicht dem `<|name|>`-Muster folgen.

### 2.2 Modell

Sechs Anfragen über alle drei Fragetypen, gegen die PyTorch-Logits:

| Präzision | Einheit | max Δ Logit | max Δ Wahrscheinlichkeit | Argmax-Flips |
|---|---|---|---|---|
| fp32 | CPU | 1,3e-05 | 2,2e-06 | 0/6 |
| fp32 | GPU | 2,9e-05 | 3,5e-06 | 0/6 |
| fp16 | CPU | 1,4e-01 | 9,6e-03 | 0/6 |
| fp16 | GPU | 3,4e-02 | 3,3e-03 | 0/6 |
| fp16 | ANE | 3,4e-02 | 3,3e-03 | 0/6 |

Die Swift-Werte sind identisch mit denen, die dieselbe Prüfung in Python direkt gegen das
`.mlpackage` misst. Das ist der Punkt der Übung: wäre im Swift-Pfad etwas anders, stünde dort
eine andere Zahl.

### 2.3 Fan-Out

Mehrere Fragen in einer Sequenz gegen den gepackten PyTorch-Lauf:
3,7e-06 in fp32, 3,3e-03 in fp16, keine abweichende Auswahl. Die Aufteilung auf einen
gemeinsamen Durchlauf kostet also nichts an Genauigkeit.

### 2.4 Ganze Suite, Frage für Frage

Kevs eigene Entwicklungssuite, 1204 Anfragen, 1468 bewertete Fragen, beide Seiten über
`POST /v1/systemone`:

| | Trefferquote | ECE | Median | p95 |
|---|---|---|---|---|
| nativ, Core ML fp16, L=1024 | 80,79 % | 0,126 | 68 ms | 80 ms |
| `kev.serve`, PyTorch fp32, MPS | 80,93 % | 0,128 | 59 ms | 236 ms |

Aggregate können sich gegenseitig aufhebende Fehler verstecken, deshalb liegen beide Läufe auch
Frage für Frage vor:

| | |
|---|---|
| Fragen verglichen | 1468 |
| abweichende Antwort | 2 (0,14 %) |
| davon nur nativ richtig | 0 |
| davon nur die Referenz richtig | 2 |
| Wahrscheinlichkeiten mehr als 0,01 auseinander | 116 (7,9 %) |
| größte Abweichung einer Wahrscheinlichkeit | 0,060 |
| mittlere Abweichung | 0,00105 |

Die beiden abweichenden Fragen sind derselbe Fall: eine Auswahl zwischen `accept` und `reject`,
bei der beide Systeme 0,50 zu 0,50 melden. Die Konfidenz ist damit 0, und jede Schwelle über
null schickt beide Fälle in die Eskalation. Auf 1468 Fragen gibt es also keine einzige
Entscheidung, die beide Systeme automatisch treffen würden und bei der sie sich uneinig wären.

Die 116 Fragen mit mehr als einer Rundungsstufe Abstand sind die ehrlichere Zahl zur halben
Genauigkeit: einzelne Wahrscheinlichkeiten bewegen sich, die Auswahl fast nie.

### 2.4b Der zurückgehaltene Test-Split

Alle Zahlen oben stammen vom Entwicklungs-Split, auf den auch kevs Modellkarte berichtet. Der
Test-Split derselben Suite war bis dahin unberührt:

| Split | Fragen | Trefferquote | ECE | Abdeckung bei 0,9 | Trefferquote dort |
|---|---|---|---|---|---|
| Entwicklung | 1468 | 80,79 % | 0,126 | 74,0 % | 89,2 % |
| Test | 1440 | 82,01 % | 0,114 | 79,3 % | 90,5 % |

Alle 1176 Anfragen des Test-Splits wurden beantwortet, keine abgelehnt. Die Zahlen liegen leicht
über denen des Entwicklungs-Splits, was gegen eine Überanpassung an die Entwicklungsdaten
spricht: an denen wurde hier nichts eingestellt, aber die Toleranzen der Paritätstests stammen
von dort.

### 2.5 Antworten über HTTP

Elf Fragen aus sechs Anfragen, beide Server, Wahrscheinlichkeiten verglichen:
maximale Abweichung 0,01 bei einem einzigen Wert, sonst exakt gleich; keine abweichende Auswahl.
0,01 ist die Rundungsstufe der API-Antwort, es ist also ein Rundungsrand, kein Modellunterschied.

### 2.6 Konformität

`vendor/kev/tests/test_api.py` gegen den nativen Server: 7 von 7 grün, darunter der Test, der
den offiziellen `typesafe_sdk`-Client benutzt. Geprüft werden Antwortform, `usage`-Felder,
Legenden, Konfidenzbereiche, Zweigisolation und vier 422-Fälle.

## 3. Geschwindigkeit

### 3.1 Eine Entscheidung

Skill-Routing, fünf Optionen, 15 Läufe, Median:

| Präzision | CPU | GPU | ANE | automatisch |
|---|---|---|---|---|
| fp16 | 251 ms | 19 ms | 86 ms | 19 ms |
| fp32 | 497 ms | 101 ms | 496 ms | 101 ms |

Beim festen Export legte Core ML fp16 unter `.all` auf die GPU: 19 ms wie mit `.cpuAndGPU`.
Seit dem Paket mit sechs Längen setzt `JevRuntime` `.all` selbst auf `.cpuAndGPU`, siehe den
Stand am Anfang. Die Neural Engine ist hier langsamer als die GPU: das Modell
ist für sie nicht zugeschnitten, und bei L=512 dominiert der Datentransport.

### 3.2 Mehrere Fragen

Drei Fragen auf demselben Ticket:

| | Durchläufe | Gesamtzeit, Median aus 11 Läufen |
|---|---|---|
| getrennt (Phase 1) | 3 | 43,7 ms |
| Fan-Out (Phase 2) | 1 | 16,3 ms |

Der Zustand läuft einmal statt dreimal durch den Backbone. Das ist das lokale Gegenstück zu dem,
was die TypeSafe-Dokumentation als Grund für Fan-Out nennt: den Kontext nicht wiederholt
übertragen. Lokal gibt es keine Übertragung, aber dieselbe Rechnung.

### 3.3 Feste gegen variable Form

Der native Pfad polstert jede Anfrage auf die exportierte Länge, der PyTorch-Server rechnet mit
der echten. Das dreht das Bild je nach Eingabe:

| | Median | p95 |
|---|---|---|
| nativ, L=1024 fest | 68 ms | 80 ms |
| `kev.serve`, variable Länge | 59 ms | 236 ms |

Bei kurzen Eingaben ist die Referenz schneller, bei langen deutlich langsamer. Die native Latenz
ist dafür planbar. Shape-Buckets verbinden beides und sind inzwischen gebaut: mit `--buckets`
hält der Server Exporte für 256, 512 und 1024 Token und nimmt je Anfrage den kürzesten passenden,
auf derselben Suite 22,7 statt 70,3 ms im Median bei denselben Antworten (`vergleich-pool.json`).

### 3.4 Einmalkosten

Modell laden 0,9 s bei warmem Compile-Cache, 2,8 s beim ersten Mal. Warmlauf 1,7 s. Ohne den
Warmlauf zahlt die erste echte Entscheidung diese 1,7 s.

## 4. Vergleich mit den veröffentlichten Zahlen

### 4.1 Banking77, 77 Kategorien

Der veröffentlichte Test benutzt 231 Zeilen; hier läuft der vollständige Test-Split mit 3080
Zeilen. Fünf davon lehnt der native Server ab: mit 77 Optionen und einem längeren Kundentext
kommt die Sequenz auf 1026 bis 1041 Token und passt nicht in den Export mit L=1024. Das ist
eine Shape-Grenze, keine Modellgrenze, und sie wird als 422 gemeldet statt still abgeschnitten.

| | n | Trefferquote | Latenz p50 | Eingabe-Tokens | Kosten je 1000 |
|---|---|---|---|---|---|
| **JevCoreML lokal (kev-0.6b)** | **3075** | **78,8 %** | **70 ms** | **976** | **0** |
| Jev 1.13 (veröffentlicht) | 231 | 78,8 % | 320 ms | 952 | 0,0400 USD |
| GPT-5.4 nano | 231 | 78,4 % | 1070 ms | 828 | 0,2049 USD |
| Gemini 3.5 Flash-Lite | 231 | 78,4 % | 670 ms | 929 | 0,3009 USD |
| Claude Haiku 4.5 | 231 | 76,2 % | 1010 ms | 1211 | 1,2550 USD |
| GPT-5.6 Terra | 231 | 84,0 % | 1100 ms | 828 | 1,9569 USD |

Die mittlere Eingabelänge liegt mit 976 gegen 952 Token nah beieinander, die Aufgabenform ist
also vergleichbar. Die Trefferquote ist es nur eingeschränkt, siehe 4.4.

### 4.2 Banking77, acht Kategorien

Der Originalsatz der acht Kategorien ist nicht veröffentlicht. Hier steht ein Nachbau derselben
Form: acht Kategorien aus dem Karten- und Zahlungscluster, 20 Zeilen je Kategorie, 160 gesamt.
Andere Zeilen, andere Kategorienauswahl, gleiche Größe.

| | n | Trefferquote | Latenz p50 |
|---|---|---|---|
| JevCoreML lokal, Nachbau | 160 | 80,6 % | 67 ms |
| `kev.serve` PyTorch, gleicher Nachbau | 160 | 80,6 % | 57 ms |
| Jev 1.13, Originalsatz | 160 | 83,8 % | 327 ms |
| GPT-5.4 nano, Originalsatz | 160 | 90,0 % | 1220 ms |

Nativ und Referenz liefern hier denselben Wert bis auf die Stelle, auch in der Abdeckungskurve.
Der Abstand zu Jev ist mit einem anderen Kategoriensatz nicht interpretierbar.

### 4.3 Abdeckung über der Konfidenzschwelle

Das ist die Kurve, an der eine Kaskade hängt: wie viel bleibt automatisch, und wie gut ist der
automatische Teil? Beide Seiten auf derselben Aufgabe, Banking77 mit 77 Kategorien:

| Schwelle | Abdeckung lokal | Trefferquote lokal | Abdeckung Jev | Trefferquote Jev |
|---|---|---|---|---|
| 0,0 | 100,0 % | 78,8 % | 100,0 % | 78,8 % |
| 0,5 | 92,0 % | 82,9 % | 92,2 % | 80,3 % |
| 0,7 | 82,2 % | 87,5 % | 82,7 % | 85,3 % |
| 0,8 | 76,5 % | 89,5 % | 77,1 % | 88,2 % |
| 0,9 | 68,6 % | 92,3 % | 66,2 % | 92,2 % |
| 0,95 | 60,7 % | 93,5 % | 56,7 % | 95,4 % |

Die beiden Kurven liegen über den ganzen Bereich innerhalb weniger Prozentpunkte übereinander.
Bei 0,9 bleiben hier 68,6 % der Fälle automatisch mit 92,3 % Trefferquote, bei Jev 66,2 % mit
92,2 %. Wer in einer Kaskade eine Schwelle sucht, käme mit beiden Systemen auf denselben Wert.

Das ist die interessantere Übereinstimmung als die Trefferquote allein. Eine gleiche Trefferquote
kann Zufall sein. Eine gleiche Kalibrierung über sechs Schwellen ist es nicht, und genau die
Kalibrierung ist das, was TypeSafe als das eigentliche Produktmerkmal beschreibt.

### 4.4 Was an diesem Vergleich nicht stimmt

Vier Einschränkungen, ohne die die Zahlen mehr behaupten, als sie zeigen.

**kev wurde auf banking77 feinjustiert.** Die Modellkarte nennt den Datensatz, das Training zog
1000 Zeilen aus dem Trainings-Split. Bewertet wird hier der Test-Split, also ungesehene Zeilen,
aber dieselbe Domäne mit denselben 77 Labels. Jev ist ein allgemeines Modell ohne dieses Ziel.
Gleiche Trefferquote bedeutet deshalb nicht gleiche Fähigkeit, sondern ein für diese Aufgabe
zugeschnittenes kleines Modell gegen ein breites großes.

**Andere Zeilen.** 3075 gegen 231. Das Konfidenzintervall des veröffentlichten Werts reicht laut
`summary.json` bei der 77-Kategorien-Aufgabe von 73,1 % bis 83,6 %. In diesem Intervall liegen
auch GPT-5.4 nano, Gemini Flash-Lite und der lokale Lauf. Auf 231 Zeilen trennt der Test diese
Systeme also gar nicht.

**Andere Hardware und andere Messstelle.** 70 ms sind hier ein lokaler Funktionsaufruf über
HTTP auf demselben Rechner. 320 ms sind bei Jev ein Netzaufruf über OpenRouter mit vier
parallelen Anfragen. Der Vergleich sagt etwas über die erlebte Latenz, nichts über Rechenzeit.

**Der Injection-Teil fehlt.** Die dritte Aufgabe des veröffentlichten Tests benutzt einen
Datensatz, der nicht mitveröffentlicht ist. Dazu gibt es hier keine Zahl.

### 4.5 Was am Vergleich trägt

Die Fehlermuster. Die häufigsten Verwechslungen des lokalen Modells auf der
Acht-Kategorien-Aufgabe sind fast dieselben wie die im veröffentlichten Jev-Lauf:

| lokal | veröffentlicht (Jev) |
|---|---|
| card_delivery_estimate → card_arrival (10) | direct debit payment not recognised → card payment not recognised (8) |
| direct_debit_payment_not_recognised → card_payment_not_recognised (7) | order physical card → card arrival (6) |
| declined_card_payment → card_payment_not_recognised (5) | card delivery estimate → card arrival (5) |
| getting_spare_card → order_physical_card (2) | getting spare card → order physical card (2) |

Zwei verschiedene Modelle, verschiedene Größen, verschiedene Trainingswege, und sie stolpern an
denselben Stellen. Das sind Kategoriengrenzen, die im Datensatz selbst unscharf sind. Der
veröffentlichte Test hält dazu fest, dass GPT-5.6 Terra bei denselben Paaren dieselben Fehler
macht. Wer an so einer Stelle Genauigkeit sucht, sollte die Kategorien ändern, nicht das Modell.

## 5. laya als zweites Modell

[laya](https://github.com/NandhaKishorM/laya) löst dieselbe Aufgabe mit einem Encoder statt
eines Decoders. Es ist nach demselben Verfahren portiert: drei `.mlpackage`, zwei Tokenizer in
Swift, derselbe Weg über Referenzdaten aus dem Original. Die Einzelheiten stehen in
[laya-port.md](laya-port.md), hier nur die beiden Zahlen, um die es geht.

### 5.1 Kommt dasselbe heraus?

| Prüfung | n | gleiche Antwort | max ∣Δp∣ |
|---|---|---|---|
| Referenzanfragen, english | 15 | 15/15 | 2,7e−03 |
| Referenzanfragen, multilingual | 15 | 15/15 | 2,3e−03 |
| Referenzanfragen, typed-decisions | 15 | 15/15 | 1,6e−03 |
| AG News | 1000 | 1000/1000 | 3,6e−03 |
| DAIR Emotion | 1000 | 999/1000 | 2,3e−02 |
| Messfälle aus `laya_baseline.py` | 17 Antworten | 17/17 | 1e−03 in der Konfidenz |
| Routingentscheidungen | 116 | 116/116 | entfällt |
| Antwort über HTTP gegen `Router.predict` | 14 Anfragen, 26 Antworten | 26/26, Form gleich | 2,1e−03 |

Trefferquote auf AG News identisch, 92,9 %; auf Emotion 57,9 % für laya und 57,8 % für den Port.
laya veröffentlicht 94,7 % und 57,3 %. Die eine abweichende Emotion-Zeile ist ein Gleichstand:
laya wählt `sadness` mit 0,4978, der Port `fear` mit 0,4977, und fp16 entscheidet, welche Klasse
vorn liegt.

### 5.2 Ist es mindestens so schnell?

Dieselbe Maschine, dieselben Anfragen, Median. laya als PyTorch auf der CPU mit
`Router(preload=True)`, der Port als Release-Build auf `cpuAndGPU`.

| Fall | Checkpoint | laya | Port | Faktor |
|---|---|---|---|---|
| 1 Frage, englisch | english | 112,9 ms | **9,0 ms** | 12,5× |
| 4 Fragen, englisch | english | 314,8 ms | **33,6 ms** | 9,4× |
| 10 Fragen, englisch | english | 709,2 ms | **83,4 ms** | 8,5× |
| 1 Frage, deutsch | multilingual | 37,4 ms | **6,0 ms** | 6,2× |
| 1 Frage, hindi | multilingual | 43,9 ms | **6,3 ms** | 7,0× |

Alle 17 Entscheidungen stimmen, die Zahlen darunter liegen höchstens 0,001 auseinander; der
Router entscheidet in allen fünf Fällen gleich. Jede Frage läuft in der kürzesten Länge des
Pakets, in die sie passt, hier 128 Token; mit fester Länge 512 waren es 17,5 bis 285,6 ms. Über
HTTP, beide Seiten im selben Lauf, liegt der Faktor zwischen 5,7 und 11,8. Gemessen auf Akku im Energiesparmodus. Mehrere
Fragen laufen nacheinander; warum der Stapel von Core ML wieder ausgebaut wurde, steht in
[laya-port.md](laya-port.md).

### 5.3 Wie laya und kev zueinander stehen

Zwei Datensätze, für die laya Zahlen veröffentlicht hat. Die kev- und laya-Spalten sind hier
selbst gemessen, die letzte Spalte ist das, was die jeweiligen Anbieter angeben.

| Datensatz | kev-Port | laya-Port | Jev 1.13 | laya veröffentlicht | Lage |
|---|---|---|---|---|---|
| DAIR Emotion | 49,7 % | **57,8 %** | 48,0 % | 57,3 % | für beide Modelle held-out |
| AG News | 90,1 % | **92,9 %** | 91,0 % | 94,7 % | in beiden Trainingsmischungen |

Beide Ports auf denselben ersten 1000 Zeilen des Test-Splits. Die eigenen laya-Zahlen liegen
nah an den veröffentlichten. Auf denselben Zeilen liefern laya und der Port zu 100 % dieselbe
Antwort, die Spalte gilt also für beide.

laya liegt auf beiden Datensätzen vorn. Der erste ist der belastbarere Vergleich, weil er in
keiner der beiden Trainingslisten steht.

Dafür kann kev, was laya nicht kann: mehrere Fragen auf einem Zustand in einem einzigen
Durchlauf, weil die block-kausale Maske die Zweige trennt und der Zustand nur einmal kodiert
wird. Bei laya hat jede Frage ihre eigene Sequenz, dort bleibt nur der Stapel. Bei drei Fragen
kostet das bei kev 16 ms statt 44 ms, bei laya bleibt es bei der Summe der Einzelläufe.

Beide Modelle liegen jetzt als Core ML vor und teilen sich Antworttypen, Anfrageformat und
Werkzeug. Welches für eine Aufgabe das richtige ist, ist damit eine Messfrage und keine
Architekturentscheidung mehr.

## 6. Kosten

Der Listenpreis von Jev liegt bei 0,042 USD je Million Eingabe-Tokens. Für den Banking77-Lauf
mit 3075 Zeilen und 976 Tokens im Mittel wären das rund 0,13 USD. Lokal fällt für die
Entscheidung selbst nichts an.

Damit verschiebt sich die Frage, die das Factsheet an die Kaskade stellt, sie verschwindet aber
nicht. Wenn die erste Stufe nichts kostet, lohnt eine Kaskade schon bei kleinen Genauigkeits-
gewinnen der zweiten Stufe. Was bleibt, sind die Kosten, die nicht im Token-Preis stehen:
1,1 GiB je fp16-Modell auf der Platte, zur Laufzeit dazu rund 14 GiB Temp-Dateien, 1,7 s Warmlauf, und die Arbeit an
Kriterien, Schwellen und Tests,
die das Factsheet zu Recht als den eigentlichen Aufwand benennt.

## 7. Grenzen dieses Ports

| | Jev | hier |
|---|---|---|
| Kontext gesamt | 64k Token | 1024 Token |
| davon Zustand | 32k Token | 384 Token im Training, mehr ist ungeprüft |
| Optionen je Frage | 255 | 96 im breiten Export, 8 im schnellen |
| Fragen je Aufruf | nicht begrenzt dokumentiert | 4 |
| Eingabearten | Text, JSON, Liste | gleich |
| Rate Limits | 250k Token/s, 1200 Anfragen/min | keine |

Die Kontextgrenze ist der harte Unterschied und liegt nicht am Export: `kev.model.MAX_STATE`
ist 384, darauf wurde trainiert. Die Optionsgrenze dagegen ist eine reine Shape-Frage beim
Export.

## 8. Wie die Zahlen entstehen

```bash
cd Exporter
./.venv/bin/python verify_coreml.py --model ../Models/Kev06B-fp16.mlpackage
./.venv/bin/python eval_suite.py     --base http://127.0.0.1:8010 --out ../Benchmarks/eval-nativ-dev.json \
                                     --dump-predictions ../Benchmarks/pred-nativ.jsonl
./.venv/bin/python bench_banking77.py --base http://127.0.0.1:8010 --task intent77 --describe \
                                      --out ../Benchmarks/bench-intent77-nativ.json
./.venv/bin/python compare_servers.py --native http://127.0.0.1:8008 --reference http://127.0.0.1:8009
./.venv/bin/python compare_predictions.py --a ../Benchmarks/pred-nativ.jsonl \
                                          --b ../Benchmarks/pred-ref.jsonl \
                                          --out ../Benchmarks/vergleich-je-frage.json
```

Die Rohergebnisse liegen als JSON unter `Benchmarks/`.
