# laya nativ: Port nach Core ML

[laya](https://github.com/NandhaKishorM/laya) ist ein Entscheidungsmodell wie kev, aber anders
gebaut. Dieser Port bringt es auf dieselbe Grundlage wie kev: ein `.mlpackage`, ein Tokenizer in
Swift, kein Python zur Laufzeit, keine Cloud. Wer es als Dienst braucht, startet es mit
`--serve` lokal hinter derselben HTTP-Schnittstelle wie kev.

Die Frage war zweiteilig. Funktioniert der Port genauso, und ist er mindestens so schnell.
Beides wird hier gemessen, nicht behauptet.

## Was laya anders macht als kev

| | kev | laya |
|---|---|---|
| Rückgrat | Qwen3-0.6B-Base, Decoder | ModernBERT-large bzw. mmBERT-base, Encoder |
| Richtung | kausal, Zustand vorn | bidirektional, Zustand hinten |
| Ablesestelle | Pointer-Kopf auf Optionstoken | MLP auf `[MASK]`-Marken |
| Mehrere Fragen | eine Sequenz, block-kausale Maske | eine Sequenz je Frage, gebündelt |
| Zweiter Kopf | keiner | Handlungskopf, gibt `act_probability` |
| Konfidenz | eigene Formel je Fragetyp | normalisierte Entropie, `1 − H(p)/log k` |
| Temperatur | keine | je Fragetyp und Optionszahl |
| Sprachen | ein Modell | drei Checkpoints plus Router |

Die Sequenz sieht so aus:

```
[CLS] "<typ> question: <anweisung>" [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] zustand [SEP]
```

Gelesen wird an den `[MASK]`-Positionen, eine je Option. Das ist strukturell dasselbe wie kevs
Pointer-Ablesung, nur dass hier ein kleiner MLP je Marke einen Logit erzeugt statt eines
Skalarprodukts gegen einen Entscheidungsvektor.

## Der Vertrag

```
input_ids      [1, L]  int32   L = 128, 256, … bis 512 bzw. 1024, alle in einem Paket
attention_mask [1, L]  int32
marker_pos     [K]     int32   Position je Option, ungenutzte Plätze zeigen auf 0
marker_mask    [K]     int32   1 für belegte Optionsplätze, 0 sonst
qtype          [1]     int32   0 choice, 1 score, 2 noul
->
logits         [K]     float32  unbelegte Plätze auf −1e4, wie in laya
act_logits     [2]     float32  Handlungskopf, roh
```

Alles mit Gewichten rechnet der Graph, auch der Handlungskopf samt seiner vier abgeleiteten
Merkmale. Draußen bleibt nur, was Konfiguration ist: die Temperaturtabelle und die
Entropiekonfidenz. Beides steht in den Metadaten des Pakets, damit Swift keine Beidatei braucht.

Drei Exporte, einer je Checkpoint:

| Datei | Checkpoint | Rückgrat | L | K | Größe |
|---|---|---|---|---|---|
| `Laya-EN-L512-K512-fp16.mlpackage` | english | ModernBERT-large | 128, 256, 512 | 512 | 805 MiB |
| `Laya-TD-L1024-K1024-fp16.mlpackage` | typed-decisions | ModernBERT-large | 128 bis 1024 | 1024 | 805 MiB |
| `Laya-ML-L1024-K1024-fp16.mlpackage` | multilingual | mmBERT-base | 128 bis 1024 | 1024 | 615 MiB |

K ist gleich L, und das ist Absicht. laya hat keine Obergrenze für Optionen. Es kürzt jede Option
bis auf vier Token und nimmt, was vor `max_len` passt: gemessen 169 Optionen auf dem englischen
Checkpoint, 257 auf dem mehrsprachigen und 340 auf typed-decisions, jeweils mit kurzen Namen
ohne Beschreibung. Die erste Fassung des Ports hatte K=32 und lehnte alles darüber ab.
Aufgefallen ist das erst beim Server, als eine Frage mit 33 Optionen 422 bekam, die laya ohne
Weiteres beantwortet. Mehr Marken als Positionen kann es nicht geben, also deckt K = L jeden Fall
ab. Der Scorer rechnet dann auf 512 oder 1024 Plätzen statt auf 32. Gerechnet ist das weniger
als ein Prozent des Encoders; gemessen geht der Unterschied im Rauschen zwischen zwei Läufen
unter. Die Pakete sind gleich groß geblieben, und auf den alten Referenzanfragen liefern alte und
neue Exporte dieselben Zahlen bis zur letzten Stelle. Berichte unter `Benchmarks/` von vor der
Umstellung nennen noch die Exporte mit K=32; wo sie zitiert werden, sind sie nachgemessen.

Passen die Optionen nicht vor `max_len`, bricht laya ab („options exceed head_max_len“) und der
Port ebenso, mit derselben Zahl passender Marken. Die Grenze von 255 Optionen, die kev beim Lesen
einer Anfrage setzt, gilt für laya nicht.

### Mehrere Längen in einem Paket

laya rechnet jede Frage nur so lang, wie sie ist: `system_one` polstert auf die längste Sequenz
im Stapel, eine einzelne Frage also gar nicht. Der Port füllte anfangs jede Frage auf 512 oder
1024 Token auf, und eine typische Frage hat 50 bis 100. Jedes Paket nimmt deshalb jetzt mehrere
Längen an, verdoppelnd ab 128, und die Laufzeit nimmt die kürzeste, in die eine Frage passt. Auf
AG News liefen so 680 von 1000 Zeilen in 128 Token und der Rest in 256.

Getrennte Pakete je Länge hätten für jede Länge eine volle Kopie der Gewichte gekostet. Eine
einzige Instanz, die zwischen den Längen wechselt, ist gemessen unbrauchbar: Core ML bereitet den
Graphen auf der GPU bei jedem Wechsel neu vor, das kostet rund 450 ms statt 8 bis 12 ms, und
keiner der Hinweise in `MLOptimizationHints` ändert daran etwas. Die Laufzeit hält deshalb eine
Instanz je Länge, angelegt beim ersten Gebrauch, und jede sieht nur ihre eine Länge. Das ist
billig. Beim englischen Checkpoint wuchs der Speicher mit drei Längen statt einer von 245 auf
320 MB, und die versteckten Temp-Dateien von Core ML blieben bei 0,85 GiB, denn die legt es
einmal je Paket an und nicht je Instanz. Wer trotzdem sparen will, schränkt die Längen mit
`LayaRuntime.Configuration(sequenceLengths:)` oder `--lengths` ein; die größte ist immer dabei.

Der erste Export mit mehreren Längen rechnete merklich ungenauer, typed-decisions lag 7,9e−02
in den Wahrscheinlichkeiten neben laya. Warum, steht unter den Befunden.

## Funktionale Gleichheit

Geprüft auf sechs Ebenen, weil ein Fehler auf jeder einzelnen lautlos bleibt.

**Tokenizer.** Die drei Checkpoints benutzen zwei verschiedene Tokenizer. Beide sind in Swift
nachgebaut und gegen aus Python gezogene IDs geprüft: die handverlesenen Fälle plus ein
Suchlauf über 14.960 beziehungsweise 16.278 Fälle. Null Abweichungen.

**Encoder.** Die Sequenz selbst, Token für Token, gegen `laya.common.build_sequence`. Dazu
gehören die Kürzungsregeln: erst jede Option auf 48 Token, wenn das Kopfbudget dann noch nicht
reicht gleichmäßig weiter, danach die Anweisung, zuletzt der Zustand von rechts. Und das
Entschärfen: steht der Maskentext selbst in Anweisung, Option oder Zustand, wird er zu einem
Leerzeichen. Welcher Text das ist, hängt am Checkpoint, `[MASK]` bei ModernBERT, `<mask>` bei
mmBERT. Die erste Fassung des Ports hatte `[MASK]` fest verdrahtet und hätte auf dem
mehrsprachigen Checkpoint aus jedem `<mask>` im Zustand ein echtes Markertoken gemacht. Zwei
Referenzanfragen enthalten jetzt beide Formen, und der Text kommt aus den Metadaten des Pakets.

**Nachverarbeitung.** Temperatureimer, Softmax, Entropiekonfidenz, Erwartungswert bei `score`,
`p[1]` bei `noul`, `max(p, 1−p)` als Ja/Nein-Konfidenz, `act_probability` aus dem zweiten Kopf.
Mit den Logits aus der Referenz gerechnet, damit ein Fehler hier nicht als Modellfehler
durchgeht.

**Modell.** Core ML gegen PyTorch über die Referenzanfragen und über zwei öffentliche
Datensätze. „Gleiche Antwort“ heißt: bei `choice` gewinnt dieselbe Option, bei `noul` liegt p
auf derselben Seite von 0,5, bei `score` rundet der Erwartungswert auf dieselbe Stufe. Wie weit
die Zahlen darunter auseinanderliegen, steht in der letzten Spalte.

| Prüfung | n | gleiche Antwort | max ∣Δp∣ |
|---|---|---|---|
| Referenzanfragen, english | 15 | 15/15 | 2,7e−03 |
| Referenzanfragen, multilingual | 15 | 15/15 | 2,3e−03 |
| Referenzanfragen, typed-decisions | 15 | 15/15 | 1,6e−03 |
| AG News, english | 1000 | 1000/1000 | 3,6e−03 |
| DAIR Emotion, english | 1000 | 999/1000 | 2,3e−02 |

Die eine abweichende Emotion-Zeile ist ein Gleichstand: „i feel like i m going to struggle and
fail and suffer and be really dumb“, laya wählt `sadness` mit 0,4978, der Port `fear` mit 0,4977.
Zwei Klassen liegen dort ein Tausendstel auseinander, und fp16 entscheidet, welche vorn liegt.
Mit dem Export fester Länge fiel dieselbe Zeile noch auf laya's Seite; das Rauschen hängt an der
Form, in der gerechnet wird. Die Trefferquote ist auf AG News gleich, 92,9 % auf beiden Seiten,
auf Emotion 57,9 % für laya und 57,8 % für den Port. Veröffentlicht hat laya 94,7 % und 57,3 %;
die Abweichung liegt im Rahmen dessen, was andere Zeilen bei n=1000 ausmachen.

Zu den Referenzanfragen gehören je eine Frage mit 150 bis 300 Optionen und eine mit 100 bis 200
Stufen. Die mit 200 Stufen auf dem mehrsprachigen Checkpoint lag mit dem Export fester Länge
0,02 neben laya, 97,5024 gegen 97,4815, und rundete deshalb auf eine andere Stufe. Mit dem
jetzigen Paket liegt sie 0,0065 daneben und auf derselben.

**Router.** Die Schrifterkennung entscheidet, welcher Checkpoint rechnet, und ein falscher
Checkpoint kostet mehr als jede Rechenungenauigkeit. Der Port ist gegen laya auf 38 Erkennungen,
70 Routen und 8 Vorrangfällen geprüft, alle identisch: dieselbe erkannte Schrift, derselbe
Sprachtipp, dieselben Anteile je Schrift bis aufs Bit und in derselben Reihenfolge, derselbe
Checkpoint. Acht der Erkennungen sind Gleichstände zwischen Schriften, warum, steht unter den
Befunden.

Was das bringt, an den eigenen deutschen Referenzanfragen gemessen: dieselbe Frage, derselbe
Zustand, einmal englischer und einmal mehrsprachiger Checkpoint.

| Frage | english | multilingual |
|---|---|---|
| `skill` (5 Optionen) | 0,0934 | 0,9501 |
| `route` (4 Optionen) | 0,1910 | 0,4975 |
| `severity` (5 Stufen) | 0,0542 | 0,2226 |

Beide wählen bei `skill` dasselbe, aber der englische Checkpoint weiß es nicht: 0,09 Konfidenz
gegen 0,95. Genau das ist laya's Argument für drei Checkpoints, und es hält auch auf unserem
eigenen Korpus.

**Antwort über HTTP.** `compare_laya_server.py` schickt dieselben Anfragen an `Router.predict` und
an `jev --engine laya --serve` und vergleicht die Antworten Schlüssel für Schlüssel. Einzelheiten
unter „Als HTTP-Dienst“.

## Geschwindigkeit

Gemessen wird auf derselben Maschine, mit denselben Anfragen, über dieselben fünf Fälle, die
`laya_baseline.py` fährt. laya läuft als PyTorch auf der CPU mit `Router(preload=True)`, der
Port als Release-Build von `jev --engine laya` auf `cpuAndGPU`. Beide Seiten messen den Median,
laya über 20 Läufe, der Port über 25. Die Port-Spalte ist mit den Paketen mehrerer Längen
nachgemessen, auf einem M5 Max auf Akku im Energiesparmodus; die laya-Spalte stammt aus dem früheren Lauf von
`laya_baseline.py`, unter welchen Bedingungen genau, ist nicht festgehalten. Beide Seiten im selben
Lauf unter denselben Bedingungen misst die Tabelle unter „Als HTTP-Dienst“.

| Fall | Checkpoint | laya | Port | Faktor | Port, schnellster |
|---|---|---|---|---|---|
| 1 Frage, englisch | english | 112,9 ms | **9,0 ms** | 12,5× | 8,1 ms |
| 4 Fragen, englisch | english | 314,8 ms | **33,6 ms** | 9,4× | 32,4 ms |
| 10 Fragen, englisch | english | 709,2 ms | **83,4 ms** | 8,5× | 82,5 ms |
| 1 Frage, deutsch | multilingual | 37,4 ms | **6,0 ms** | 6,2× | 4,3 ms |
| 1 Frage, hindi | multilingual | 43,9 ms | **6,3 ms** | 7,0× | 4,6 ms |

Mit fester Länge waren es 17,5, 113,7, 285,6, 9,6 und 10,3 ms. Alle fünf Fälle laufen jetzt in
128 Token statt in 512 oder 1024, so wie laya sie auch nur so lang rechnet, wie sie sind.

Der Router entscheidet in allen fünf Fällen gleich, und alle 17 Entscheidungen stimmen: dieselbe
Auswahl, bei `noul` dieselbe Seite von 0,5, bei `score` dieselbe gerundete Stufe, dieselbe
`act_probability`. Die Zahlen darunter liegen nicht auf vier Stellen gleich, wie hier zuerst
stand; die größten Abstände sind 0,0010 in der Konfidenz und 0,0003 im Wert, das ist fp16.
Tabelle und Vergleich schreibt `compare_laya_port.py` nach `Benchmarks/laya-vergleich.json`.

laya's README nennt für eine Frage auf einer T4 39,5 ms mit dem englischen und 32,8 ms mit dem
mehrsprachigen Checkpoint; die 33 ms aus der Überschrift sind der zweite Wert. Auf dieser Maschine
braucht laya auf der CPU für dieselben Fälle 112,9 und 37,4 ms. Für die Frage, ob der Port
mithält, zählt nur die Zahl auf derselben Hardware.

### Zwei Dinge, die beim Messen auffielen

Die erste Grundlinie wurde verworfen. Sie meldete für die deutsche Anfrage eine Konfidenz von
0,3133, ein zweiter Lauf desselben laya auf derselben Maschine ergab 0,0642. Der Port liefert
0,0636, deckt sich also mit dem zweiten Lauf, und alle drei Antworten wählen dasselbe. Warum
laya zwischen zwei Läufen eine andere Konfidenz meldet, ist nicht geklärt; der erste Lauf fand
statt, während die Checkpoints noch geladen wurden, das ist aber eine Vermutung und keine
Erklärung. Berichtet wird der reproduzierbare Lauf.

Core ML bricht unter Last gelegentlich ab. Dreimal in dieser Arbeit endete `predict` in einer
Ausnahme ohne Beschreibung: zweimal, während ein zweiter Prozess dieselbe GPU belegte, einmal im
Swift-Test und einmal mitten in einem Korpuslauf über 1000 Zeilen, und einmal unmittelbar nach
einem vorangegangenen Testlauf, dessen Prozess die GPU noch freigab. Sechs Wiederholungen
desselben Tests ohne Vorgänger liefen danach alle durch. Der Fehler liegt in Core ML, nicht im
Port, und er wird durchgereicht statt verschluckt. Wer Messungen parallel oder dicht
hintereinander fährt, sollte damit rechnen.

### Wo der Abstand am kleinsten ist

Auf dem englischen Checkpoint bei mehreren Fragen. laya bündelt alle Sequenzen in einen Durchlauf, der Port rechnet sie
nacheinander, denn Anweisung und Optionen stehen mit im Text und jede Frage hat damit ihre
eigene Sequenz. Ein gemeinsamer Durchlauf wie bei kev geht hier nicht.

Die naheliegende Entsprechung wäre `MLModel.predictions(fromBatch:)`, und die war eine Zeit lang
eingebaut. Gemessen brachte sie bei vier Fragen nichts am Median und bei zehn Fragen gut ein
Fünftel: 267,7 ms gegenüber dem damaligen Median von 332,5 ms ohne Stapel. Sie ist trotzdem wieder draußen. Ein eigens dafür geschriebener
Test, der den Stapel gegen Einzelläufe hält, beendete einen von vier Läufen mit Signal 6: eine
Objective-C-Ausnahme aus der E5-Laufzeit von Core ML, „MPSGraph tensor shape is missing or has
unexpected rank“, bei unveränderten Eingaben. Die Ausnahme erreicht kein `catch` in Swift, der
Prozess endet einfach. Das ist teurer als ein Fünftel bei zehn Fragen, zumal der Port auch ohne
Stapel achtmal so schnell ist wie laya.

Ein Export mit echter Batch-Dimension, vier Zeilen statt einer, ist ebenfalls gemessen. Er rechnet
richtig und bringt 11 %: 141,0 gegen 158,4 ms für vier Fragen, beides über coremltools. Bei
L=512 ist die GPU mit einer Sequenz schon ausgelastet, mehr Zeilen je Aufruf ändern daran wenig.

Mit fester Länge fiel der Abstand zwischen Median und schnellstem Lauf auf: 113,7 gegen 55,0 ms
bei vier Fragen. Mit den kürzeren Längen ist er fast verschwunden, 33,6 gegen 32,4 ms. Vorher
Vorhersage für Vorhersage gemessen, zeigte sich, woher er kam. Die erste einer Reihe
braucht 11 ms, unter Dauerlast wird jede weitere etwas langsamer, nach einigen Dutzend liegen sie
bei 26 bis 29 ms und bleiben dort. Pausen von 50 ms dazwischen holen die 11 ms nicht zurück. Vier
Fragen je Lauf heißen viermal so viele Vorhersagen in derselben Reihe, also mehr davon im
langsamen Bereich, und der schnellste Lauf ist einer vom Anfang. Gemessen wurde auf Akku im
Energiesparmodus (`pmset`: powermode 1), der den Takt unter Last vermutlich drückt. Mit einem
Viertel der Arbeit je Vorhersage kommt der Rechner offenbar kaum noch in diesen Bereich. Am
Netzteil ohne Energiesparmodus ist das nicht nachgemessen. Berichtet wird der Median, weil
laya's Messskript ihn auch berichtet, und er ist die vorsichtigere Zahl.

## Befunde, die den Port fast lautlos falsch gemacht hätten

### Die Neural Engine rechnet die festen Exporte falsch

Bei fp16 liefern die drei Rechenwerke drei verschiedene Ergebnisse. Gemessen am englischen Export
mit fester Länge L=512 gegen PyTorch, alle drei in einem Lauf über dieselben 15
Referenzanfragen (`verify_laya_coreml.py --units all --timing 25`). Die Zeiten sind über
coremltools gemessen, also mit dem Aufwand von Python; die Zahlen des Swift-Ports stehen unter
„Geschwindigkeit“.

| Rechenwerk | max ∣Δlogit∣ | max ∣Δp∣ | gekippte Entscheidungen | Median |
|---|---|---|---|---|
| CPU | 1,5e−01 | 1,1e−02 | 0 von 15 | 224,7 ms |
| GPU | 2,4e−02 | 2,8e−03 | 0 von 15 | 32,5 ms |
| Neural Engine | 5,0e+00 | 9,9e−01 | 9 von 15 | 92,4 ms |

„Gekippt“ heißt: bei `choice` gewinnt eine andere Option, bei `noul` liegt p auf der anderen
Seite von 0,5, bei `score` rundet der Erwartungswert auf eine andere Stufe. Die Neural Engine
wählt bei `skill` `research` statt `excel` und bei elf Optionen `tool_9` statt `calendar`. Kein
Fehler, keine Warnung, plausibel aussehende Antworten. Und sie ist dabei fast dreimal langsamer
als die GPU. Mit K=32 sah es genauso aus, 7 von 13.

Das Paket mit mehreren Längen verhält sich anders, dieselbe Messung daran (Bericht in
`Benchmarks/laya-backends-EN.json`):

| Rechenwerk | max ∣Δlogit∣ | max ∣Δp∣ | gekippte Entscheidungen | Median |
|---|---|---|---|---|
| CPU | 1,5e−01 | 1,1e−02 | 0 von 15 | 85,7 ms |
| GPU | 2,5e−02 | 2,7e−03 | 0 von 15 | 6,3 ms |
| Neural Engine | 1,4e−02 | 1,3e−03 | 0 von 15 | 268,4 ms |

Hier rechnet die Neural Engine richtig, braucht dafür aber das Vierzigfache der GPU. Warum sie
beim festen Export falsch lag und hier nicht, ist nicht geklärt; an der Entscheidung ändert es
nichts.

`.all` überlässt die Wahl dem Planer von Core ML. Der hat im Test zwischen 22 ms und 115 ms
geschwankt, je nachdem, worauf er das Modell gerade legt. Deshalb ist die Vorgabe der laya-
Laufzeit `.cpuAndGPU`, und `.cpuAndNeuralEngine` wird abgelehnt, solange niemand ausdrücklich
`allowNeuralEngine: true` setzt.

### Eine voll maskierte Zeile ergibt NaN

Die ersten Exporte lieferten auf allen Rechenwerken NaN. Die Ursache liegt zwei Schritte
hintereinander:

ModernBERT füllt seine Aufmerksamkeitsmaske mit `torch.finfo(float32).min`. In fp16 wird daraus
`−inf`. Das allein wäre noch harmlos. ModernBERT rechnet aber jede dritte Schicht mit einem
gleitenden Fenster von 128 Token, und dessen Maske hängt anders als die globale von der Zeile
ab. Eine Auffüllposition, deren ganzes Fenster aus Auffüllung besteht, hat damit eine Zeile, in
der jeder Schlüssel maskiert ist. Softmax rechnet dort 0/0, und weil die Aufmerksamkeit
anschließend `0 · NaN` bildet, wandert das NaN über den Residualstrom in jede Position.

Erste betroffene Schicht ist die mit dem Index 1, die erste mit Fenster. Schicht 0 ist global
und daher unauffällig, was die Suche zunächst in die falsche Richtung lenkte.

Zwei Änderungen beheben das, und beide ändern am Ergebnis nichts: die Maske wird mit einem
endlichen Wert gefüllt, denn `exp(−1e4)` unterläuft exakt zu 0, und die Diagonale bleibt offen.
Für gültige Anfragen steht die Diagonale ohnehin auf 0; für Auffüllzeilen zählt das Ergebnis
nirgends, denn als Schlüssel sind sie überall maskiert.

Gefunden wurde das nicht durch Nachdenken, sondern durch Halbieren: erst Schicht für Schicht
durch den Encoder, dann Zwischenwert für Zwischenwert innerhalb der Schicht, jeweils auf allen
drei Rechenwerken. Die Tabelle, die es zeigte:

| Zwischenwert | PyTorch | CPU | GPU | ANE |
|---|---|---|---|---|
| Ausgabe Schicht 0 | 43,4 | 43,3 | 43,4 | 31,4 |
| nach `attn_norm` | 3,70 | 3,71 | 3,70 | 3,51 |
| nach `Wqkv` | 6,81 | 6,82 | 6,82 | 6,87 |
| **nach der Aufmerksamkeit** | **2,60** | **−65504** | **2,59** | **2,72** |

−65504 ist der kleinste fp16-Wert. So meldet ein `reduce_max` über lauter NaN, dass es nichts
gefunden hat.

### Foundation und Swift verändern Zeichenketten

Zwei Fehler, die nichts auslösen und nur ein Vergleich gegen eine Referenz findet. Beide
treffen nur einzelne Token des Wortschatzes, beide machen den Tokenizer an genau diesen Stellen
falsch.

`JSONSerialization` entfernt ein führendes U+FEFF aus jeder dekodierten Zeichenkette. Der
mmBERT-Wortschatz führt 256.000 Token, darunter `U+FEFF #` neben `#` und `U+FEFF \r` neben
`\r`. Beim Einlesen fielen sie paarweise zusammen, der zweite überschrieb den ersten. Sichtbar
wurde es an einer URL: das `#` bekam die ID von `U+FEFF #`.

Swift vergleicht Zeichenketten nach kanonischer Äquivalenz. Ein Wortschatz braucht aber
Byte-Identität. U+0341 und U+0301 sind kanonisch dasselbe, ebenso U+0340 und U+0300, U+2126 und
U+03A9, U+212B und U+00C5. Alle vier Paare stehen im mmBERT-Wortschatz als eigene Token. Als
`String` geschlüsselt fielen sie zusammen, und aus `▁` plus `Ω` wurde ein Merge, den es gar
nicht gibt.

Der Byte-Level-Wortschatz von kev und dem englischen laya ist davon nicht betroffen, und zwar
nicht aus Glück: dessen Alphabet besteht aus 256 druckbaren Zeichen, die weder kombinierende
Marken noch kanonische Singletons enthalten. Der Suchlauf bestätigt das, er findet dort 0 Fallen
gegenüber 661 im mmBERT-Wortschatz.

Beides ist jetzt behoben, beides bleibt geprüft: der Tokenizer-Suchlauf enthält gezielt jedes
Token mit einem kanonisch gleichwertigen Gegenstück und jedes, das mit U+FEFF oder einer
kombinierenden Marke beginnt. Der Byte-Level-Weg lehnt eine Datei mit U+FEFF ausdrücklich ab,
statt stillschweigend das Falsche zu tun.

### Unicode hat Versionen, und die Referenz ist nicht auf dem neuesten Stand

Eine Prüfung mit fünf unabhängigen Agenten hat danach zwei weitere Fehlerklassen gefunden, die
kein Satzkorpus traf, weil sie nicht an Sätzen hängen, sondern an einzelnen Codepunkten.

Die Regex-Engine hinter HuggingFace ist Oniguruma, nicht ICU. Für jeden der 1,1 Millionen
Codepunkte abgefragt, versteht Oniguruma unter `\p{L}` die Buchstaben von Unicode 16.0; ICU auf
diesem macOS kennt schon Unicode 17 und zählt 4662 neue Zeichen dazu. Dazu kommen private
Codepunkte, denen Apples ICU eigene Eigenschaften gibt: U+F8A1 bis U+F8A7 gelten dort als
Ziffern. Zusammen schnitt der Pretokenizer an 4693 Codepunkten anders als die Referenz. Die
Klassen im Port ziehen deshalb alles nach Unicode 16.0 und alle privaten Codepunkte ab,
berechnet aus der Unicode-Version und nicht als feste Liste.

Der NFC-Normalizer der Referenz ist noch älter. Die 814 kombinierenden Marken bis Unicode 9.0
kennt er alle, von den 154 danach keine einzige. Zeichen nach 9.0 behandelt der Port
deshalb wie die Referenz, als Starter ohne Zerlegung und Komposition. Für alles davor gilt
Foundation, und das ist exakt, weil Unicode die Normalisierung einmal zugewiesener Zeichen nie
mehr ändert.

Beim Nachprüfen kamen drei Fehler in Foundations NFC selbst dazu. Ein astraler Starter mit Marke
wird auf 16 Bit gekürzt, das war schon bekannt. Eine vorkomponierte Hangul-Silbe bekommt ihren
Schlusskonsonanten nicht angefügt: aus U+AC00 U+11A8 wird nicht 각. Und hinter einer zerlegten
Silbe verschluckt Foundation U+11A7, das nach dem Standard nie komponiert. Der Port zerlegt
deshalb erst, komponiert dann, prüft über die NFD, ob sich die Äquivalenzklasse verschoben hat,
und behandelt U+11A7 als Trennstelle.

Geprüft wird das jetzt Codepunkt für Codepunkt statt Satz für Satz: `UnicodeReferenceTests`
hält die Regex-Klassen für alle 1,1 Millionen Codepunkte gegen Oniguruma und NFC für 260.000
Einzelzeichen, 3688 Markenfolgen, 17.355 vorkomponierte Zeichen mit Marke und jeden astralen
Codepunkt mit Akut gegen den Normalizer der Referenz. Die Daten schreibt `dump_unicode_reference.py`.

### Eine variable Länge rechnet die Positionswinkel in fp16

Der erste Export mit mehreren Längen bestand die eigene Prüfung beim Export, eine Frage mit 50
Token, in jeder Länge. Gegen die 15 Referenzanfragen lag er deutlich schlechter als der feste:
typed-decisions 7,9e−02 in den Wahrscheinlichkeiten statt 3,9e−03, multilingual 0,98 in den
Logits statt 0,10, dort mit einer Stufe daneben.

Die Ursache ist die Rotary-Positionskodierung von ModernBERT. Jede Aufmerksamkeitsschicht rechnet
cos und sin der Winkel Position mal Frequenz. Bei fester Länge sind die Positionen eine Konstante,
und coremltools faltet die ganze Rechnung beim Übersetzen in fp32 und legt nur das Ergebnis ab.
Bei variabler Länge geht das nicht, dann läuft der Winkel zur Laufzeit durch den fp16-Graphen,
und fp16 löst einen Winkel um 1000 nur auf 0,5 genau auf. Der Export rechnet die Tabellen jetzt
einmal in fp32 bis zur größten Länge vor, und der Graph nimmt nur die ersten L Zeilen
(`patch_rotary_tables` in `layax.py`). Danach liegen alle drei Checkpoints so nah an laya wie
die festen Exporte oder näher, typed-decisions bei 1,6e−03.

### Ein Gleichstand zwischen zwei Schriften

`laya.lang.detect_script` zählt Buchstaben je Schrift und nimmt mit `max()` die häufigste. Bei
Gleichstand gewinnt in Python der erste Eintrag des dict, und laya trägt `latin` erst ein, nachdem
alle anderen Schriften gezählt sind. Ein Text, der halb lateinisch und halb kyrillisch ist, gilt
dort als kyrillisch und geht an den mehrsprachigen Checkpoint. Der Port hatte `latin` vorn und
schickte denselben Text an den englischen. Verwirrend daran: `script_profile` im selben Modul
führt `latin` tatsächlich vorn. Die beiden Funktionen sehen gleich aus und ordnen verschieden.

Keine der 108 Routingproben traf einen Gleichstand. Gefunden hat es der Vergleich des Servers mit
`Router.predict` an „Order 4411: заказ пришёл дважды, refund please“, 17 lateinische und 17
kyrillische Buchstaben. laya rechnete 55 Token auf dem mehrsprachigen Checkpoint, der Port 59 auf
dem englischen, und erst die abweichende Tokenzahl in `usage` zeigte, dass etwas nicht stimmt.
Die Routingreferenz enthält jetzt acht Gleichstände: Latein gegen eine andere Schrift, drei
Schriften zu gleichen Teilen, zwei nicht-lateinische Schriften in beiden Reihenfolgen. Die Anteile
je Schrift werden seitdem bis aufs Bit verglichen statt auf fünf Stellen, denn der Server gibt
sie aus.

## Was vom Port besser ist als vom Original

- **Rechenwerk festgenagelt statt ausgewürfelt.** laya wählt CPU oder CUDA, hier wird das
  Backend gewählt, das nachweislich richtig rechnet, und das falsche abgelehnt.
- **Ein Prozess, kein Python.** Laden und Rechnen stecken in einem Swift-Paket, das Modell ist
  eine Datei.
- **Der Tokenizer ist geprüft, nicht geliehen.** 31.238 Sätze über beide Wortschätze, dazu
  Regex-Klassen und NFC für jeden einzelnen Codepunkt gegen die Referenz.
- **Lazy geladen.** Wer nur englische Anfragen sieht, zahlt die 615 MiB für den mehrsprachigen
  Checkpoint nicht. Das gilt für die Bibliothek und die Kommandozeile, auch für die einzelnen
  Längen. `--serve` lädt und wärmt beim Start alle vorhandenen Checkpoints in allen Längen,
  damit keine Anfrage auf das Laden wartet; mit allen dreien sind das 2,4 GiB versteckte
  Temp-Dateien von Core ML.

## Presets

Die fünf Fragensätze aus `laya.presets` liegen als `LayaPresets` vor: Triage, Mail, Guardrails,
Moderation und Modellwahl, zusammen 24 Fragen. Abgeschrieben ist davon nichts.
`gen_laya_presets.py` erzeugt die Swift-Datei aus laya selbst und schreibt zu jedem Satz laya's
Antwort auf einen typischen Zustand mit. `LayaPresetTests` prüft beides: dass jede Frage dieselbe
Tokenfolge ergibt wie das Original, und dass der Port auf denselben Zuständen dieselben 24
Antworten gibt.

## Mailtexte

`LayaEmail` bildet `laya.email` nach: zitierten Verlauf, Signaturen und Haftungsausschlüsse
entfernen, dann Betreff, Text und Absender zu einem Zustand bündeln. Das Original ist
Python-Regex, der Port läuft über ICU, und die beiden meinen an vier Stellen Verschiedenes. `\s`
umfasst in Python genau die 29 Zeichen von `str.isspace`, ICU lässt U+001C bis U+001F und U+0085
weg. `\w` ist in Python genau `[\p{L}\p{N}_]`, ICU nimmt kombinierende Marken dazu. `$` und `.`
behandeln in ICU jeden Zeilentrenner als Zeilenende, in Python nur `\n`. Keines der Muster ist
deshalb wörtlich übernommen, die Zeichenklassen sind aus Python abgeleitet und ausgeschrieben.

Geprüft über 4043 Mails: 43 handverlesene, darunter Unicode-Leerraum, Zeilentrenner, `ſ` und das
Kelvinzeichen unter Groß-/Kleinschreibung und die Signaturgrenze bei 60 % der Zeilen, dazu 4000
aus Zeilenbausteinen erzeugte. Jede Mail zweimal, mit 3000 und mit 200 Zeichen Obergrenze. Alle
8086 Ergebnisse sind byte-identisch.

## Als HTTP-Dienst

laya hat keinen Server, nur `Router.predict` in Python. Der Port bietet denselben Aufruf über HTTP
an, mit demselben Server wie kev:

```
jev --engine laya --models Models --serve --port 8008
```

Die Anfrage ist die von kev, `state` und `questions`, dazu die drei Hinweise, die
`Router.predict` kennt. `task` und `lang` stehen als eigene Felder in der Anfrage. `model` wirkt
nur, wenn es einen Checkpoint nennt, also `english`, `multilingual`, `typed-decisions` oder einen
der Aliasnamen aus `laya.router`; `laya` heißt dort wie hier der englische. Der TypeSafe-Client
schickt in diesem Feld `kev-latest`, und das soll die Schrifterkennung nicht abschalten.
`--checkpoint` und `--lang` auf der Kommandozeile gelten für Anfragen ohne eigenen Hinweis.
`/v1/models` nennt die Checkpoints als Aliasnamen.

Die Antwort ist, was `Router.predict` zurückgibt, als JSON: dieselben Schlüssel in derselben
Reihenfolge, Zahlen auf vier Stellen gerundet wie in laya, `usage` mit den Token aller Sequenzen
und null Ausgabetoken, `routing` mit Modell, Repo, Erkennung und Ablauf. Der Server hängt wie bei
kev `latency_ms` und `passes` an.

```json
{"model": "laya-rl-agent",
 "answers": {"churn": {"type": "noul", "noul": 0.9043, "confidence": 0.9043,
                       "action": {"act_probability": 1.0}}},
 "usage": {"input_tokens": 52, "output_tokens": 0}, "latency_ms": 36.01, "passes": 1,
 "routing": {"model": "english", "repo": "convaiinnovations/laya",
             "reason": "englischer Text in lateinischer Schrift",
             "detection": {"script": "latin", "script_profile": {"latin": 1.0}, "language": "en",
                           "is_english": true, "non_latin_fraction": 0.0},
             "workflow": null}}
```

Was bewusst anders ist:

- `reason` ist deutsch, wie jede Meldung des Ports.
- `repo` ist immer eine Zeichenkette. laya legt beim automatisch erkannten typed-decisions-Ablauf
  das rohe Tupel aus Repo und Unterordner ab, das im JSON zur Liste wird; überall sonst schreibt
  es selbst `repo/unterordner`.
- `/v1/systemone/separate` rechnet und routet jede Frage für sich und übernimmt das `routing` der
  ersten. Ohne automatische Ablauferkennung, die laya nur auf Wunsch einschaltet und der Server
  gar nicht, entscheidet allein der Zustand, und der ist für alle Fragen derselbe.

`compare_laya_server.py` prüft das gegen laya selbst: 14 Anfragen, die fünf Messfälle und neun
weitere für die Hinweise, Legenden aus Zahlen, `null` und Objekten, einen Zustand ohne Buchstaben,
gemischte Schrift, 150 Optionen und 300 Optionen auf typed-decisions. In allen 14 ist die Form
gleich, Schlüssel für Schlüssel und in derselben Reihenfolge, und alle 26 Entscheidungen stimmen.
Die größten Abstände sind 0,0021 in einer Wahrscheinlichkeit und 0,0010 in der Konfidenz, das ist
fp16. `LayaServerTests` hält Form, Rundung und Routing ohne Python fest.

Die Zeiten dazu, beide Seiten im selben Lauf auf einem M5 Max auf Akku im Energiesparmodus, laya
im eigenen Prozess, der Port von außen über HTTP samt Verbindungsaufbau und JSON:

| Fall | laya im Prozess | Port über HTTP | Faktor |
|---|---|---|---|
| 1 Frage, englisch | 114,9 ms | **9,8 ms** | 11,8× |
| 4 Fragen, englisch | 317,4 ms | **51,2 ms** | 6,2× |
| 10 Fragen, englisch | 710,6 ms | **123,8 ms** | 5,7× |
| 1 Frage, deutsch | 38,5 ms | **5,4 ms** | 7,1× |
| 1 Frage, hindi | 44,9 ms | **7,7 ms** | 5,8× |

Allein gemessen, ohne laya dazwischen, braucht der Server für zehn Fragen 92 ms über HTTP und
89 ms intern, fast so viel wie die Kommandozeile mit 83 ms. Im gemeinsamen Lauf kostet laya's
CPU-Last zwischen den Anfragen den Port Zeit, vermutlich über Wärme und Energiesparmodus.
Anfangs brauchte der Server auch allein 121 ms intern: seine Queue und Tasks liefen ohne
Priorität, und der CPU-Anteil jeder Vorhersage landete auf den Effizienzkernen. Beide laufen
jetzt mit `.userInitiated`.

## Was der Port nicht hat

- **Kein Training, keine Feinjustierung.** Wie beim kev-Port: der Export nimmt das Modell, wie
  es ist.
