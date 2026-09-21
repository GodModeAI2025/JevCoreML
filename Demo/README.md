# Jev Demo

Eine macOS-App in SwiftUI, die beide Modelle ausprobiert: Text eingeben, Fragensatz wählen,
⌘↩. Sie zeigt jede Antwort mit ihren Wahrscheinlichkeiten, bei laya dazu den gewählten
Checkpoint samt Begründung und die Handlungswahrscheinlichkeit.

![Die Demo-App](../docs/assets/demo-hell.png)

## Starten

```bash
scripts/fetch-models.sh alle     # einmal, im Wurzelordner des Repos: laya und kev, rund 3,3 GB
open Demo/JevDemo.xcodeproj
```

In Xcode das Schema `JevDemo` starten. Die App sucht die Modelle in dieser Reihenfolge: zuletzt
gewählter Ordner, Umgebungsvariable `JEV_MODELS`, das eigene App-Bundle, der Ordner `Models` des
Repos, aus dem sie gebaut wurde. Über „Ordner wählen …“ geht auch jeder andere.

Was sie kann:

- **laya und kev** nebeneinander, mit denselben Fragen.
- **Fünf Fragensätze** aus `LayaPresets`: Support-Triage, E-Mail, Guardrails, Moderation,
  Modellwahl. Dazu eine eigene Auswahlfrage, eine Option je Zeile als `name: Beschreibung`.
- **Beispiele** auf Deutsch, Englisch und Hindi, damit der Router etwas zu tun hat.
- **Ehrliche Zeiten.** Jede Entscheidung läuft viermal; gezeigt wird der Median der letzten
  drei, und der erste Lauf steht daneben, wenn er Laden und Vorbereiten enthielt.

Ein geladener kev-Export legt im Temp-Verzeichnis bis zu 14 GB ab, solange die App läuft. Die
App prüft vorher, ob so viel frei ist.

## Ohne Fenster

```bash
JevDemo.app/Contents/MacOS/JevDemo --selbsttest [Modellordner]
JevDemo.app/Contents/MacOS/JevDemo --bildschirmfoto bild.png [Modellordner]
```

`--selbsttest` rechnet die Triage mit laya und kev und gibt die Antworten aus, `--bildschirmfoto`
rendert das Fenster nach einer echten Entscheidung in eine PNG-Datei. Daher stammt das Bild oben.

## Projektdatei

`JevDemo.xcodeproj` ist mit [XcodeGen](https://github.com/yonaskolb/XcodeGen) aus `project.yml`
erzeugt und liegt fertig im Repo. Wer `project.yml` ändert: `xcodegen generate` im Ordner `Demo`.

Die App läuft ohne Sandbox, damit sie Modelle aus jedem Ordner lesen kann. Eine App für den App
Store legt die Modelle stattdessen ins Bundle und lädt sie mit `LayaRouted.bundled()`.
