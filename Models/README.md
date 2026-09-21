# Models

Hier liegen die Core-ML-Modelle. Im Repo steht nur diese Datei, die Modelle selbst kommen aus
dem GitHub-Release:

```bash
scripts/fetch-models.sh          # laya, drei Checkpoints, rund 2,2 GB
scripts/fetch-models.sh kev      # kev, ein Paket für alle Längen, rund 1,1 GB
scripts/fetch-models.sh alle     # beides
```

| Datei | Modell | Eingabelängen | Fragen | Optionen | Größe |
|---|---|---|---|---|---|
| `Laya-EN-L512-K512-fp16.mlpackage` | laya, english | 128, 256, 512 | 1 | 512 | 805 MiB |
| `Laya-ML-L1024-K1024-fp16.mlpackage` | laya, multilingual | 128 bis 1024 | 1 | 1024 | 615 MiB |
| `Laya-TD-L1024-K1024-fp16.mlpackage` | laya, typed-decisions | 128 bis 1024 | 1 | 1024 | 805 MiB |
| `JevCoreML.mlpackage` | kev-0.6b, alle Anwendungsfälle | 128, 256, 512, 1024, 2048, 3072 | 8 je Durchlauf | 256 | 1,1 GiB |
| `laya-tokenizer-english.json` | Tokenizer für english und typed-decisions | | | | 3,6 MB |
| `laya-tokenizer-multilingual.json` | Tokenizer für multilingual | | | | 34 MB |
| `tokenizer.json` | Tokenizer für kev | | | | 11 MB |

Das kev-Paket nimmt sechs Längen an, und die Laufzeit rechnet jede Anfrage in der kürzesten,
in die sie passt; es gibt nichts auszuwählen. Die drei getrennten Zuschnitte aus dem Release
`models-v1` (`Kev06B-Q4-fp16`, `Kev06B-L256-Q4-fp16`, `Kev06B-L1024-Q4K96-fp16`) laufen
weiter; wer sie braucht, holt sie mit
`JEV_RELEASE=models-v1 JEV_WANT="Kev06B-Q4-fp16 Kev06B-L256-Q4-fp16 Kev06B-L1024-Q4K96-fp16" scripts/fetch-models.sh`.

Herkunft und Lizenzen der Gewichte stehen in `NOTICE`.
