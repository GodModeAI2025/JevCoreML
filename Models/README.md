# Models

Hier liegen die Core-ML-Modelle. Im Repo steht nur diese Datei, die Modelle selbst kommen aus
dem GitHub-Release:

```bash
scripts/fetch-models.sh          # laya, drei Checkpoints, rund 2,2 GB
scripts/fetch-models.sh kev      # kev, drei Längen, rund 3,3 GB
scripts/fetch-models.sh alle     # beides
```

| Datei | Modell | Eingabelängen | Größe |
|---|---|---|---|
| `Laya-EN-L512-K512-fp16.mlpackage` | laya, english | 128, 256, 512 | 805 MiB |
| `Laya-ML-L1024-K1024-fp16.mlpackage` | laya, multilingual | 128 bis 1024 | 615 MiB |
| `Laya-TD-L1024-K1024-fp16.mlpackage` | laya, typed-decisions | 128 bis 1024 | 805 MiB |
| `Kev06B-Q4-fp16.mlpackage` | kev, bis 4 Fragen und 8 Optionen je Durchlauf | 512 | 1,1 GiB |
| `Kev06B-L256-Q4-fp16.mlpackage` | kev, kurze Anfragen | 256 | 1,1 GiB |
| `Kev06B-L1024-Q4K96-fp16.mlpackage` | kev, bis 96 Optionen | 1024 | 1,1 GiB |
| `laya-tokenizer-english.json` | Tokenizer für english und typed-decisions | | 3,4 MB |
| `laya-tokenizer-multilingual.json` | Tokenizer für multilingual | | 33 MB |
| `tokenizer.json` | Tokenizer für kev | | 11 MB |

Herkunft und Lizenzen der Gewichte stehen in `NOTICE`.
