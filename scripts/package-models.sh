#!/usr/bin/env bash
# Packt die Core-ML-Modelle aus Models/ als ZIP-Dateien nach dist/, eine je Paket, dazu die
# Tokenizer und Prüfsummen. Das sind die Dateien für das GitHub-Release models-v2, aus dem
# scripts/fetch-models.sh lädt. GitHub nimmt je Datei bis 2 GB, das größte Paket hat 1,1 GB.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-Laya-EN-L512-K512-fp16 Laya-ML-L1024-K1024-fp16 Laya-TD-L1024-K1024-fp16 JevCoreML}"
OUT="${OUT:-dist}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

for m in $MODELS; do
  [ -d "Models/$m.mlpackage" ] || { echo "fehlt: Models/$m.mlpackage" >&2; exit 1; }
  echo "packe $m"
  rm -f "$OUT/$m.mlpackage.zip"
  ditto -c -k --keepParent "Models/$m.mlpackage" "$OUT/$m.mlpackage.zip"
done
echo "packe Tokenizer"
rm -f "$OUT/tokenizers.zip"
(cd Models && zip -q "$OUT/tokenizers.zip" tokenizer.json laya-tokenizer-english.json laya-tokenizer-multilingual.json)
(cd "$OUT" && shasum -a 256 *.zip > SHA256SUMS)
echo
ls -lh "$OUT"
