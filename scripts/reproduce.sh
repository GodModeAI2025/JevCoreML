#!/usr/bin/env bash
# Baut die Core-ML-Modelle selbst, von Null: Umgebung, Originalgewichte, Export, Referenzdaten,
# Paritätsprüfung. Wer die Modelle nur benutzen will, braucht das nicht, sondern
# scripts/fetch-models.sh. Braucht einen Apple-Silicon-Mac, Xcode, uv und rund 20 GB Platz.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

echo "== 1/9 Python-Umgebung =="
cd Exporter
[ -d .venv ] || uv venv --python 3.12 .venv
VIRTUAL_ENV="$PWD/.venv" uv pip install -q \
  "torch>=2.7,<2.9" "transformers>=4.51,<4.58" "peft>=0.15" "coremltools>=9.0" \
  "huggingface-hub>=0.26" "accelerate>=1.0" "pydantic>=2.9" "numpy>=2.0" \
  "datasets>=3.0" "scikit-learn>=1.5" "httpx>=0.28" "pytest>=8" "laya>=0.3.4"

echo "== 2/9 kev-Quellen =="
[ -d "$ROOT/vendor/kev" ] || git clone --depth 1 https://github.com/jaredpalmer/kev.git "$ROOT/vendor/kev"
echo "$ROOT/vendor/kev" > .venv/lib/python3.12/site-packages/kev_src.pth

echo "== 3/9 Golden-Referenzen aus PyTorch =="
./.venv/bin/python dump_golden.py
# Breiter Suchlauf fuer den Tokenizer: Basiszeichen mit mehreren kombinierenden Marken.
./.venv/bin/python dump_tokenizer_fuzz.py

echo "== 4/9 Core-ML-Export =="
# Phase 1: eine Frage, Maske als Eingabe. Bleibt im Baum, weil die Paritätskette daran hängt.
./.venv/bin/python export_kev_coreml.py --precision fp32 --out "$ROOT/Models/Kev06B-fp32.mlpackage"
./.venv/bin/python export_kev_coreml.py --precision fp16 --out "$ROOT/Models/Kev06B-fp16.mlpackage"
# Phase 2: bis zu vier Fragen in einem Durchlauf, Maske im Graphen.
./.venv/bin/python export_kev_fanout.py --precision fp32 --out "$ROOT/Models/Kev06B-Q4-fp32.mlpackage"
./.venv/bin/python export_kev_fanout.py --precision fp16 --out "$ROOT/Models/Kev06B-Q4-fp16.mlpackage"
# Der kleine Bucket für --buckets: kurze Anfragen, rund drei Viertel von kevs Suite.
./.venv/bin/python export_kev_fanout.py --precision fp16 --length 256 --questions 4 --options 8 \
  --out "$ROOT/Models/Kev06B-L256-Q4-fp16.mlpackage"
# Breiter Export für Aufgaben mit vielen Optionen, etwa Banking77 mit 77 Kategorien.
./.venv/bin/python export_kev_fanout.py --precision fp16 --length 1024 --questions 4 --options 96 \
  --out "$ROOT/Models/Kev06B-L1024-Q4K96-fp16.mlpackage"
# Das Paket im Release: ein Export mit sechs aufgezählten Längen, acht Fragen und 256 Optionen.
# Die Laufzeit rechnet jede Anfrage in der kürzesten Länge, in die sie passt. Die getrennten
# Zuschnitte darüber bleiben für den Vergleich.
./.venv/bin/python export_kev_fanout.py --precision fp16 --lengths 128,256,512,1024,2048,3072 \
  --default-length 512 --questions 8 --options 256 --name JevCoreML \
  --out "$ROOT/Models/JevCoreML.mlpackage"
cp "$(ls -d "$HOME"/.cache/huggingface/hub/models--jaredpalmer--kev-0.6b/snapshots/* | head -1)/tokenizer.json" "$ROOT/Models/tokenizer.json"

echo "== 5/9 Core ML gegen PyTorch (Python) =="
./.venv/bin/python verify_coreml.py --model "$ROOT/Models/Kev06B-fp32.mlpackage"
./.venv/bin/python verify_coreml.py --model "$ROOT/Models/Kev06B-fp16.mlpackage"
# Das Paket JevCoreML je Länge und Recheneinheit, dazu Logit für Logit gegen den festen Export.
./.venv/bin/python verify_fanout.py --model "$ROOT/Models/JevCoreML.mlpackage" \
  --compare "$ROOT/Models/Kev06B-Q4-fp16.mlpackage"

echo "== 6/9 laya: Export, Referenzen, Prüfung =="
cd "$ROOT/Exporter"
# Drei Checkpoints, ein Vertrag. fp16 und cpuAndGPU. Jedes Paket nimmt mehrere Eingabelängen an
# (128, 256, ... bis L), K ist gleich L, so viele Optionen, wie laya überhaupt in die Sequenz
# bekommt. Siehe docs/laya-port.md.
./.venv/bin/python export_laya_coreml.py --checkpoint english --length 512 \
  --precision fp16 --out "$ROOT/Models/Laya-EN-L512-K512-fp16.mlpackage"
./.venv/bin/python export_laya_coreml.py --checkpoint typed-decisions --length 1024 \
  --precision fp16 --out "$ROOT/Models/Laya-TD-L1024-K1024-fp16.mlpackage"
./.venv/bin/python export_laya_coreml.py --checkpoint multilingual --length 1024 \
  --precision fp16 --out "$ROOT/Models/Laya-ML-L1024-K1024-fp16.mlpackage"
for cp in english multilingual typed-decisions; do
  ./.venv/bin/python dump_laya_golden.py --checkpoint "$cp"
done
# Der Suchlauf sucht gezielt im Wortschatz: Token mit kanonisch gleichwertigem Gegenstück und
# solche, die mit U+FEFF oder einer kombinierenden Marke beginnen. Dort lagen beide Fehler.
./.venv/bin/python dump_laya_tokenizer_fuzz.py --checkpoint english
./.venv/bin/python dump_laya_tokenizer_fuzz.py --checkpoint multilingual
cp "$ROOT/Golden/laya/english/tokenizer-fuzz.json" "$ROOT/Golden/laya/typed-decisions/tokenizer-fuzz.json"
./.venv/bin/python dump_laya_routing.py
# Unicode-Verhalten der Referenz je Codepunkt: Regex-Klassen und NFC. Daran hängen die
# UnicodeReferenceTests, die ganze Fehlerklassen finden, die kein Satzkorpus trifft.
./.venv/bin/python dump_unicode_reference.py
# Erzeugt LayaPresets.swift aus laya.presets und die Referenzantworten dazu.
./.venv/bin/python gen_laya_presets.py
./.venv/bin/python dump_laya_email.py
LAYA_TOK="$(ls -d "$HOME"/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/* | head -1)"
cp "$LAYA_TOK/tokenizer/tokenizer.json" "$ROOT/Models/laya-tokenizer-english.json"
cp "$LAYA_TOK/multilingual/tokenizer/tokenizer.json" "$ROOT/Models/laya-tokenizer-multilingual.json"
./.venv/bin/python verify_laya_coreml.py --checkpoint english \
  --model "$ROOT/Models/Laya-EN-L512-K512-fp16.mlpackage" --units gpu
./.venv/bin/python verify_laya_coreml.py --checkpoint multilingual \
  --model "$ROOT/Models/Laya-ML-L1024-K1024-fp16.mlpackage" --units gpu

echo "== 7/9 Swift gegen PyTorch =="
cd "$ROOT"
for model in JevCoreML Kev06B-fp32 Kev06B-fp16 Kev06B-Q4-fp32 Kev06B-Q4-fp16 Kev06B-L1024-Q4K96-fp16; do
  echo "--- $model ---"
  # --skip Laya: die laya-Tests haengen nicht an JEV_MODEL und laufen gleich darunter selbst.
  JEV_REQUIRE_MODEL=1 JEV_MODEL="$ROOT/Models/$model.mlpackage" swift test --skip Laya
done
for cp in english multilingual typed-decisions; do
  echo "--- laya $cp ---"
  LAYA_CHECKPOINT="$cp" swift test --filter Laya
done
swift build -c release

echo "== 8/9 Konformität gegen kevs eigene Tests =="
"$ROOT/.build/release/jev" \
  --model "$ROOT/Models/JevCoreML.mlpackage" --tokenizer "$ROOT/Models/tokenizer.json" \
  --serve --port 8008 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sS -m 2 http://127.0.0.1:8008/healthz > /dev/null 2>&1; do sleep 0.5; done
(cd "$ROOT/vendor/kev" && KEV_BASE_URL=http://127.0.0.1:8008 \
  "$ROOT/Exporter/.venv/bin/python" -m pytest tests/test_api.py -q)

echo "== 9/9 laya gegen das Original, Ende zu Ende =="
cd "$ROOT/Exporter"
./.venv/bin/python laya_baseline.py --device cpu --repeat 20 --out "$ROOT/Benchmarks/laya-baseline-cpu.json"
# Der Port gegen laya auf denselben fünf Fällen, Antworten und Zeiten.
./.venv/bin/python compare_laya_port.py
# Und über HTTP: jev --engine laya --serve gegen Router.predict, Form, Routing und Zeiten.
"$ROOT/.build/release/jev" --engine laya --models "$ROOT/Models" \
  --serve --port 8131 > /dev/null 2>&1 &
LAYA_SERVER=$!
trap 'kill $SERVER $LAYA_SERVER 2>/dev/null || true' EXIT
until curl -sS -m 2 http://127.0.0.1:8131/healthz > /dev/null 2>&1; do sleep 0.5; done
./.venv/bin/python compare_laya_server.py --url http://127.0.0.1:8131/v1/systemone
kill $LAYA_SERVER 2>/dev/null || true
for task in agnews emotion; do
  ./.venv/bin/python laya_parity_corpus.py --task "$task" --limit 1000 --units gpu \
    --model "$ROOT/Models/Laya-EN-L512-K512-fp16.mlpackage" \
    --out "$ROOT/Benchmarks/laya-parity-$task-gpu.json"
done

echo
echo "fertig. Demo:"
echo "  ./.build/release/jev --models Models --demo --repeat 15"
echo "  ./.build/release/jev --engine laya --models Models --demo --repeat 15"
