#!/usr/bin/env bash
# Lädt die fertigen Core-ML-Modelle aus dem GitHub-Release nach Models/ und prüft die Prüfsummen.
#
#   scripts/fetch-models.sh          laya, drei Checkpoints, rund 2,2 GB
#   scripts/fetch-models.sh kev      kev, ein Paket für alle Längen, rund 1,1 GB
#   scripts/fetch-models.sh alle     beides
#
# JEV_REPO und JEV_RELEASE überschreiben Repo und Release, JEV_BASE_URL die ganze Quelle,
# JEV_WANT die Liste der Pakete. Das ältere Release models-v1 hält kev noch als drei getrennte
# Zuschnitte (Kev06B-Q4-fp16, Kev06B-L256-Q4-fp16, Kev06B-L1024-Q4K96-fp16).
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${JEV_REPO:-GodModeAI2025/JevCoreML}"
RELEASE="${JEV_RELEASE:-models-v2}"
BASE="${JEV_BASE_URL:-https://github.com/$REPO/releases/download/$RELEASE}"

LAYA="Laya-EN-L512-K512-fp16 Laya-ML-L1024-K1024-fp16 Laya-TD-L1024-K1024-fp16"
KEV="JevCoreML"
case "${1:-laya}" in
  laya) WANT="$LAYA" ;;
  kev) WANT="$KEV" ;;
  alle|all) WANT="$LAYA $KEV" ;;
  *) echo "Aufruf: $0 [laya|kev|alle]" >&2; exit 2 ;;
esac
# Einzelne Pakete statt eines Satzes, etwa JEV_WANT="Laya-EN-L512-K512-fp16".
WANT="${JEV_WANT:-$WANT}"

mkdir -p Models .downloads
fetch() { curl -fL --retry 3 --progress-bar -o ".downloads/$1" "$BASE/$1"; }

fetch SHA256SUMS
check() { (cd .downloads && grep " $1\$" SHA256SUMS | shasum -a 256 -c -); }

for file in tokenizers.zip $(for m in $WANT; do echo "$m.mlpackage.zip"; done); do
  echo "lade $file"
  fetch "$file"
  check "$file"
done

unzip -oq .downloads/tokenizers.zip -d Models
for m in $WANT; do
  rm -rf "Models/$m.mlpackage"
  ditto -x -k ".downloads/$m.mlpackage.zip" Models
done
rm -rf .downloads
echo
echo "fertig, in Models/:"
ls Models
