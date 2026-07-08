#!/usr/bin/env bash

#   bash meaningful_modification/scripts/evaluate_gbert_large.sh       # GPU (Colab)
#   bash meaningful_modification/scripts/evaluate_gbert_large.sh -1    # CPU (lokal)
set -e
cd "$(dirname "$0")/.."


GPU_ID="${1:-0}"
#MODEL_DIR="./models/gbert-large-syn-seed-42-model-best/model-best"
MODEL_DIR="./models/gbert-large-syn-seed42-model-best"
TEST_DATA="./data/test.spacy"
OUTPUT="./models/gbert-large-syn-seed42-model-best/test_metrics.json"

mkdir -p "$(dirname "$OUTPUT")"

python -m spacy benchmark accuracy \
  "$MODEL_DIR" \
  "$TEST_DATA" \
  --output "$OUTPUT" \
  --gpu-id "$GPU_ID"

echo "Fertig: models/gbert-large-syn-seed42-model-best/test_metrics.json"
