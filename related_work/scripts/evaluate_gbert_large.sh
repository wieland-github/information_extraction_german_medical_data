#!/usr/bin/env bash

#   bash scripts/evaluate_gbert_large.sh       # GPU (Colab)
#   bash scripts/evaluate_gbert_large.sh -1    # CPU (lokal)
set -e
cd "$(dirname "$0")/.."


GPU_ID="${1:-0}"
MODEL_DIR="./models/gbert-large-test/model-best"
TEST_DATA="./data_split/test.spacy"
OUTPUT="./models/gbert-large-test/test_metrics.json"

mkdir -p "$(dirname "$OUTPUT")"

python -m spacy benchmark accuracy \
  "$MODEL_DIR" \
  "$TEST_DATA" \
  --output "$OUTPUT" \
  --gpu-id "$GPU_ID"

echo "Fertig: models/gbert-large/test_metrics.json"
