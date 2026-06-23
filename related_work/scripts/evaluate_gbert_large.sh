#!/usr/bin/env bash

#   bash scripts/evaluate_gbert_large.sh       # GPU (Colab)
#   bash scripts/evaluate_gbert_large.sh -1    # CPU (lokal)
set -e
cd "$(dirname "$0")/.."

GPU_ID="${1:-0}"
MODEL_DIR="models/gbert-large/model-best"

python -m spacy evaluate "$MODEL_DIR" ./data_split/test.spacy \
  --output ./models/gbert-large/test_metrics.json \
  --gpu-id "$GPU_ID"

echo "Fertig: models/gbert-large/test_metrics.json"
