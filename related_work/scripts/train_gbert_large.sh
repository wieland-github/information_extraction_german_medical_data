#!/usr/bin/env bash

#   bash scripts/train_gbert_large.sh           # voller Lauf, GPU
#   bash scripts/train_gbert_large.sh -1        # voller Lauf, CPU (Mac)
#   bash scripts/train_gbert_large.sh -1 test   # Testlauf: 50 Steps

set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE    # verhindert OMP-Fehler #15 auf dem Mac

MODEL="deepset/gbert-large"
GPU_ID="${1:-0}"                    # 0 = GPU, -1 = CPU
MODE="${2:-full}"                   # full, test

EXTRA=""
OUT="models/$(basename "$MODEL")"
if [ "$MODE" = "test" ]; then
  EXTRA="
  --training.max_steps 50 
  --training.eval_frequency 25 
  --training.patience 0"
  OUT="$OUT-test"
fi

python -m spacy train config_spacy.cfg \
  --components.transformer.model.name "$MODEL" \
  --output "./$OUT" \
  --paths.train ./data_split/train.spacy \
  --paths.dev   ./data_split/validation.spacy \
  --gpu-id "$GPU_ID" \
  $EXTRA

echo "Fertig: $OUT/model-best"
