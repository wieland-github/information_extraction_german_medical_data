#!/usr/bin/env bash

#   bash related_work/scripts/train_gbert_base.sh           # voller Lauf, GPU
#   bash related_work/scripts/train_gbert_base.sh -1        # voller Lauf, CPU 
#   bash related_work/scripts/train_gbert_base.sh -1 test   # Testlauf: 50 Steps

set -e
cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK=TRUE     # verhindert OMP-Fehler #15 auf dem Mac
export HF_HUB_DISABLE_XET=1          # Xet-Download umgehen

MODEL="deepset/gbert-base"
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

# paper spezifisch ab; optimizer
python -m spacy train config_spacy.cfg \
  --components.transformer.model.name "$MODEL" \
  --output "./$OUT" \
  --paths.train ./data/train.spacy \
  --paths.dev   ./data/validation.spacy \
  --gpu-id "$GPU_ID" \
  --training.optimizer.learn_rate.initial_rate 5e-5 \
  --training.batcher.size 128 \
  $EXTRA

echo "Fertig: $OUT/model-best"
