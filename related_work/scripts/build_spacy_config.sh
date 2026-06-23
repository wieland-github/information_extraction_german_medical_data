#!/usr/bin/env bash

# bash scripts/create_spacy_config.sh
# ./scripts/create_spacy_config.sh

set -e
cd "$(dirname "$0")/.."

python -m spacy init config config_spacy.cfg \
  --lang de --pipeline ner --optimize accuracy --gpu --force

echo "Fertig: config_spacy.cfg"
