
# Skripts

### `data_preparation.py`

Downloads the dataset from Hugging Face, converts it to spaCy `DocBin` files, and saves:

- 'gptnermed'
- 'train.spacy'
- 'validation.spacy'
- 'test.spacy'

```bash
python related_work/scripts/data_preparation.py --outdir <output-directory>
```

### build_spacy_config.sh

Creates a spaCy configuration for a German NER pipeline optimized for accuracy and GPU training.

```bash
bash related_work/scripts/build_spacy_config.sh
```

Output: related_work/config_spacy.cfg

### `train_gbert_large.sh`

Trains a German spaCy NER model using `deepset/gbert-large`.

```bash
# GPU training
bash related_work/scripts/train_gbert_large.sh

# CPU training
bash related_work/scripts/train_gbert_large.sh -1

# Short test run with 50 steps
bash related_work/scripts/train_gbert_large.sh -1 test
```

Output: `related_work/models/gbert-large/model-best`

Test output: `related_work/models/gbert-large-test/model-best`

### `evaluate_gbert_large.sh`

Evaluates the trained `gbert-large` model on the test dataset and saves the evaluation metrics.

```bash
# GPU evaluation
bash related_work/scripts/evaluate_gbert_large.sh

# CPU evaluation
bash related_work/scripts/evaluate_gbert_large.sh -1
```

Output: `related_work/models/gbert-large/test_metrics.json`
