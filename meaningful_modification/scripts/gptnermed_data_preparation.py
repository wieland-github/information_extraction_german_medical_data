import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

import argparse
import json
from datasets import load_dataset
from pathlib import Path

"""
GPTNERMED Related Work reproduktion:
-> hier werden die Daten 
Ausführen: python meaningful_modification/scripts/goptnermed_data_preparation.py --outdir meaningful_modification/data
"""

def get_label(label_id):
    """
    Maps the label ID to its string.
    """
    label_mapping = {
        0: "Medikation",
        1: "Dosis",
        2: "Diagnose",
    }
    return label_mapping.get(label_id)

def data_to_docbin(*datasets):
    """
    Converts one or more datasets to a single DocBin for spaCy training.
    Mehrere Datensaetze werden in eine DocBin zusammengefuehrt (z.B. Original + synthetisch).
    """
    doc_bin = DocBin()
    nlp = spacy.blank("de")

    for data in datasets:
        for example in data:
            text = example["sentence"]
            labels = example["ner_labels"]
            doc = nlp(text)

            ents = []
            for start, stop, label_id in zip (labels["start"], labels["stop"], labels["ner_class"]):

                label = get_label(label_id)
                span = doc.char_span(int(start), int(stop), label=label, alignment_mode="expand")
                ents.append(span)

            doc.ents = filter_spans(ents)
            doc_bin.add(doc)

    return doc_bin


def load_synthetic(path):
    """
    Laedt die synthetischen Saetze aus einer JSONL-Datei (versionsunabhaengig).
    Bewusst kein datasets.load_from_disk: der Arrow-Ordner wird ggf. mit einer
    neueren datasets-Version geschrieben als die auf Colab (dort <3.0.0 noetig).
    Format pro Zeile: {"sentence": ..., "ner_labels": {"ner_class": [...], "start": [...], "stop": [...]}}
    """
    records = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_arguments():
    """
    Parses the given arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="jfrei/GPTNERMED")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--synthetic", default=None,
                        help="Pfad zur synthetischen JSONL-Datei "
                             "(synthetic_gptnermed.jsonl), wird an die Trainingsdaten angehaengt.")

    args = parser.parse_args()
    return args

def main():

    args = parse_arguments()

    print("Load Dataset...")

    dataset = args.dataset
    outdir = Path(args.outdir) if args.outdir else Path(__file__).parent
    outdir.mkdir(exist_ok=True)

    dataset = load_dataset(dataset, trust_remote_code=True)
    training_data = dataset["train"]
    validation_data = dataset["validation"]
    test_data = dataset["test"]

    # Synthetische Saetze NUR an das Training anhaengen.
    # Validation/Test bleiben das Original -> Ergebnisse bleiben mit der Baseline vergleichbar.
    train_parts = [training_data]
    if args.synthetic:
        synthetic_data = load_synthetic(args.synthetic)
        print(f"Synthetische Saetze angehaengt: {len(synthetic_data)}")
        train_parts.append(synthetic_data)

    doc_bin_train = data_to_docbin(*train_parts)
    doc_bin_validation = data_to_docbin(validation_data)
    doc_bin_test = data_to_docbin(test_data)

    doc_bin_train.to_disk(f"{outdir}/train.spacy")
    doc_bin_validation.to_disk(f"{outdir}/validation.spacy")
    doc_bin_test.to_disk(f"{outdir}/test.spacy")
    dataset.save_to_disk(f"{outdir}/gptnermed")


    print("Finished Data preparation")

    return 


if __name__ == "__main__":
    main()