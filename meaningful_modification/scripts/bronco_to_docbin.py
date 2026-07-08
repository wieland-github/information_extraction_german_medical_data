"""
BRONCO150 (CoNLL) -> spaCy DocBin fuer die Zero-Shot-Evaluation.

Mappt die BRONCO-Entitaeten auf dein Label-Set:
  MED  -> Medikation
  DIAG -> Diagnose
  TREAT (OPS) wird verworfen (kein Gegenstueck), "Dosis" existiert in BRONCO nicht.

WICHTIG vor dem ersten Lauf:
  1. Schau in eine CoNLL-Datei ("head -20 randomSentSet1.conll") und pruefe:
     - In welcher Spalte steht der NER-Tag? (siehe TAG_COL unten)
     - Wie heissen die Typen genau? (z.B. "B-MED" -> Typ "MED", oder "B-MEDICATION")
  2. Passe LABEL_MAP und ggf. TAG_COL entsprechend an.

Aufruf (alle 5 Splits als ein Test-Set):
  python meaningful_modification/scripts/bronco_to_docbin.py \
    --conll pfad/zu/randomSentSet1.conll pfad/zu/randomSentSet2.conll \
            pfad/zu/randomSentSet3.conll pfad/zu/randomSentSet4.conll \
            pfad/zu/randomSentSet5.conll \
    --out meaningful_modification/data/bronco_test.spacy
"""
import argparse
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

# Spaltenindex des NER-Tags in der CoNLL-Datei (0 = Token). Nach Bedarf anpassen.
TAG_COL = 1

# BRONCO-Entitaetstyp (Teil nach "B-"/"I-") -> dein Label. Nach dem ersten Blick pruefen!
LABEL_MAP = {
    "MED": "Medikation",
    "DIAG": "Diagnose",
    # "TREAT": bewusst NICHT gemappt -> wird ignoriert
}


def read_conll(path):
    """Liest eine CoNLL-Datei -> Liste von Saetzen; je Satz Liste von (token, tag)."""
    sentences, tokens = [], []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    sentences.append(tokens)
                    tokens = []
                continue
            cols = line.split("\t") if "\t" in line else line.split()
            token = cols[0]
            tag = cols[TAG_COL] if len(cols) > TAG_COL else "O"
            tokens.append((token, tag))
    if tokens:
        sentences.append(tokens)
    return sentences


def iob_entities(tags):
    """IOB-Tags -> Liste von (tok_start, tok_end_exklusiv, label), gemappt auf dein Label-Set."""
    ents = []
    i, n = 0, len(tags)
    while i < n:
        prefix, _, etype = tags[i].partition("-")
        label = LABEL_MAP.get(etype)
        if prefix == "B" and label:
            j = i + 1
            while j < n:
                p2, _, e2 = tags[j].partition("-")
                if p2 == "I" and LABEL_MAP.get(e2) == label:
                    j += 1
                else:
                    break
            ents.append((i, j, label))
            i = j
        else:
            i += 1
    return ents


def build_doc(nlp, tokens):
    """(token, tag)-Satz -> spaCy Doc mit gemappten Entitaeten (char spans)."""
    words = [tok for tok, _ in tokens]
    tags = [tag for _, tag in tokens]

    # Text durch Join mit einfachem Leerzeichen; char-Offsets pro Token merken.
    text = " ".join(words)
    offsets, pos = [], 0
    for word in words:
        offsets.append((pos, pos + len(word)))
        pos += len(word) + 1  # +1 fuer das Leerzeichen

    doc = nlp.make_doc(text)
    spans = []
    for tok_start, tok_end, label in iob_entities(tags):
        start_char = offsets[tok_start][0]
        end_char = offsets[tok_end - 1][1]
        span = doc.char_span(start_char, end_char, label=label, alignment_mode="expand")
        if span is not None:
            spans.append(span)
    doc.ents = filter_spans(spans)
    return doc


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conll", nargs="+", required=True,
                        help="Eine oder mehrere CoNLL-Dateien (randomSentSet1-5).")
    parser.add_argument("--out", default="meaningful_modification/data/bronco_test.spacy")
    return parser.parse_args()


def main():
    args = parse_arguments()
    nlp = spacy.blank("de")
    doc_bin = DocBin()

    n_sent = n_ent = 0
    for path in args.conll:
        for tokens in read_conll(path):
            doc = build_doc(nlp, tokens)
            doc_bin.add(doc)
            n_sent += 1
            n_ent += len(doc.ents)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc_bin.to_disk(str(out))
    print(f"{n_sent} Saetze, {n_ent} Entitaeten (Medikation/Diagnose) -> {out}")


if __name__ == "__main__":
    main()
