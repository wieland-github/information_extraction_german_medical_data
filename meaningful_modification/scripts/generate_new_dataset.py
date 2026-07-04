"""
Synthetische NER-Trainingsdaten mit einem LLM (Gemini) erzeugen.

Aufruf (um 5 sätze zu erzeugen): python meaningful_modification/scripts/generate_new_dataset.py --num_sentences 5 
Aufruf (gemini-2.5-flash-lite): python meaningful_modification/scripts/generate_new_dataset.py --num_sentences 5 --model gemini-2.5-flash-lite
"""

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from dotenv import load_dotenv
from google import genai
from google.genai import errors


# --------------------------------------------------------------------------
# Konfiguration & Konstanten
# --------------------------------------------------------------------------
# Ordner mit den offiziellen medizinischen Listen
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "medical_information"

# Struktur (Reihenfolge/Anzahl) der Entitaeten, die pro Satz gesampelt werden
SHAPES = [
    ("Medikation",),                                  
    ("Medikation", "Dosis"),
    ("Medikation", "Dosis", "Diagnose"),
    ("Diagnose",),
    ("Medikation", "Diagnose"),
    ("Medikation", "Dosis", "Medikation", "Dosis"),   
    ("Diagnose", "Diagnose"),                          
]


# original 12 sentences from prompt used to generate the GPTNERMED dataset
ORIGINAL_SENTENCES = [
    '<s>Zur weiteren Bekämpfung des <class="Diagnose">Juckreiz</class> wird die Einnahme von täglich <class="Dosis">100mg</class> <class="Medikation">Cortison</class> empfohlen.</s>',
    '<s>Bei wiederkehrender Infektion wie einer <class="Diagnose">Sepsis</class> oder schweren <class="Diagnose">Pnseumonien</class> wird eine Überwachung erforderlich sein.</s>',
    '<s><class="Medikation">Valsartan</class>/<class="Medikation">HCT</class> <class="Dosis">160</class>/<class="Dosis">12,5 mg</class> 1-0-0</s>',
    '<s><class="Medikation">Pantoprazol</class> <class="Dosis">40 mg</class> p.o.</s>',
    '<s>Die feingewebliche histopathologische Untersuchung ergab den Befund einer <class="Diagnose">Metastase</class> des bekannten malignen <class="Diagnose">Melanoms</class>.</s>',
    '<s><class="Diagnose">Diabetes Typ 2</class>-Patienten müssen regelmäßig <class="Medikation">Insulin</class> (mindestens mit <class="Dosis">12ml</class> dosiert) spritzen.</s>',
    '<s>Ich nehme <class="Medikation">Antibiotika</class> seit Tagen. Seitdem ist die <class="Diagnose">Mandelentzündung</class> deutlich besser geworden.</s>',
    '<s>Entlassung: <class="Dosis">40mg</class> <class="Medikation">Lidocain</class> wegen <class="Diagnose">Kopfschmerzen</class></s>',
    '<s>Zusammenfassende D: Zervix-PE bei 11 und 2 Uhr mit ausgeprägter <class="Diagnose">chronisch-florider Zervizitis</class>.</s>',
    '<s>Die Verschreibung von <class="Medikation">Hämatokrin</class> <class="Dosis">43mg</class> war unnötig.</s>',
    '<s>Der Patient klagt über <class="Diagnose">Karditiden</class> und nimmt täglich <class="Medikation">Nifedipin</class> ein.</s>',
    '<s>D: PE-Material der Portio bei 1 Uhr mit Nachweis einer schwergradigen <class="Diagnose">squamösen intraepithelialen Läsion</class> (<class="Diagnose">HSIL</class>; hier noch <class="Diagnose">CIN II</class>).</s>',
]

# System-Prompt, der dem LLM die Regeln fuer die Satzerzeugung vorgibt
SYSTEM_PROMT = (

    "Du erzeugst genau einen synthetischen deutschen medizinischen Satz für das Training eines NER-Modells.\n"

    "Verbindliche Regeln:\n"

    "1. Verwende jeden unter „Begriffe“ angegebenen Ausdruck exakt in der vorgegebenen Schreibweise, einschließlich Groß-/Kleinschreibung, Satzzeichen und Leerzeichen.\n"
    "2. Jeder vorgegebene Ausdruck muss im Feld „text“ genau einmal vorkommen.\n"
    "3. Verwende ausschließlich die vorgegebenen medizinischen Informationen.\n"
    "4. Ergänze keine weiteren Medikamente, Wirkstoffe, Dosierungen, Diagnosen, Symptome, Indikationen, Applikationswege, Darreichungsformen, Zeitangaben oder Behandlungseigenschaften.\n"
    "5. Allgemeine nichtmedizinische Funktionswörter und neutrale Satzbestandteile sind erlaubt.\n"
    "6. Erzeuge genau einen grammatisch vollständigen, plausiblen deutschen Satz.\n"
    "7. Erzeuge für jeden vorgegebenen Begriff genau einen Eintrag in „entities“.\n"
    "8. Der Wert von „entities.text“ muss exakt mit dem entsprechenden Ausdruck im Satz übereinstimmen.\n"
    "9. Die Einträge in „entities“ müssen in derselben Reihenfolge stehen, in der die Begriffe im Satz vorkommen.\n"
    "10. Verwende ausschließlich diese Labels: „Medikation“, „Dosis“ und „Diagnose“.\n"
    "11. Verwende keine Entität, die nicht in den Eingabebegriffen enthalten ist.\n"
    "12. Prüfe vor der Ausgabe intern:\n"
    "- Sind alle Begriffe exakt einmal enthalten?\n"
    "- Wurden keine medizinischen Informationen ergänzt?\n"
    "- Stimmen Text und Labels exakt überein?\n"
    "- Ist das JSON syntaktisch gültig?\n"
    "13. Antworte ausschließlich mit einem einzelnen gültigen JSON-Objekt. Kein Markdown, keine Erklärung und kein zusätzlicher Text.\n"
)

# Feature-Schema von jfrei/GPTNERMED, damit die bestehende Aufbereitung
# (gptnermed_data_preparation.py) direkt weiterlaeuft
DATASET_FEATURES = Features({
    "sentence": Value("string"),
    "ner_labels": Sequence(feature={
        "ner_class": ClassLabel(names=["Medikation", "Dosis", "Diagnose"]),
        "start": Value("int32"),
        "stop": Value("int32"),
    }),
})

STYLES = [
    "ein vollständiger, grammatischer Satz",
    "eine knappe, telegrammartige Klinik-Notiz (kein voller Satzbau)",
    "eine Medikationszeile mit Dosierschema (z.B. 1-0-0, p.o.)",
    "eine Aussage aus Patientensicht (Ich-Form)",
    "ein Auszug aus einem Arztbrief",
]
CONTEXTS = ["Aufnahme", "Anamnese", "Verlauf", "Entlassung", "Befund", "Medikationsplan"]


# --------------------------------------------------------------------------
# Daten aus den CSV-Dateien laden
# --------------------------------------------------------------------------
def load_medical_data(data_dir=DATA_DIR):
    whoACTddd = pd.read_csv(data_dir / "who_atc_ddd.csv")
    # Nur Zeilen mit Tagesdosis (ddd) und Einheit
    whoACTddd = whoACTddd.dropna(subset=["ddd", "uom"])

    AlphaIDSE = pd.read_csv(
        data_dir / "Alpha-ID-SE.csv",
        sep=";",
        encoding="utf-8-sig",
        dtype=str
    )

    ICD10GM = pd.read_csv(
        data_dir / "ICD-10-GM.csv",
        sep=";",
        encoding="utf-8-sig",
        dtype=str
    )

    MEDS = (whoACTddd["atc_name"].astype(str).to_list())
    DOSE_UNITS = (whoACTddd["uom"].astype(str).to_list())
    DOSE_VAL = (whoACTddd["ddd"].astype(str).to_list())
    DIAGS = (ICD10GM["Diagnose"].astype(str).to_list())

    return MEDS, DIAGS, DOSE_VAL, DOSE_UNITS


# --------------------------------------------------------------------------
# Begriffe (Entitaeten) zufaellig sampeln
# --------------------------------------------------------------------------
def make_dosage(rng, values, units):
    val = rng.choice(values)
    unit = rng.choice(units)

    space_prob = rng.random()
    if (space_prob < 0.7):
        text = f"{val}{unit}"
    else:
        text = f"{val} {unit}"

    return text.replace(".", ",")

def sample_entities(rng, medication, diagnoses, dose_values, dose_units):
    entities = []
    for word in rng.choice(SHAPES):
        if word == "Medikation":
            entities.append({"text": rng.choice(medication), "label": "Medikation"})
        elif word == "Dosis":
            entities.append({"text": make_dosage(rng, dose_values, dose_units), "label": "Dosis"})
        elif word == "Diagnose":
            entities.append({"text": rng.choice(diagnoses), "label": "Diagnose"})

    return entities


# --------------------------------------------------------------------------
# Few-Shot-Beispiele aufbereiten
# --------------------------------------------------------------------------
def parse_original_sentence(tagged):
    CLASS_RE = re.compile(r'<class="([^"]+)">(.*?)</class>')
    inner = tagged.replace("<s>", "").replace("</s>", "")

    entities = [
        {"text": match.group(2), "label": match.group(1)}
        for match in CLASS_RE.finditer(inner)
    ]
    text = CLASS_RE.sub(r"\2", inner)   # Tags entfernen, nur Inhalt behalten

    return {"text": text, "entities": entities}

# Few-Shot-Pool aus den Original-Saetzen (wird von build_llm_prompt genutzt)
EXAMPLE_POOL = [parse_original_sentence(sentence) for sentence in ORIGINAL_SENTENCES]

def build_examples(rng, pool, num_examples=2):
    chosen = rng.sample(pool, num_examples)
    lines = ["Beispiel Sätze:"]
    for example in chosen:
        lines.append(json.dumps(example, ensure_ascii=False))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Prompt bauen
# --------------------------------------------------------------------------
def build_task(entities):
    lines = ["Begriffe:"]
    for ent in entities:
        lines.append(f"- {ent['label']}: {ent['text']}")
    return "\n".join(lines)

def build_format():
    lines = ["Format:"]
    lines.append('{"text": "...", "entities": [{"text": "...", "label": "Medikation|Dosis|Diagnose"}, ...]}')
    return "\n".join(lines)

def build_llm_prompt(rng, medication, diagnoses, entities, example_pool=EXAMPLE_POOL, num_examples=2):

    message = []
    message.append(build_task(entities))
    message.append("\n")
    message.append("\n")
    message.append(f"Formuliere als: {rng.choice(STYLES)}. \nKontext: {rng.choice(CONTEXTS)}. ")
    message.append("\n")
    message.append("\n")
    message.append(build_format())
    message.append("\n")
    message.append("\n")
    message.append(build_examples(rng, example_pool, num_examples))
    message.append("\n")

    return "".join(message)


# --------------------------------------------------------------------------
# LLM aufrufen
# --------------------------------------------------------------------------
# HTTP-Codes, bei denen sich ein erneuter Versuch lohnt:
# 429 = Rate-Limit, 5xx = transiente Serverfehler.
RETRYABLE_CODES = {429, 500, 502, 503, 504}

# Client einmal erzeugen und wiederverwenden (nicht pro Aufruf neu).
_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _CLIENT


def call_LLM(model, message, max_retries=6, base_delay=2.0):
    client = _get_client()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=message,
                config={
                    "system_instruction": SYSTEM_PROMT,
                },
            )
            return response.text
        except errors.APIError as error:
            code = getattr(error, "code", None)
            # Nur bei Rate-Limit/Serverfehlern warten und erneut versuchen.
            if code in RETRYABLE_CODES and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"[retry] API {code}, warte {delay:.1f}s "
                      f"(Versuch {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            # Nicht behebbar (z.B. 400/401/403) oder Retries erschoepft: weiterreichen.
            raise

    raise RuntimeError("call_LLM: max_retries erschoepft")


# --------------------------------------------------------------------------
# LLM-Antwort in GPTNERMED-Datensatzform
# --------------------------------------------------------------------------
def get_label_id(label):
    label_mapping = {
        "Medikation": 0,
        "Dosis": 1,
        "Diagnose": 2
    }
    return label_mapping.get(label)

def find_spans(sentence, entities):
    spans = []

    sentence = sentence.lower()

    for ent in entities:
        ent_lower = ent["text"].lower()

        start = sentence.find(ent_lower)
        stop = start + len(ent_lower)

        spans.append({
            "start": start,
            "stop": stop,
            "label": ent["label"]
        })

    spans.sort(key=lambda span: span["start"])

    return {
        "ner_class": [get_label_id(span["label"]) for span in spans],
        "start": [span["start"] for span in spans],
        "stop": [span["stop"] for span in spans]
    }

def build_dataset_form(sentence, spans):
    return {'sentence': sentence["text"], 'ner_labels': spans}


# --------------------------------------------------------------------------
# Argumente
# --------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Erzeugt einen synthetischen NER-Datensatz (GPTNERMED-Format) mit Gemini.",
    )
    parser.add_argument("--num_sentences", type=int, default=50,
                        help="Anzahl gueltiger Saetze, die erzeugt werden sollen.")
    parser.add_argument("--outdir", default=None,
                        help="Zielordner (Default: meaningful_modification/data/synthetic).")
    parser.add_argument("--model", default="gemini-3.1-flash-lite",
                        help="Gemini-Modellname.")
    parser.add_argument("--num_examples", type=int, default=2,
                        help="Anzahl Few-Shot-Beispiele pro Prompt.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed fuer reproduzierbares Sampling.")
    return parser.parse_args()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    args = parse_arguments()
    load_dotenv()

    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("GEMINI_API_KEY fehlt.")

    outdir = Path(args.outdir or DATA_DIR.parent / "synthetic")
    outdir.mkdir(parents=True, exist_ok=True)

    MEDS, DIAGS, DOSE_VAL, DOSE_UNITS = load_medical_data(DATA_DIR)
    rng = random.Random(args.seed)
    records = []

    for _ in range(args.num_sentences * 5):
        if len(records) == args.num_sentences:
            break

        entities = sample_entities(rng, MEDS, DIAGS, DOSE_VAL, DOSE_UNITS)
        prompt = build_llm_prompt(
            rng,
            MEDS,
            DIAGS,
            entities,
            EXAMPLE_POOL,
            args.num_examples,
        )

        try:
            raw = call_LLM(args.model, prompt).strip()
            raw = re.sub(r"^```\w*\s*|\s*```$", "", raw)
            result = json.loads(raw)
        except Exception as error:
            print(f"[skip] {error}")
            continue

        spans = find_spans(result["text"], entities)

        if -1 in spans["start"]:
            print("[skip] Begriff nicht gefunden")
            continue

        records.append(build_dataset_form(result, spans))
        print(f"{len(records)}/{args.num_sentences}")

    if not records:
        raise SystemExit("Keine gültigen Sätze erzeugt.")

    dataset = Dataset.from_list(records, features=DATASET_FEATURES)
    dataset_path = outdir / "synthetic_gptnermed"
    jsonl_path = outdir / "synthetic_gptnermed.jsonl"

    dataset.save_to_disk(str(dataset_path))

    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{len(records)} Sätze gespeichert:")
    print(dataset_path)
    print(jsonl_path)


if __name__ == "__main__":
    main()
