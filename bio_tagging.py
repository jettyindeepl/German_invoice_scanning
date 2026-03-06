import io
import json
import os
import shutil

import pandas as pd
from difflib import SequenceMatcher
from PIL import Image
import pytesseract

# Find tesseract binary on PATH (works on macOS/Linux/Windows)
_tess = shutil.which("tesseract")
if _tess:
    pytesseract.pytesseract.tesseract_cmd = _tess

# Detect tessdata dir from tesseract itself and pick language
_TESSDATA = "/opt/homebrew/share/tessdata"
OCR_LANG = "deu" if os.path.exists(os.path.join(_TESSDATA, "deu.traineddata")) else "eng"

# -----------------------------------------------
# 1. Field Definitions
# -----------------------------------------------
# Each field is treated as a "question" asked to the document.
# The model predicts a (start, end) token span as the answer.

FIELDS = [
    "Der Name der Bank",
    "Der Name der Firma",
    "Die Adresse der Firma",
    "Falligkeitsdatum",
    "IBAN",
    "Rechnungsdatum",
    "Rechnungsnummer",
    "Summe",
    "Telefonnummer",
]

# Typed BIO label set: O, B-<field>, I-<field>
LABEL_LIST = ["O"] + [f"B-{f}" for f in FIELDS] + [f"I-{f}" for f in FIELDS]
label2id   = {l: i for i, l in enumerate(LABEL_LIST)}
id2label   = {i: l for l, i in label2id.items()}

# Human-readable question prompts for each field
FIELD_QUESTIONS = {
    "Der Name der Bank":      "Was ist der Name der Bank?",
    "Der Name der Firma":     "Was ist der Name der Firma?",
    "Die Adresse der Firma":  "Was ist die Adresse der Firma?",
    "Falligkeitsdatum":       "Was ist das Fälligkeitsdatum?",
    "IBAN":                   "Was ist die IBAN?",
    "Rechnungsdatum":         "Was ist das Rechnungsdatum?",
    "Rechnungsnummer":        "Was ist die Rechnungsnummer?",
    "Summe":                  "Was ist die Gesamtsumme?",
    "Telefonnummer":          "Was ist die Telefonnummer?",
}



# -----------------------------------------------
# 3. BIO Label Builder
# -----------------------------------------------

def build_bio_labels(ocr_words: list, spans: dict) -> list:
    """
    Convert per-field span annotations into a typed BIO label sequence.

    Each token gets one of:
      - "O"          — not part of any field
      - "B-<field>"  — first token of a field span
      - "I-<field>"  — continuation token of a field span

    Conflict resolution: if spans overlap, the first-encountered field wins.
    """
    labels = ["O"] * len(ocr_words)

    for field, span in spans.items():
        if not span["found"]:
            continue
        start, end = span["start"], span["end"]
        for i in range(start, end + 1):
            # Only assign if not already claimed by a previous field
            if labels[i] == "O":
                labels[i] = f"B-{field}" if i == start else f"I-{field}"

    return labels

# -----------------------------------------------
# 4. OCR Helper
# -----------------------------------------------

def run_ocr(image: Image.Image):
    """
    Run Tesseract OCR on an image.
    Returns list of dicts: {word, bbox}
    Bounding boxes normalized to 0-1000 for LayoutLMv2.
    """
    width, height = image.size

    data = pytesseract.image_to_data(
        image,
        lang="deu",#OCR_LANG,
        output_type=pytesseract.Output.DICT
    )

    words = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        if int(data["conf"][i]) < 0:
            continue

        x1 = data["left"][i]
        y1 = data["top"][i]
        x2 = x1 + data["width"][i]
        y2 = y1 + data["height"][i]

        words.append({
            "word": word,
            "bbox": [
                max(0, min(1000, int(x1 / width  * 1000))),
                max(0, min(1000, int(y1 / height * 1000))),
                max(0, min(1000, int(x2 / width  * 1000))),
                max(0, min(1000, int(y2 / height * 1000))),
            ]
        })

    return words

# -----------------------------------------------
# 5. Fuzzy Span Finder
# -----------------------------------------------

def fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_span(value: str, ocr_words: list, threshold=0.75):
    """
    Find the (start_idx, end_idx) of value in ocr_words using
    sliding window fuzzy matching.

    Returns (start, end) word indices, or None if not found.
    """
    value_tokens = value.split()
    n = len(value_tokens)
    if n == 0:
        return None

    best_score = 0
    best_span  = None

    for i in range(len(ocr_words) - n + 1):
        window     = " ".join(ocr_words[j]["word"] for j in range(i, i + n))
        score      = fuzzy_score(window, " ".join(value_tokens))
        if score > best_score:
            best_score = score
            best_span  = (i, i + n - 1)

    if best_score >= threshold:
        return best_span

    # Retry with lower threshold for short values
    if n <= 2:
        best_score2 = 0
        best_span2  = None
        for i in range(len(ocr_words) - n + 1):
            window = " ".join(ocr_words[j]["word"] for j in range(i, i + n))
            score  = fuzzy_score(window, " ".join(value_tokens))
            if score > best_score2:
                best_score2 = score
                best_span2  = (i, i + n - 1)
        if best_score2 >= 0.5:
            return best_span2

    return None

# -----------------------------------------------
# 6. Span Annotation
# -----------------------------------------------

def annotate_spans(ocr_words: list, gt_fields: dict):
    """
    For each field, find the (start, end) token span in ocr_words.

    Returns a dict:
        {
            field_name: {
                "start":         int or None,
                "end":           int or None,
                "answer_text":   str,
                "matched_words": [str],
                "found":         bool
            }
        }
    """
    spans = {}

    for field in FIELDS:
        value = gt_fields.get(field, "")

        if not value or str(value).strip() == "":
            spans[field] = {
                "start":         None,
                "end":           None,
                "answer_text":   "",
                "matched_words": [],
                "found":         False
            }
            continue

        span = find_span(str(value), ocr_words, threshold=0.75)

        if span:
            start, end = span
            matched = [ocr_words[i]["word"] for i in range(start, end + 1)]
            spans[field] = {
                "start":         start,
                "end":           end,
                "answer_text":   str(value),
                "matched_words": matched,
                "found":         True
            }
        else:
            spans[field] = {
                "start":         None,
                "end":           None,
                "answer_text":   str(value),
                "matched_words": [],
                "found":         False
            }

    return spans

# -----------------------------------------------
# 7. Process All Samples
# -----------------------------------------------

def process_dataset(df: pd.DataFrame):
    """
    Process all rows. Each sample produces one entry per FIELD
    (question-answer pair), i.e. like DocVQA/SQuAD style.

    Returns list of dicts ready for LayoutLMv2ForQuestionAnswering.
    """
    dataset = []
    skipped = 0
    total_found   = 0
    total_missing = 0

    for idx, row in df.iterrows():
        image = Image.open(
            io.BytesIO(row["image"]["bytes"])
        ).convert("RGB")

        gt_fields = json.loads(row["ground_truth"])["gt_parse"]
        ocr_words = run_ocr(image)

        if not ocr_words:
            print(f"  [SKIP] Sample {idx}: no OCR words detected")
            skipped += 1
            continue

        words = [w["word"] for w in ocr_words]
        boxes = [w["bbox"]  for w in ocr_words]
        spans = annotate_spans(ocr_words, gt_fields)
        bio_labels = build_bio_labels(ocr_words, spans)

        found   = sum(1 for s in spans.values() if s["found"])
        missing = sum(1 for s in spans.values() if not s["found"] and s["answer_text"])
        total_found   += found
        total_missing += missing

        # One sample per field (QA style)
        for field in FIELDS:
            span = spans[field]
            dataset.append({
                "image":        image,
                "words":        words,
                "boxes":        boxes,
                "question":     FIELD_QUESTIONS[field],
                "field":        field,
                "answer_text":  span["answer_text"],
                "start_idx":    span["start"],   # word-level index
                "end_idx":      span["end"],      # word-level index
                "found":        span["found"],
                "bio_labels":   bio_labels,       # typed BIO tags for all tokens
            })

        print(f"  Sample {idx:3d}: {len(words):4d} words | "
              f"{found}/{len(FIELDS)} fields found | "
              f"{missing} missing")

    print(f"\nProcessed: {len(df) - skipped} samples | Skipped: {skipped}")
    print(f"Total field spans found:   {total_found}")
    print(f"Total field spans missing: {total_missing}")
    print(f"Total QA pairs created:    {len(dataset)}")
    return dataset



