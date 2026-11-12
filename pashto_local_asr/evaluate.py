"""WER / CER against a reference TSV (Common Voice style: `path` and `sentence` columns)."""
import csv
import json
import re
import string
import unicodedata
from pathlib import Path

# ASCII punctuation plus Arabic comma, semicolon, question mark and full stop.
_PUNCT = re.compile("[" + re.escape(string.punctuation + "،؛؟۔") + "]")


LEVELS = ("basic", "pashto")

# "pashto" level: orthographic variants that writers (and the Common Voice references)
# use interchangeably. Letters that are distinct in Pashto (ې, ۍ, ئ, ګ, ښ, ...) are kept.
_PASHTO_MAP = str.maketrans({
    "ي": "ی",  # ي Arabic yeh -> ی  (the references mix both; see README)
    "ى": "ی",  # ى alef maksura -> ی
    "ك": "ک",  # ك Arabic kaf -> ک
    "ة": "ه",  # ة teh marbuta -> ه
    "ہ": "ه",  # ہ heh goal -> ه
    "‌": None, "‍": None, "ـ": None,  # ZWNJ, ZWJ, tatweel
    **{chr(0x0660 + d): str(d) for d in range(10)},  # Arabic-Indic digits
    **{chr(0x06F0 + d): str(d) for d in range(10)},  # Extended Arabic-Indic digits
})


def normalize(text, level="basic"):
    """'basic': NFC, punctuation stripped, whitespace collapsed (the headline scoring).
    'pashto': also removes diacritics and joiners and unifies the variants in _PASHTO_MAP."""
    if level not in LEVELS:
        raise ValueError(f"unknown normalisation level {level!r}; choose from {LEVELS}")
    text = unicodedata.normalize("NFC", text or "")
    if level == "pashto":
        # Strip standalone diacritics (harakat) on the NFC text, not after NFD, so that
        # precomposed letters such as ئ and آ keep their marks.
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = text.translate(_PASHTO_MAP)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def edit_distance(ref, hyp):
    """Levenshtein distance between two sequences."""
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def wer_counts(reference, hypothesis, level="basic"):
    ref = normalize(reference, level).split()
    return edit_distance(ref, normalize(hypothesis, level).split()), len(ref)


def cer_counts(reference, hypothesis, level="basic"):
    ref = normalize(reference, level)
    return edit_distance(ref, normalize(hypothesis, level)), len(ref)


def wer(reference, hypothesis, level="basic"):
    errors, n = wer_counts(reference, hypothesis, level)
    return errors / n if n else (0.0 if not normalize(hypothesis, level) else 1.0)


def cer(reference, hypothesis, level="basic"):
    errors, n = cer_counts(reference, hypothesis, level)
    return errors / n if n else (0.0 if not normalize(hypothesis, level) else 1.0)


def evaluate(reference_tsv, hyp_dir, level="basic"):
    """Corpus-level WER/CER. Each reference row's `path` stem is matched to
    `<stem>.transcript.json` in hyp_dir. Missing hypotheses count as empty output."""
    hyp_dir = Path(hyp_dir)
    w_err = w_n = c_err = c_n = 0
    rows, missing = [], []
    with open(reference_tsv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            stem = Path(row["path"]).stem
            hyp_file = hyp_dir / (stem + ".transcript.json")
            if hyp_file.is_file():
                hypothesis = json.loads(hyp_file.read_text(encoding="utf-8")).get("text", "")
            else:
                hypothesis = ""
                missing.append(stem)
            we, wn = wer_counts(row["sentence"], hypothesis, level)
            ce, cn = cer_counts(row["sentence"], hypothesis, level)
            w_err, w_n, c_err, c_n = w_err + we, w_n + wn, c_err + ce, c_n + cn
            rows.append({"path": row["path"], "wer": we / wn if wn else None,
                         "cer": ce / cn if cn else None})
    return {
        "normalisation": level,
        "utterances": len(rows),
        "missing_hypotheses": missing,
        "wer": w_err / w_n if w_n else None,
        "cer": c_err / c_n if c_n else None,
        "per_utterance": rows,
    }
