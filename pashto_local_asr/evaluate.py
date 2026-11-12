"""WER / CER against a reference TSV (Common Voice style: `path` and `sentence` columns)."""
import csv
import json
import re
import string
import unicodedata
from pathlib import Path

# ASCII punctuation plus Arabic comma, semicolon, question mark and full stop.
_PUNCT = re.compile("[" + re.escape(string.punctuation + "،؛؟۔") + "]")


def normalize(text):
    text = unicodedata.normalize("NFC", text or "")
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


def wer_counts(reference, hypothesis):
    ref = normalize(reference).split()
    return edit_distance(ref, normalize(hypothesis).split()), len(ref)


def cer_counts(reference, hypothesis):
    ref = normalize(reference)
    return edit_distance(ref, normalize(hypothesis)), len(ref)


def wer(reference, hypothesis):
    errors, n = wer_counts(reference, hypothesis)
    return errors / n if n else (0.0 if not normalize(hypothesis) else 1.0)


def cer(reference, hypothesis):
    errors, n = cer_counts(reference, hypothesis)
    return errors / n if n else (0.0 if not normalize(hypothesis) else 1.0)


def evaluate(reference_tsv, hyp_dir):
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
            we, wn = wer_counts(row["sentence"], hypothesis)
            ce, cn = cer_counts(row["sentence"], hypothesis)
            w_err, w_n, c_err, c_n = w_err + we, w_n + wn, c_err + ce, c_n + cn
            rows.append({"path": row["path"], "wer": we / wn if wn else None,
                         "cer": ce / cn if cn else None})
    return {
        "utterances": len(rows),
        "missing_hypotheses": missing,
        "wer": w_err / w_n if w_n else None,
        "cer": c_err / c_n if c_n else None,
        "per_utterance": rows,
    }
