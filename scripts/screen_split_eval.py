"""Report corpus WER/CER separately for drafts that passed or failed the quality screen.

    python scripts/screen_split_eval.py <reference.tsv> <drafts_dir>
"""
import csv
import json
import sys
from pathlib import Path

from pashto_local_asr.evaluate import cer_counts, wer_counts


def main(reference_tsv, drafts_dir):
    drafts_dir = Path(drafts_dir)
    totals = {}
    with open(reference_tsv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            draft = drafts_dir / (Path(row["path"]).stem + ".transcript.json")
            record = json.loads(draft.read_text(encoding="utf-8"))
            group = "passed" if record["quality_screen"]["passed"] else "flagged"
            for key in ("all", group):
                t = totals.setdefault(key, [0, 0, 0, 0, 0])
                w_err, w_n = wer_counts(row["sentence"], record["text"])
                c_err, c_n = cer_counts(row["sentence"], record["text"])
                t[0] += w_err; t[1] += w_n; t[2] += c_err; t[3] += c_n; t[4] += 1
    for key in ("all", "passed", "flagged"):
        if key in totals:
            t = totals[key]
            print(f"{key:8} n={t[4]:4d}  WER={t[0] / t[1]:.3f}  CER={t[2] / t[3]:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
