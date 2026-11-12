"""Build the benchmark subset from a Common Voice Spontaneous Speech release.

Selects the official `test` split rows that have a non-empty human
transcription, writes a `path`/`sentence` reference TSV, and copies the
matching clips into one folder. Nothing is downloaded; point it at a release
you obtained yourself under its terms (the data may not be re-shared).

    python scripts/cv_spontaneous_subset.py <release_dir> <out_dir>
"""
import csv
import shutil
import sys
from pathlib import Path


def main(release_dir, out_dir):
    release_dir, out_dir = Path(release_dir), Path(out_dir)
    corpus = next(release_dir.glob("ss-corpus-*.tsv"))
    clips_out = out_dir / "test_clips"
    clips_out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(corpus, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("split") == "test" and (row.get("transcription") or "").strip():
                rows.append(row)
    with open(out_dir / "test_reference.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["path", "sentence"])
        for row in rows:
            writer.writerow([row["audio_file"], row["transcription"]])
            shutil.copy2(release_dir / "audios" / row["audio_file"], clips_out)
    minutes = sum(int(r["duration_ms"]) for r in rows) / 60000
    print(f"{len(rows)} test clips, {minutes:.1f} minutes -> {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
