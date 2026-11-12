"""Compare several transcription runs on the same reference set as a Markdown table.

    python scripts/compare_models.py --reference test_reference.tsv runs/model_a runs/model_b ...

Each run folder holds `<stem>.transcript.json` drafts and (optionally) the
`run_meta.json` that `pashto-asr transcribe --out` writes. Rows follow the order given.
Missing drafts count as empty output (same rule as `pashto-asr evaluate`). With
`--common-only`, only clips that every run has a draft for are scored, so a run that was
stopped early does not look worse (or better) than runs that finished; wall time and RTF
still describe each run's own processed clips.
"""
import argparse
import csv
import json
from pathlib import Path

from pashto_local_asr.evaluate import cer_counts, wer_counts


def summarise(reference_tsv, run_dir, only_stems=None):
    """Per-run statistics dict. Pure file reading, no model needed."""
    run_dir = Path(run_dir)
    meta_file = run_dir / "run_meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
    tot = {"all": [0, 0, 0, 0, 0], "passed": [0, 0, 0, 0, 0]}
    missing = 0
    model = meta.get("model")
    revision = meta.get("revision")
    with open(reference_tsv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            stem = Path(row["path"]).stem
            if only_stems is not None and stem not in only_stems:
                continue
            draft = run_dir / (stem + ".transcript.json")
            if draft.is_file():
                record = json.loads(draft.read_text(encoding="utf-8"))
                text, passed = record.get("text", ""), record["quality_screen"]["passed"]
                model, revision = model or record.get("model"), revision or record.get("revision")
            else:
                text, passed, missing = "", False, missing + 1
            we, wn = wer_counts(row["sentence"], text)
            ce, cn = cer_counts(row["sentence"], text)
            for key in ("all", "passed") if passed else ("all",):
                t = tot[key]
                t[0] += we; t[1] += wn; t[2] += ce; t[3] += cn; t[4] += 1

    def rate(num, den):
        return num / den if den else None

    a, p = tot["all"], tot["passed"]
    audio, wall = meta.get("audio_seconds"), meta.get("wall_seconds")
    card = meta.get("model_card") or {}
    return {
        "model": model or run_dir.name, "revision": revision, "n": a[4], "missing": missing,
        "architecture": meta.get("architecture"), "params": meta.get("parameters"),
        "license": card.get("license"),
        "wer": rate(a[0], a[1]), "cer": rate(a[2], a[3]),
        "pass_rate": rate(p[4], a[4]),
        "wer_passed": rate(p[0], p[1]), "cer_passed": rate(p[2], p[3]), "n_passed": p[4],
        "wall_seconds": wall, "audio_seconds": audio,
        "rtf": rate(wall, audio) if wall is not None and audio else None,
    }


def common_stems(run_dirs):
    """Stems that have a draft in every run folder."""
    sets = [{p.name[:-len(".transcript.json")] for p in Path(d).glob("*.transcript.json")}
            for d in run_dirs]
    return set.intersection(*sets) if sets else set()


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _dur(seconds):
    if seconds is None:
        return "n/a"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _params(n):
    return "n/a" if not n else f"{n / 1e6:.0f}M"


def render_table(rows):
    header = ("| Model | Revision | Arch | Params | Licence | WER | CER | Screen pass | "
              "WER (passed) | CER (passed) | Wall time | RTF |")
    lines = [header, "|" + "---|" * 12]
    for r in rows:
        lines.append("| " + " | ".join([
            r["model"], (r["revision"] or "n/a")[:8], r["architecture"] or "n/a",
            _params(r["params"]), r["license"] or "None", _pct(r["wer"]), _pct(r["cer"]),
            f"{_pct(r['pass_rate'])} ({r['n_passed']}/{r['n']})",
            _pct(r["wer_passed"]), _pct(r["cer_passed"]), _dur(r["wall_seconds"]),
            "n/a" if r["rtf"] is None else f"{r['rtf']:.2f}"]) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("runs", nargs="+", type=Path, help="Run folders of drafts.")
    parser.add_argument("--common-only", action="store_true",
                        help="Score only clips every run has a draft for.")
    args = parser.parse_args(argv)
    only = common_stems(args.runs) if args.common_only else None
    rows = [summarise(args.reference, run, only) for run in args.runs]
    if only is not None:
        print(f"Scored on the {len(only)} clips common to all runs.\n")
    print(render_table(rows))
    print("\nRTF = wall-clock seconds / audio seconds (lower is faster); n/a where no run_meta.json.")
    for r in rows:
        if r["missing"]:
            print(f"WARNING: {r['model']}: {r['missing']} reference clips have no draft (counted as empty).")


if __name__ == "__main__":
    main()
