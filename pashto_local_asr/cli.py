"""Command line interface: `pashto-asr transcribe ...` and `pashto-asr evaluate ...`."""
import argparse
import os
import sys
import time
from pathlib import Path

from .transcribe import (DEFAULT_LANGUAGE, DEFAULT_MODEL, DEFAULT_REVISION, find_audio,
                         load_pipeline, model_card_info, output_path_for,
                         transcribe_file, update_run_meta, write_json)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pashto-asr", description="Offline Pashto/Dari transcription drafts.")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="Transcribe an audio file or a folder of audio.")
    t.add_argument("input", type=Path, help="Audio file or folder (searched recursively).")
    t.add_argument("--out", type=Path, help="Output folder (default: next to each audio file).")
    t.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id.")
    t.add_argument("--revision", default=DEFAULT_REVISION,
                   help="Exact model commit hash (pin it for reproducibility).")
    t.add_argument("--language", default=DEFAULT_LANGUAGE, help="Language code (default: ps).")
    t.add_argument("--threads", type=int, default=2, help="CPU threads.")
    t.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    t.add_argument("--record-model-card", action="store_true",
                   help="With --out, also fetch the model's licence and training datasets from the "
                        "Hugging Face Hub into run_meta.json (a network call; off by default).")

    e = sub.add_parser("evaluate", help="Compute WER/CER against a reference TSV.")
    e.add_argument("--reference", type=Path, required=True,
                   help="TSV with `path` and `sentence` columns (Common Voice format).")
    e.add_argument("--hyp-dir", type=Path, required=True,
                   help="Folder holding <stem>.transcript.json outputs.")
    e.add_argument("--json-out", type=Path, help="Also write the full report as JSON.")
    e.add_argument("--normalise", choices=("basic", "pashto"), default="basic",
                   help="'basic' (default): punctuation and whitespace only. 'pashto': also "
                        "unify spelling variants such as ي/ی and strip diacritics (see README).")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        from .evaluate import evaluate
        report = evaluate(args.reference, args.hyp_dir, args.normalise)
        if args.json_out:
            write_json(args.json_out, report)
        print(f"normalisation: {report['normalisation']}")
        print(f"utterances: {report['utterances']}")
        print(f"missing hypotheses: {len(report['missing_hypotheses'])}")
        print(f"WER: {report['wer']}")
        print(f"CER: {report['cer']}")
        return 0

    if not args.input.exists():
        print(f"Not found: {args.input}", file=sys.stderr)
        return 2
    files = find_audio(args.input)
    if not files:
        print("No audio files found.", file=sys.stderr)
        return 2
    t_load = time.perf_counter()
    pipe = load_pipeline(args.model, args.revision, args.threads)
    load_seconds = time.perf_counter() - t_load
    failed = 0
    card = params = None
    meta_dir = args.out if args.out else None
    for audio in files:
        target = output_path_for(audio, args.out)
        if target.exists() and not args.overwrite:
            print(f"skip (exists): {target}")
            continue
        t0 = time.perf_counter()
        try:
            record = transcribe_file(pipe, audio, args.model, args.revision, args.language)
        except Exception as error:  # keep going through a folder
            failed += 1
            print(f"FAILED {audio.name}: {error}", file=sys.stderr)
            continue
        elapsed = time.perf_counter() - t0
        write_json(target, record)
        if meta_dir:  # incremental, so a stopped run still has timing
            if card is None:
                fetch = args.record_model_card and os.environ.get("HF_HUB_OFFLINE") != "1"
                card = model_card_info(args.model, args.revision) if fetch else {}
                params = sum(p.numel() for p in pipe.model.parameters())
            update_run_meta(meta_dir, args.model, args.revision, pipe.architecture,
                            args.threads, 1, record["duration_seconds"], elapsed,
                            load_seconds, params=params, card=card or None)
        flags = record["quality_screen"]["flags"]
        note = "flags: " + ", ".join(flags) if flags else "screen passed"
        print(f"{audio.name} -> {target} [{note}]")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
