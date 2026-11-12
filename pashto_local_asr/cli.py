"""Command line interface: `pashto-asr transcribe ...` and `pashto-asr evaluate ...`."""
import argparse
import sys
from pathlib import Path

from .transcribe import (DEFAULT_LANGUAGE, DEFAULT_MODEL, DEFAULT_REVISION, find_audio,
                         load_pipeline, output_path_for, transcribe_file, write_json)


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

    e = sub.add_parser("evaluate", help="Compute WER/CER against a reference TSV.")
    e.add_argument("--reference", type=Path, required=True,
                   help="TSV with `path` and `sentence` columns (Common Voice format).")
    e.add_argument("--hyp-dir", type=Path, required=True,
                   help="Folder holding <stem>.transcript.json outputs.")
    e.add_argument("--json-out", type=Path, help="Also write the full report as JSON.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        from .evaluate import evaluate
        report = evaluate(args.reference, args.hyp_dir)
        if args.json_out:
            write_json(args.json_out, report)
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
    pipe = load_pipeline(args.model, args.revision, args.threads)
    failed = 0
    for audio in files:
        target = output_path_for(audio, args.out)
        if target.exists() and not args.overwrite:
            print(f"skip (exists): {target}")
            continue
        try:
            record = transcribe_file(pipe, audio, args.model, args.revision, args.language)
        except Exception as error:  # keep going through a folder
            failed += 1
            print(f"FAILED {audio.name}: {error}", file=sys.stderr)
            continue
        write_json(target, record)
        flags = record["quality_screen"]["flags"]
        note = "flags: " + ", ".join(flags) if flags else "screen passed"
        print(f"{audio.name} -> {target} [{note}]")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
