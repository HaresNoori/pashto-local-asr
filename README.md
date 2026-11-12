# pashto-local-asr

Offline speech-to-text drafts for Pashto, with provenance on every output and a built-in quality screen. Audio never leaves your machine.

## The problem

Pashto (roughly 40-60 million speakers) and Dari are under-served by speech technology: few open benchmarks, few maintained models, and weak support in mainstream tools. The usual workaround is a cloud transcription service, which means uploading recordings that may contain sensitive testimony, legal material or identifying details to a third party.

`pashto-local-asr` is a small, auditable alternative. It runs a pinned open model on your own CPU, records exactly which audio and which model produced each transcript, and flags the obvious failure modes instead of presenting every output as trustworthy.

## Who it is for

- Researchers working with Pashto or Dari field recordings
- Journalists handling sensitive interviews
- Legal aid and humanitarian workers who cannot send client audio to the cloud
- Diaspora organisations and community archives

## Install

Requires Python 3.10+ and enough RAM for a Whisper-family model. No GPU needed.

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"       # installs numpy, av (PyAV), torch, transformers, pytest
```

The first run downloads the model from the Hugging Face Hub. After it is cached, set `HF_HUB_OFFLINE=1` to guarantee no network access.

## Quick start

```bash
# one file, output written next to it as interview.transcript.json
pashto-asr transcribe interview.m4a

# a folder (recursive), outputs to a separate folder
pashto-asr transcribe recordings/ --out drafts/

# choose a different model and pin its exact commit
pashto-asr transcribe interview.m4a --model ORG/MODEL --revision COMMIT_HASH
```

Existing outputs are skipped unless you pass `--overwrite`. `python -m pashto_local_asr ...` also works.

## Output and provenance schema

Each input produces `<name>.transcript.json`:

| Field | Meaning |
|---|---|
| `schema_version` | Output format version (currently 1) |
| `status` | Always `"unverified_draft"` |
| `source_audio` | Input file name only (no directory path) |
| `sha256` | SHA-256 of the input file bytes |
| `duration_seconds` | Decoded length at 16 kHz mono |
| `model`, `revision` | Hugging Face model id and exact commit used |
| `language` | Language code passed to the model (default `ps`) |
| `tool` | Name and version of this package |
| `created_at` | UTC timestamp |
| `text` | Full transcript |
| `segments` | List of `{timestamp: [start, end], text}` |
| `quality_screen` | `{passed, flags, metrics, thresholds}` |

### Quality screen

Heuristic checks that catch obvious failures. Thresholds live in `pashto_local_asr/quality.py` and are echoed into every output.

| Flag | Fires when |
|---|---|
| `empty_output` | The transcript is empty or whitespace |
| `missing_segment_timestamps` | There are no segments, or any segment lacks a start time |
| `repeated_word_run` | The same word occurs 4 times in a row (a common decoding loop) |
| `implausible_chars_per_second` | Non-space characters per second of audio is below 1.0 or above 30.0 (only checked on clips of 2 s or more) |

Passing the screen means no check fired. It does not mean the transcript is correct.

## Evaluating accuracy

```bash
pashto-asr evaluate --reference test.tsv --hyp-dir drafts/ --json-out report.json
```

The reference is a TSV with `path` and `sentence` columns (the Common Voice layout). Each row's file stem is matched to `<stem>.transcript.json` in `--hyp-dir`; missing outputs count as empty. Text is NFC-normalised, punctuation is stripped, and corpus-level WER and CER are reported (total edit distance over total reference length).

## Benchmark: Common Voice Spontaneous Speech, Pashto (12 Nov 2026)

Conversational audio is what this tool is for, so the first benchmark uses spontaneous rather than read speech.

| Item | Value |
|---|---|
| Dataset | Mozilla Common Voice Spontaneous Speech 5.0, Pashto (`sps-corpus-5.0-2026-09-11-ps`), CC0 |
| Subset | Official `test` split, every row with a non-empty human transcription: **133 clips, 101.3 minutes** |
| Model | `ihanif/exp_008_base_cv24_vanilla` @ `fb28d0b4afb14184aaad9846013267305ba44d3e` (default) |
| Hardware | CPU only, 12 threads, offline (`HF_HUB_OFFLINE=1`); 8 min 55 s wall time (about 11x real time) |
| Scoring | Corpus-level WER/CER after NFC normalisation and punctuation stripping (`pashto-asr evaluate`) |

| Drafts | Clips | WER | CER |
|---|---|---|---|
| All | 133 | **70.2%** | **58.2%** |
| Passed the quality screen | 75 | 44.3% | 25.1% |
| Flagged by the quality screen | 58 | 73.9% | 62.8% |

What this shows:

- **The default model is not good enough for unsupervised use on spontaneous Pashto.** Roughly seven words in ten are wrong across the full set. Treat every draft as a starting point for a human transcriber.
- **The quality screen does useful triage.** Drafts it passes have about 2.5x lower character error than drafts it flags, so a reviewer can start with the passed drafts and expect the flagged ones to need near-full re-transcription. The flags fired were `missing_segment_timestamps` (57 clips) and `repeated_word_run` (14).
- Flagged clips are mostly the longer recordings, which is why they dominate the corpus-level figure.

These are numbers for one model on one release. They are not a claim about Pashto ASR in general.

### Reproducing it

The dataset may not be re-hosted, so it is not in this repository and neither are the transcripts. Download the Pashto Spontaneous Speech release yourself from the [Mozilla Data Collective](https://mozilladatacollective.com/) (free account, accept the terms), extract it, then:

```bash
python scripts/cv_spontaneous_subset.py <extracted_release_dir> bench/
HF_HUB_OFFLINE=1 pashto-asr transcribe bench/test_clips --out bench/drafts --threads 12
pashto-asr evaluate --reference bench/test_reference.tsv --hyp-dir bench/drafts
python scripts/screen_split_eval.py bench/test_reference.tsv bench/drafts
```

Run the model once without `HF_HUB_OFFLINE` first so it is cached.

## Limitations

- **Accuracy is low on spontaneous speech.** On the benchmark above the default model scores 70.2% WER. Read-speech accuracy may be better but has not been measured here.
- **An early private test failed the quality screen.** In a small internal trial on real conversational audio, the output was flagged by the screen. That trial is not published and is not a benchmark.
- Dari is not supported yet in any validated sense. The language code can be changed, but nothing has been tested.
- The screen is heuristic. It will miss fluent but wrong text, and may flag legitimate output.
- Whisper-family models can hallucinate text on silence, noise and music.
- Dialect coverage is unknown. Regional variation in Pashto is large.
- CPU inference is slow on long recordings.

## Roadmap

1. ~~Publish a Common Voice Pashto benchmark~~ (done, spontaneous speech, above). Next: the read-speech test split.
2. Compare larger open models (Whisper small/medium/large fine-tunes) on the same subset, to see how much accuracy CPU users give up.
3. Dari support: model selection, a benchmark, and a language-specific screen.
4. Optional reviewer workflow for correcting drafts, ordered by quality-screen result.

## Credits and licences

- Default model: [`ihanif/exp_008_base_cv24_vanilla`](https://huggingface.co/ihanif/exp_008_base_cv24_vanilla) on Hugging Face, pinned to revision `fb28d0b4afb14184aaad9846013267305ba44d3e`. It is a Whisper-family fine-tune, published under Apache-2.0 according to its model card metadata; see the card for its author and training data. Whisper itself is by OpenAI (MIT licence).
- Common Voice data is by Mozilla and its contributors (CC0); check the release you download.
- Built on PyAV, PyTorch and Hugging Face Transformers.
- This code is licensed under Apache-2.0 (see `LICENSE`). Models are separate works with their own licences.

See also [ETHICS.md](ETHICS.md).
