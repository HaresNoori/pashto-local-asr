"""Local CPU transcription with a pinned Hugging Face Whisper-family model."""
import json
import os
import time
from pathlib import Path

from . import __version__
from .audio import SAMPLE_RATE, decode
from .hashing import sha256_file
from .quality import screen

# Default: a community Whisper fine-tune for Pashto. Its accuracy on
# conversational audio is NOT validated; see README Limitations.
DEFAULT_MODEL = "ihanif/exp_008_base_cv24_vanilla"
DEFAULT_REVISION = "fb28d0b4afb14184aaad9846013267305ba44d3e"
DEFAULT_LANGUAGE = "ps"
SCHEMA_VERSION = 1
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".flac", ".aac", ".webm", ".amr"}


def load_pipeline(model=DEFAULT_MODEL, revision=DEFAULT_REVISION, threads=2):
    """Load the model on CPU. The model is fetched from the Hugging Face Hub on
    first use unless HF_HUB_OFFLINE=1 and it is already cached."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

    torch.set_num_threads(threads)
    net = WhisperForConditionalGeneration.from_pretrained(
        model, revision=revision, use_safetensors=True, trust_remote_code=False)
    processor = WhisperProcessor.from_pretrained(
        model, revision=revision, trust_remote_code=False)
    return pipeline(
        "automatic-speech-recognition", model=net,
        tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
        device=-1, chunk_length_s=30, stride_length_s=5)


def build_record(audio_path, digest, duration, model, revision, language, text, segments):
    """Assemble the output document. Pure function, so the schema is testable offline."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "unverified_draft",
        "source_audio": Path(audio_path).name,
        "sha256": digest,
        "duration_seconds": round(duration, 3),
        "model": model,
        "revision": revision,
        "language": language,
        "tool": {"name": "pashto-local-asr", "version": __version__},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "text": text,
        "segments": segments,
        "quality_screen": screen(text, segments, duration),
    }


def transcribe_file(pipe, audio_path, model=DEFAULT_MODEL, revision=DEFAULT_REVISION,
                    language=DEFAULT_LANGUAGE):
    audio_path = Path(audio_path)
    digest = sha256_file(audio_path)
    samples = decode(audio_path)
    result = pipe({"array": samples, "sampling_rate": SAMPLE_RATE}, return_timestamps=True,
                  generate_kwargs={"language": language, "task": "transcribe",
                                   "do_sample": False, "num_beams": 1})
    segments = []
    for chunk in result.get("chunks", []):
        stamp = list(chunk.get("timestamp") or (None, None))
        segments.append({"timestamp": stamp, "text": chunk.get("text", "")})
    return build_record(audio_path, digest, len(samples) / SAMPLE_RATE, model, revision,
                        language, result.get("text", ""), segments)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def find_audio(target):
    target = Path(target)
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)


def output_path_for(audio_path, out_dir=None):
    audio_path = Path(audio_path)
    folder = Path(out_dir) if out_dir else audio_path.parent
    return folder / (audio_path.stem + ".transcript.json")
