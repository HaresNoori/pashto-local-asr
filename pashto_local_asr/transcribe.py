"""Local CPU transcription with a pinned Hugging Face model (Whisper or CTC)."""
import json
import os
import sys
import time
from pathlib import Path

from . import __version__
from .audio import SAMPLE_RATE, decode
from .hashing import sha256_file
from .quality import screen

# Default: a small community Whisper fine-tune for Pashto (70.2% WER on the
# spontaneous-speech benchmark in the README; see Limitations).
DEFAULT_MODEL = "ihanif/exp_008_base_cv24_vanilla"
DEFAULT_REVISION = "fb28d0b4afb14184aaad9846013267305ba44d3e"
DEFAULT_LANGUAGE = "ps"
SCHEMA_VERSION = 1
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".flac", ".aac", ".webm", ".amr"}


WHISPER = "whisper"
CTC = "ctc"
# Fallback list used only when the config carries no `architectures` entry.
_CTC_MODEL_TYPES = {"wav2vec2", "wav2vec2-bert", "wav2vec2-conformer", "hubert", "wavlm",
                    "data2vec-audio", "unispeech", "unispeech-sat", "sew", "sew-d", "mctct"}


def detect_architecture(config):
    """Classify a model config as 'whisper' (seq2seq) or 'ctc'.

    `config` may be a transformers config object or a plain dict with `model_type`
    and optionally `architectures`. Raises ValueError for anything else, so an
    unsupported model fails loudly instead of being run the wrong way."""
    get = config.get if isinstance(config, dict) else (lambda k, d=None: getattr(config, k, d))
    model_type = get("model_type", None)
    architectures = get("architectures", None) or []
    if model_type == "whisper" or any(a.startswith("WhisperFor") for a in architectures):
        return WHISPER
    if any(a.endswith("ForCTC") for a in architectures) or model_type in _CTC_MODEL_TYPES:
        return CTC
    raise ValueError(
        f"Unsupported ASR architecture (model_type={model_type!r}, architectures={architectures!r}); "
        "supported: Whisper and CTC models (wav2vec2, w2v-BERT, HuBERT, WavLM, ...).")


def _from_pretrained(cls, model, revision):
    """Prefer safetensors. If the repo only ships a pickled .bin checkpoint, fall back
    to it; transformers loads those with torch.load(weights_only=True)."""
    try:
        return cls.from_pretrained(model, revision=revision, use_safetensors=True,
                                   trust_remote_code=False)
    except OSError as error:
        if "safetensors" not in str(error).lower():
            raise
        print(f"note: {model} has no safetensors weights; loading .bin with weights_only",
              file=sys.stderr)
        return cls.from_pretrained(model, revision=revision, use_safetensors=False,
                                   trust_remote_code=False)


def repair_whisper_generation_config(gen, tokenizer):
    """Fill in Whisper generation-config fields that old community fine-tunes omit (no
    generation_config.json in the repo), derived from the model's own tokenizer, so that
    language/task forcing and timestamp decoding work. Existing values are never replaced
    (a one-element eos_token_id list is only unwrapped to an int).
    Returns the names of the fields that were filled."""
    filled = []

    def fill(name, value):
        if getattr(gen, name, None) in (None, {}):
            setattr(gen, name, value)
            filled.append(name)

    from transformers.models.whisper.tokenization_whisper import LANGUAGES
    to_id = tokenizer.convert_tokens_to_ids
    fill("no_timestamps_token_id", to_id("<|notimestamps|>"))
    fill("lang_to_id", {f"<|{code}|>": to_id(f"<|{code}|>") for code in LANGUAGES
                        if to_id(f"<|{code}|>") != tokenizer.unk_token_id})
    fill("task_to_id", {"transcribe": to_id("<|transcribe|>"), "translate": to_id("<|translate|>")})
    fill("is_multilingual", True)
    # transformers' timestamp logits processor slices with eos_token_id and needs an int;
    # some fine-tunes ship a one-element list, which crashes on some clips.
    eos = getattr(gen, "eos_token_id", None)
    if isinstance(eos, (list, tuple)) and len(eos) == 1:
        gen.eos_token_id = int(eos[0])
        filled.append("eos_token_id(list->int)")
    return filled


def load_pipeline(model=DEFAULT_MODEL, revision=DEFAULT_REVISION, threads=2):
    """Load any supported ASR model on CPU. The model is fetched from the Hugging Face
    Hub on first use unless HF_HUB_OFFLINE=1 and it is already cached. The returned
    pipeline carries `.architecture` ('whisper' or 'ctc')."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import (AutoConfig, AutoModelForCTC, AutoProcessor,
                              WhisperForConditionalGeneration, WhisperProcessor, pipeline)

    torch.set_num_threads(threads)
    config = AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=False)
    architecture = detect_architecture(config)
    if architecture == WHISPER:
        net = _from_pretrained(WhisperForConditionalGeneration, model, revision)
        processor = WhisperProcessor.from_pretrained(
            model, revision=revision, trust_remote_code=False)
        filled = repair_whisper_generation_config(net.generation_config, processor.tokenizer)
        if filled:
            print(f"note: {model} has an incomplete generation config; filled {filled} from its tokenizer",
                  file=sys.stderr)
    else:
        net = _from_pretrained(AutoModelForCTC, model, revision)
        processor = AutoProcessor.from_pretrained(
            model, revision=revision, trust_remote_code=False)
    net.eval()
    pipe = pipeline(
        "automatic-speech-recognition", model=net,
        tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
        device=-1, chunk_length_s=30, stride_length_s=5)
    pipe.architecture = architecture
    return pipe


def model_card_info(model, revision):
    """Best-effort licence / training-dataset info from the Hub card (needs network)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model, revision=revision)
        card = info.card_data.to_dict() if info.card_data else {}
        return {"license": card.get("license"), "datasets": card.get("datasets"),
                "base_model": card.get("base_model"), "gated": info.gated}
    except Exception as error:  # offline or private: not fatal
        return {"license": None, "error": str(error)}


def build_record(audio_path, digest, duration, model, revision, language, text, segments,
                 architecture=WHISPER):
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
        "quality_screen": screen(text, segments, duration,
                                timestamps_expected=architecture == WHISPER or bool(segments)),
    }


def transcribe_file(pipe, audio_path, model=DEFAULT_MODEL, revision=DEFAULT_REVISION,
                    language=DEFAULT_LANGUAGE):
    audio_path = Path(audio_path)
    digest = sha256_file(audio_path)
    samples = decode(audio_path)
    architecture = detect_architecture(pipe.model.config)
    audio = {"array": samples, "sampling_rate": SAMPLE_RATE}
    if architecture == WHISPER:
        result = pipe(audio, return_timestamps=True,
                      generate_kwargs={"language": language, "task": "transcribe",
                                       "do_sample": False, "num_beams": 1})
    else:
        # CTC: no language/task forcing. Word offsets come cheaply from the CTC frames;
        # if the tokenizer cannot provide them, fall back to text only.
        try:
            result = pipe(audio, return_timestamps="word")
        except Exception:
            result = pipe(audio)
    segments = []
    for chunk in result.get("chunks", []):
        stamp = list(chunk.get("timestamp") or (None, None))
        segments.append({"timestamp": stamp, "text": chunk.get("text", "")})
    return build_record(audio_path, digest, len(samples) / SAMPLE_RATE, model, revision,
                        language, result.get("text", ""), segments, architecture)


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


RUN_META_NAME = "run_meta.json"


def update_run_meta(folder, model, revision, architecture, threads, processed, audio_seconds,
                    wall_seconds, load_seconds, params=None, card=None):
    """Write/merge `run_meta.json` in the output folder. Timing accumulates across resumed
    runs of the same model+revision (processed clips, audio seconds and wall seconds add up);
    unknown keys already in the file (e.g. added by hand) are preserved."""
    path = Path(folder) / RUN_META_NAME
    meta = {}
    if path.is_file():
        meta = json.loads(path.read_text(encoding="utf-8"))
        if (meta.get("model"), meta.get("revision")) != (model, revision):
            meta = {}
    meta.update({
        "model": model, "revision": revision, "architecture": architecture,
        "threads": threads, "tool_version": __version__,
        "clips_transcribed": meta.get("clips_transcribed", 0) + processed,
        "audio_seconds": round(meta.get("audio_seconds", 0.0) + audio_seconds, 3),
        "wall_seconds": round(meta.get("wall_seconds", 0.0) + wall_seconds, 3),
        "model_load_seconds": round(load_seconds, 3),
    })
    if params is not None:
        meta["parameters"] = params
    if card:
        meta["model_card"] = card
    write_json(path, meta)
    return meta
