"""Architecture dispatch, CTC screening and model comparison. No model download needed."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pashto_local_asr.quality import screen
from pashto_local_asr.transcribe import build_record, detect_architecture, update_run_meta


def test_detect_architecture_whisper_and_ctc():
    assert detect_architecture({"model_type": "whisper",
                                "architectures": ["WhisperForConditionalGeneration"]}) == "whisper"
    assert detect_architecture({"model_type": "wav2vec2",
                                "architectures": ["Wav2Vec2ForCTC"]}) == "ctc"
    assert detect_architecture({"model_type": "wav2vec2-bert",
                                "architectures": ["Wav2Vec2BertForCTC"]}) == "ctc"
    assert detect_architecture({"model_type": "hubert"}) == "ctc"  # no architectures listed


def test_detect_architecture_rejects_unsupported():
    with pytest.raises(ValueError):
        detect_architecture({"model_type": "bert", "architectures": ["BertForMaskedLM"]})


def test_detect_architecture_accepts_config_object():
    assert detect_architecture(SimpleNamespace(model_type="whisper", architectures=None)) == "whisper"


def test_ctc_without_timestamps_not_flagged():
    res = screen("سلام ورور څنګه یې", [], 5.0, timestamps_expected=False)
    assert res["passed"] and res["not_applicable"] == ["missing_segment_timestamps"]
    ok = screen("سلام ورور څنګه یې", [{"timestamp": [0, 1], "text": "x"}], 5.0)
    assert "not_applicable" not in ok


def test_build_record_ctc_schema_and_screen(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    ctc = build_record(f, "d", 5.0, "m", "r", "ps", "سلام ورور څنګه یې", [], architecture="ctc")
    assert ctc["quality_screen"]["passed"]
    whisper = build_record(f, "d", 5.0, "m", "r", "ps", "سلام ورور څنګه یې", [])
    assert "missing_segment_timestamps" in whisper["quality_screen"]["flags"]
    assert set(ctc) == set(whisper)  # output schema identical across architectures


def test_run_meta_accumulates_and_resets_on_model_change(tmp_path):
    update_run_meta(tmp_path, "m", "r", "ctc", 4, 2, 10.0, 5.0, 1.0, params=100)
    meta = update_run_meta(tmp_path, "m", "r", "ctc", 4, 3, 6.0, 2.5, 0.5)
    assert meta["clips_transcribed"] == 5 and meta["audio_seconds"] == 16.0
    assert meta["wall_seconds"] == 7.5 and meta["parameters"] == 100
    other = update_run_meta(tmp_path, "m2", "r", "whisper", 4, 1, 1.0, 1.0, 0.1)
    assert other["clips_transcribed"] == 1


def _load_compare():
    spec = importlib.util.spec_from_file_location(
        "compare_models", Path(__file__).parents[1] / "scripts" / "compare_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_table(tmp_path):
    cm = _load_compare()
    ref = tmp_path / "ref.tsv"
    ref.write_text("path\tsentence\na.mp3\tسلام ورور\nb.mp3\tښه یم\n", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    for stem, text in (("a", "سلام ورور"), ("b", "")):
        rec = build_record(tmp_path / f"{stem}.mp3", "d", 5.0, "org/m", "abcdef123456", "ps", text,
                           [{"timestamp": [0, 1], "text": text}])
        (run / f"{stem}.transcript.json").write_text(json.dumps(rec), encoding="utf-8")
    (run / "run_meta.json").write_text(json.dumps(
        {"architecture": "whisper", "parameters": 72593920, "audio_seconds": 10.0,
         "wall_seconds": 5.0, "model_card": {"license": "mit"}}), encoding="utf-8")
    row = cm.summarise(ref, run)
    assert row["wer"] == pytest.approx(0.5) and row["pass_rate"] == pytest.approx(0.5)
    assert row["wer_passed"] == 0.0 and row["rtf"] == 0.5
    table = cm.render_table([row])
    assert "org/m" in table and "abcdef12" in table and "73M" in table and "mit" in table


def test_repair_whisper_generation_config_fills_only_missing():
    from pashto_local_asr.transcribe import repair_whisper_generation_config

    class Tok:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return {"<|notimestamps|>": 50363, "<|ps|>": 50340, "<|transcribe|>": 50359,
                    "<|translate|>": 50358}.get(token, 0)

    gen = SimpleNamespace(no_timestamps_token_id=None, lang_to_id=None, task_to_id=None,
                          is_multilingual=None)
    filled = repair_whisper_generation_config(gen, Tok())
    assert gen.no_timestamps_token_id == 50363 and gen.lang_to_id == {"<|ps|>": 50340}
    assert gen.task_to_id["transcribe"] == 50359 and "is_multilingual" in filled
    complete = SimpleNamespace(no_timestamps_token_id=7, lang_to_id={"<|x|>": 1},
                               task_to_id={"transcribe": 2}, is_multilingual=True)
    assert repair_whisper_generation_config(complete, Tok()) == []
    assert complete.no_timestamps_token_id == 7


def test_repair_unwraps_single_element_eos_list():
    from pashto_local_asr.transcribe import repair_whisper_generation_config

    class Tok:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return 1

    gen = SimpleNamespace(no_timestamps_token_id=1, lang_to_id={"a": 1}, task_to_id={"b": 1},
                          is_multilingual=True, eos_token_id=[50257])
    assert repair_whisper_generation_config(gen, Tok()) == ["eos_token_id(list->int)"]
    assert gen.eos_token_id == 50257


def test_compare_common_only(tmp_path):
    cm = _load_compare()
    ref = tmp_path / "ref.tsv"
    ref.write_text("path\tsentence\na.mp3\tسلام ورور\nb.mp3\tښه یم\n", encoding="utf-8")
    runs = []
    for name, stems in (("full", "ab"), ("partial", "a")):
        run = tmp_path / name
        run.mkdir()
        for stem in stems:
            rec = build_record(tmp_path / f"{stem}.mp3", "d", 5.0, name, "r", "ps", "سلام ورور",
                               [{"timestamp": [0, 1], "text": "x"}])
            (run / f"{stem}.transcript.json").write_text(json.dumps(rec), encoding="utf-8")
        runs.append(run)
    assert cm.common_stems(runs) == {"a"}
    row = cm.summarise(ref, runs[1], cm.common_stems(runs))
    assert row["n"] == 1 and row["missing"] == 0 and row["wer"] == 0.0
    assert cm.summarise(ref, runs[1])["missing"] == 1
