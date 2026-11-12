import hashlib
import json
import math
import struct
import wave

import pytest

from pashto_local_asr import evaluate as ev
from pashto_local_asr.hashing import sha256_file
from pashto_local_asr.quality import has_repeated_run, screen
from pashto_local_asr.transcribe import build_record, output_path_for

GOOD_SEGS = [{"timestamp": [0.0, 3.0], "text": "x"}]


def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world" * 100000)
    assert sha256_file(p) == hashlib.sha256(b"hello world" * 100000).hexdigest()


def test_screen_passes_plausible_output():
    r = screen("one two three four five six seven eight", GOOD_SEGS, 5.0)
    assert r["passed"] and r["flags"] == []


def test_screen_empty_output():
    assert "empty_output" in screen("   ", GOOD_SEGS, 5.0)["flags"]


def test_screen_missing_timestamps():
    assert "missing_segment_timestamps" in screen("a b c d e f g h", [], 5.0)["flags"]
    bad = [{"timestamp": [None, None], "text": "a"}]
    assert "missing_segment_timestamps" in screen("a b c d e f g h", bad, 5.0)["flags"]


def test_screen_repeated_run():
    assert "repeated_word_run" in screen("a b c c c c d e f g", GOOD_SEGS, 5.0)["flags"]
    assert not has_repeated_run("a b c c c d".split())


def test_screen_chars_per_second_bounds():
    assert "implausible_chars_per_second" in screen("abcdefghij" * 40, GOOD_SEGS, 5.0)["flags"]
    assert "implausible_chars_per_second" in screen("ab", GOOD_SEGS, 60.0)["flags"]
    # very short clips are exempt from the rate check
    assert "implausible_chars_per_second" not in screen("abcdefghij" * 40, GOOD_SEGS, 1.0)["flags"]


def test_wer_cer_maths():
    assert ev.wer("a b c d", "a b c d") == 0
    assert ev.wer("a b c d", "a x c") == pytest.approx(2 / 4)  # 1 sub + 1 deletion
    assert ev.cer("abcd", "abxd") == pytest.approx(1 / 4)
    assert ev.wer("a b", "") == 1.0
    assert ev.edit_distance("kitten", "sitting") == 3


def test_normalize_strips_punctuation_and_whitespace():
    assert ev.normalize("  سلام،   نړی۔ ") == \
        "سلام نړی"


def test_basic_level_keeps_spelling_variants():
    assert ev.normalize("ي ی") == "ي ی"


def test_pashto_level_unifies_variants_only():
    # ي -> ی, ك -> ک, ZWNJ and tatweel removed, harakat stripped, digits to ASCII
    assert ev.normalize("كي", "pashto") == "کی"
    assert ev.normalize("م‌ـن", "pashto") == "من"
    assert ev.normalize("بَلّ", "pashto") == "بل"
    assert ev.normalize("۱۲ ١", "pashto") == "12 1"
    # distinct Pashto letters survive: ې, ۍ, ئ (precomposed, not decomposed), ګ, ښ
    kept = "ې ۍ ئ ګ ښ آ"
    assert ev.normalize(kept, "pashto") == kept


def test_pashto_level_changes_scores_only_when_asked():
    ref, hyp = "زمونږ کورني", "زمونږ کورنی"
    assert ev.wer(ref, hyp) == 0.5
    assert ev.wer(ref, hyp, "pashto") == 0.0


def test_unknown_level_rejected():
    with pytest.raises(ValueError):
        ev.normalize("x", "aggressive")


def test_evaluate_against_tsv(tmp_path):
    ref = tmp_path / "ref.tsv"
    ref.write_text("path\tsentence\nclip1.mp3\ta b c d\nclip2.mp3\te f\n", encoding="utf-8")
    (tmp_path / "clip1.transcript.json").write_text(json.dumps({"text": "a b x d"}))
    rep = ev.evaluate(ref, tmp_path)  # clip2 has no hypothesis -> counts as empty
    assert rep["missing_hypotheses"] == ["clip2"]
    assert rep["wer"] == pytest.approx((1 + 2) / 6)


def test_output_schema():
    rec = build_record("/x/clip.wav", "ab" * 32, 4.0, "org/model", "deadbeef", "ps",
                       "one two three four five six", GOOD_SEGS)
    for key in ["schema_version", "status", "source_audio", "sha256", "duration_seconds",
                "model", "revision", "language", "tool", "created_at", "text",
                "segments", "quality_screen"]:
        assert key in rec
    assert rec["status"] == "unverified_draft"
    assert rec["source_audio"] == "clip.wav"  # filename only, no directory leak
    assert {"passed", "flags", "metrics", "thresholds"} <= set(rec["quality_screen"])
    json.dumps(rec)


def test_output_path(tmp_path):
    assert output_path_for(tmp_path / "a.wav").name == "a.transcript.json"
    assert output_path_for(tmp_path / "a.wav", tmp_path / "o").parent == tmp_path / "o"


def test_decode_sine_wave(tmp_path):
    pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    from pashto_local_asr.audio import decode
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 8000)))
            for i in range(8000)))
    samples = decode(path)
    assert samples.dtype == np.float32
    assert abs(len(samples) - 16000) < 200  # 1 s resampled to 16 kHz
