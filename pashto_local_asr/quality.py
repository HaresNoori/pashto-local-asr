"""Heuristic quality screen. It catches obvious failures only.

Passing the screen does NOT mean a transcript is correct; it means none of
the checks below fired. Every output remains an unverified draft.
"""

# Thresholds (module-level so they are easy to read and override in tests).
REPEAT_RUN_LENGTH = 4        # this many identical consecutive words is a loop
MIN_CHARS_PER_SECOND = 1.0   # below this, text is implausibly sparse for the audio
MAX_CHARS_PER_SECOND = 30.0  # above this, text is implausibly dense (hallucination)
MIN_DURATION_FOR_RATE_CHECK = 2.0  # seconds; rate is meaningless on very short clips


def has_repeated_run(words, run_length=REPEAT_RUN_LENGTH):
    if run_length < 2:
        raise ValueError("run_length must be >= 2")
    for i in range(len(words) - run_length + 1):
        if len(set(words[i:i + run_length])) == 1:
            return True
    return False


def screen(text, segments, duration_seconds):
    """Return {'passed': bool, 'flags': [...], 'metrics': {...}, 'thresholds': {...}}."""
    flags = []
    text = text or ""
    segments = segments or []
    words = text.split()

    if not text.strip():
        flags.append("empty_output")

    if not segments or any(
        not seg.get("timestamp") or seg["timestamp"][0] is None for seg in segments
    ):
        flags.append("missing_segment_timestamps")

    if has_repeated_run(words):
        flags.append("repeated_word_run")

    chars = len("".join(words))
    cps = chars / duration_seconds if duration_seconds and duration_seconds > 0 else None
    if text.strip() and cps is not None and duration_seconds >= MIN_DURATION_FOR_RATE_CHECK:
        if cps < MIN_CHARS_PER_SECOND or cps > MAX_CHARS_PER_SECOND:
            flags.append("implausible_chars_per_second")

    return {
        "passed": not flags,
        "flags": flags,
        "metrics": {
            "characters": chars,
            "words": len(words),
            "chars_per_second": None if cps is None else round(cps, 3),
        },
        "thresholds": {
            "repeat_run_length": REPEAT_RUN_LENGTH,
            "min_chars_per_second": MIN_CHARS_PER_SECOND,
            "max_chars_per_second": MAX_CHARS_PER_SECOND,
            "min_duration_for_rate_check_seconds": MIN_DURATION_FOR_RATE_CHECK,
        },
    }
