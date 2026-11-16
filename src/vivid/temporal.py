"""Exact interval arithmetic for native editing scripts, without a project workflow."""

from __future__ import annotations

import math
from fractions import Fraction

from vivid.errors import VividError


def _number(value, field, *, positive=False):
    try:
        finite = math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        finite = False
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not finite
        or (value <= 0 if positive else value < 0)
    ):
        raise VividError(
            "invalid_time",
            f"{field} must be a finite {'positive' if positive else 'nonnegative'} number",
        )
    return Fraction(str(value))


def _seconds(value):
    try:
        result = float(value)
    except OverflowError as exc:
        raise VividError(
            "invalid_time", "Computed time exceeds the supported numeric range"
        ) from exc
    if not math.isfinite(result):
        raise VividError("invalid_time", "Computed time exceeds the supported numeric range")
    return result


def compose_segments(segments: list[dict]) -> dict:
    """Compose declared trim/constant-speed segments; this does not inspect an encode.

    source_range is [start, end) in source-relative seconds. speed=2 means
    twice normal playback speed. Optional source_duration bounds the range.
    Fraction arithmetic avoids accumulating binary floating-point rounding.
    """
    if not isinstance(segments, list) or not 1 <= len(segments) <= 10000:
        raise VividError("invalid_segments", "Provide 1–10000 declared segments")
    cursor = Fraction(0)
    result = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise VividError("invalid_segments", f"segments[{index}] must be an object")
        extra = set(segment) - {"source", "source_range", "speed", "source_duration"}
        if extra:
            raise VividError("invalid_segments", f"Unknown segment fields: {sorted(extra)}")
        source = segment.get("source")
        span = segment.get("source_range")
        if not isinstance(source, str) or not source.strip():
            raise VividError("invalid_segments", f"segments[{index}].source must identify a source")
        if not isinstance(span, list) or len(span) != 2:
            raise VividError(
                "invalid_segments", f"segments[{index}].source_range must be [start,end)"
            )
        start, end = (_number(v, f"segments[{index}].source_range") for v in span)
        speed = _number(segment.get("speed", 1), f"segments[{index}].speed", positive=True)
        if end <= start:
            raise VividError("invalid_time", f"segments[{index}] has an empty/reversed range")
        if "source_duration" in segment and end > _number(
            segment["source_duration"], "source_duration"
        ):
            raise VividError("invalid_time", f"segments[{index}] exceeds source_duration")
        stop = cursor + (end - start) / speed
        row = {
            "source": source,
            "source_range": [_seconds(start), _seconds(end)],
            "output": [_seconds(cursor), _seconds(stop)],
        }
        if row["output"][1] <= row["output"][0]:
            raise VividError("invalid_time", "Segment is too small for output timestamp precision")
        if speed != 1:
            row["speed"] = _seconds(speed)
        result.append(row)
        cursor = stop
    return {
        "segments": result,
        "duration": _seconds(cursor),
        "basis": "declared_segments",
        "verified_against_output": False,
        "semantics": "Half-open [start,end), file-relative seconds. No transition/overlay inference; validate actual encoded output separately.",
    }


def map_output_time(timeline: dict, output_time: float) -> dict:
    """Map a declared concat timeline, rejecting gaps, overlap and inconsistent speeds."""
    point = _number(output_time, "output_time")
    if not isinstance(timeline, dict) or not isinstance(timeline.get("segments"), list):
        raise VividError("invalid_mapping", "A segments array is required")
    segments = timeline["segments"]
    if not 1 <= len(segments) <= 10000:
        raise VividError("invalid_mapping", "Provide 1–10000 mapping segments")
    cursor = Fraction(0)
    selected = None
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise VividError("invalid_mapping", "Each mapping segment must be an object")
        source, output = segment.get("source_range"), segment.get("output")
        if (
            not isinstance(source, list)
            or len(source) != 2
            or not isinstance(output, list)
            or len(output) != 2
        ):
            raise VividError("invalid_mapping", "Mappings need source_range and output [start,end)")
        if not isinstance(segment.get("source"), str) or not segment["source"].strip():
            raise VividError("invalid_mapping", "Mapping source identity is required")
        s0, s1 = (_number(v, "source_range") for v in source)
        o0, o1 = (_number(v, "output") for v in output)
        if o0 != cursor or o1 <= o0 or s1 <= s0:
            raise VividError(
                "invalid_mapping",
                "Mapping must start at zero, be contiguous, and use positive intervals",
            )
        speed = _number(segment.get("speed", 1), "speed", positive=True)
        if abs((s1 - s0) / speed - (o1 - o0)) > Fraction(1, 1_000_000_000):
            raise VividError(
                "invalid_mapping", "Output duration conflicts with source range and speed"
            )
        if o0 <= point < o1:
            selected = {
                "segment_index": index,
                "source": segment["source"],
                "source_time": _seconds(s0 + (point - o0) * (s1 - s0) / (o1 - o0)),
                "output_time": _seconds(point),
                "basis": "declared_segments",
                "verified_against_output": False,
            }
        cursor = o1
    if selected is None:
        raise VividError(
            "outside_timeline",
            "Timestamp is outside [0, duration); the final endpoint is exclusive",
        )
    return selected


def map_source_range(timeline, source, start, end):
    """Return every surviving occurrence, including repeated/reordered source clips."""
    map_output_time(timeline, 0)  # Validate the entire declared mapping first.
    a, b = _number(start, "start"), _number(end, "end")
    if b <= a or not isinstance(source, str) or not source:
        raise VividError("invalid_mapping", "Provide source and an ordered source interval")
    result = []
    for index, row in enumerate(timeline["segments"]):
        if row["source"] != source:
            continue
        s0, s1 = map(lambda v: Fraction(str(v)), row["source_range"])
        lo, hi = max(a, s0), min(b, s1)
        if hi <= lo:
            continue
        o0, o1 = map(lambda v: Fraction(str(v)), row["output"])
        scale = (o1 - o0) / (s1 - s0)
        result.append(
            {
                "segment_index": index,
                "source": source,
                "source_range": [_seconds(lo), _seconds(hi)],
                "output": [_seconds(o0 + (lo - s0) * scale), _seconds(o0 + (hi - s0) * scale)],
                "clipped": lo != a or hi != b,
            }
        )
    return result


def map_captions(timeline, captions):
    """Trim/remap source cues through cuts and constant speeds, without inventing text.

    Words use source-relative start/end. Segment-only text crossing a cut is flagged
    for local review rather than silently claiming a complete synchronized sentence.
    """
    map_output_time(timeline, 0)
    if not isinstance(captions, list) or len(captions) > 10000:
        raise VividError("invalid_captions", "Provide at most 10000 source cues")
    result = []
    for cue_index, cue in enumerate(captions):
        if not isinstance(cue, dict) or not isinstance(cue.get("text"), str):
            raise VividError("invalid_captions", "Each cue needs source/start/end/text")
        spans = map_source_range(timeline, cue.get("source"), cue.get("start"), cue.get("end"))
        words = cue.get("words", [])
        if not isinstance(words, list):
            raise VividError("invalid_captions", "words must be an array")
        previous = _number(cue.get("start"), "cue.start")
        cue_end = _number(cue.get("end"), "cue.end")
        for word in words:
            if not isinstance(word, dict) or not isinstance(word.get("word"), str):
                raise VividError("invalid_captions", "Words need start/end/word")
            if _number(word.get("end"), "word.end") <= _number(word.get("start"), "word.start"):
                raise VividError("invalid_captions", "Words need positive source intervals")
            word_start = _number(word["start"], "word.start")
            if word_start < previous or _number(word["end"], "word.end") > cue_end:
                raise VividError("invalid_captions", "Words must be ordered and fit their cue")
            previous = word_start
        for span in spans:
            text, review = cue["text"], span["clipped"]
            if words:
                lo, hi = span["source_range"]
                selected = [w for w in words if w["start"] < hi and w["end"] > lo]
                if not selected:
                    continue
                text = "".join(w["word"] for w in selected).strip()
                review = any(w["start"] < lo or w["end"] > hi for w in selected)
                row = timeline["segments"][span["segment_index"]]
                s0, s1 = row["source_range"]
                o0, o1 = row["output"]
                span["output"] = [
                    o0 + (max(lo, selected[0]["start"]) - s0) * (o1 - o0) / (s1 - s0),
                    o0 + (min(hi, selected[-1]["end"]) - s0) * (o1 - o0) / (s1 - s0),
                ]
            result.append(
                {
                    "start": span["output"][0],
                    "end": span["output"][1],
                    "text": text,
                    "source": cue["source"],
                    "cue_index": cue_index,
                    "segment_index": span["segment_index"],
                    "requires_review": review,
                }
            )
    return sorted(result, key=lambda c: (c["start"], c["end"]))


def boundary_windows(timeline, radius=0.5):
    """Plan small output checks around every declared cut, including head/tail."""
    map_output_time(timeline, 0)
    r = _number(radius, "radius", positive=True)
    duration = Fraction(str(timeline["segments"][-1]["output"][1]))
    points = [Fraction(0)] + [Fraction(str(s["output"][1])) for s in timeline["segments"]]
    return [
        {
            "cut_time": _seconds(p),
            "start": _seconds(max(Fraction(0), p - r)),
            "end": _seconds(min(duration, p + r)),
        }
        for p in points
    ]
