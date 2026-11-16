"""Small native-script helpers for repair comparisons and cut/subtitle review."""

from __future__ import annotations

import math
import tempfile
import wave
from pathlib import Path

import numpy as np

from vivid.errors import VividError
from vivid.media import _contact_sheet
from vivid.temporal import boundary_windows


def compare_frames(service, source, output, source_time, output_time, roi=None):
    """Compare declared corresponding positions; correspondence is not inferred."""
    results = []
    for path, timestamp in ((source, source_time), (output, output_time)):
        args = {
            "path": path,
            "start": timestamp,
            "end": timestamp + 1,
            "max_frames": 1,
            "width": 640,
        }
        if roi is not None:
            args["roi"] = roi
        result = service.dispatch("inspect_video", args)
        if not result.get("ok"):
            return result
        results.append(result)
    with tempfile.TemporaryDirectory(dir=service.memory.staging_directory()) as folder:
        sheet = Path(folder) / "before_after.png"
        _contact_sheet(
            [
                (
                    Path(r["frames"][0]["preview_path"]),
                    f"{label}: {r['frames'][0]['timestamp']:.6f}s",
                )
                for label, r in zip(("SOURCE", "OUTPUT"), results)
            ],
            sheet,
        )
        return service.memory.register_evidence(
            "comparison",
            results[1]["asset_id"],
            {
                "time_range": [output_time, output_time],
                "source_evidence": [r["frames"][0]["evidence_id"] for r in results],
                "source_versions": [r["version_id"] for r in results],
                "images": [{"path": str(sheet), "mime_type": "image/png"}],
                "semantics": "Declared correspondence only. Judge repair effects and side effects separately; this does not prove perceptual quality or correct localization.",
            },
            [sheet],
        )


def plan_output_checks(timeline=None, captions=(), requirements=(), radius=0.5, duration=None):
    """Return targeted windows for actual-output review, not a success assertion."""
    windows, unplanned = [], []
    if timeline is not None:
        try:
            boundaries = boundary_windows(timeline, radius)
            timeline_duration = timeline["segments"][-1]["output"][1]
            if duration is not None and timeline_duration > duration + 0.1:
                raise VividError("invalid_review", "Timeline exceeds the output duration")
            windows.extend({**w, "reason": "edit_boundary"} for w in boundaries)
            if duration is None:
                duration = timeline_duration
        except VividError as exc:
            unplanned.append({"kind": "timeline", "error": exc.as_dict()["error"]})
    for kind, items in (("subtitle", captions), ("public_requirement", requirements)):
        for index, item in enumerate(items):
            lo, hi = item.get("start"), item.get("end")
            if (
                duration is None
                or any(
                    isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
                    for v in (lo, hi)
                )
                or not 0 <= lo < hi <= duration
            ):
                unplanned.append(
                    {
                        "kind": kind,
                        "index": index,
                        "item": item,
                        "reason": "Provide start and end in output seconds within a known output duration.",
                    }
                )
                continue
            windows.append(
                {
                    "start": max(0, lo - radius),
                    "end": min(duration, hi + radius),
                    "reason": kind,
                    "text": item.get("text", ""),
                    "item_index": index,
                }
            )
    return {
        "windows": windows,
        "unplanned": unplanned,
        "status": (
            "partial" if unplanned and windows else ("unavailable" if unplanned else "planned")
        ),
        "semantic_success": None,
        "guidance": "Inspect actual output. Include caption/dialogue correspondence, subject visibility, required events, and forbidden content; samples alone cannot prove absence.",
    }


def audio_cut_candidates(path, target, *, origin=0, radius=0.25):
    """Rank nearby 10ms PCM16 low-energy windows; not speech/forced alignment."""
    if (
        any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
            for v in (target, origin, radius)
        )
        or origin < 0
        or radius <= 0
        or target < origin
    ):
        raise VividError("invalid_review", "Provide finite source times and a positive radius")
    with wave.open(str(path), "rb") as recording:
        if recording.getsampwidth() != 2:
            raise VividError("invalid_audio", "Use the PCM16 audio_path returned by inspect_audio")
        rate, channels = recording.getframerate(), recording.getnchannels()
        if target >= origin + recording.getnframes() / rate:
            raise VividError("invalid_review", "Target is outside the audio window")
        first = max(0, round((target - origin - radius) * rate))
        if first >= recording.getnframes():
            raise VividError("invalid_review", "Target is outside the audio window")
        recording.setpos(first)
        last = min(recording.getnframes(), round((target - origin + radius) * rate))
        raw = recording.readframes(last - first)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64).reshape(-1, channels) / 32768
    step = max(1, round(rate * 0.01))
    candidates = []
    for i in range(0, len(samples) - step + 1, step):
        candidates.append(
            {
                "source_time": origin + (first + i + step / 2) / rate,
                "rms": float(np.sqrt(np.mean(samples[i : i + step] ** 2))),
            }
        )
    return {
        "candidates": sorted(candidates, key=lambda c: (c["rms"], abs(c["source_time"] - target)))[
            :8
        ],
        "semantics": "Acoustic candidates only; confirm complete meaning and word boundaries locally. Low energy does not establish the intended cut.",
    }
