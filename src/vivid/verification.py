"""Deterministic verification of a concrete, hash-bound file; no semantic gates."""

from __future__ import annotations

import contextlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from vivid.errors import VividError
from vivid.memory import MEDIA_EXTENSIONS, probe_media, sha256_file
from vivid.runtime import bounded_lock, check_budget, run_process, stage
from vivid.workspace import _read_object, atomic_json, managed_path


def _copy_file(source: Path, destination: Path) -> None:
    """Copy with the same cooperative deadline used by media subprocesses."""
    check_budget()
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            check_budget()
            outgoing.write(block)
    check_budget()


def verify_file(
    path: Path | str,
    *,
    expected_width=None,
    expected_height=None,
    expected_duration=None,
    expected_audio=None,
    expected_sha256=None,
    duration_tolerance=0.1,
    expected_video_codec=None,
    expected_audio_codec=None,
    expected_sample_rate=None,
    expected_channels=None,
    cache_dir=None,
    _immutable=False,
) -> dict:
    """Verify an isolated copy and bind all checks to its SHA-256 identity.

    The returned path is the requested file, not the temporary copy. A changed source
    fails even when the isolated copy passed. Non-media files receive file/hash checks.
    """
    path = Path(path).expanduser().resolve()
    result = {
        "ok": False,
        "path": str(path),
        "checked_sha256": None,
        "sha256": None,
        "checks": [],
        "scope": "deterministic_file_and_media_only",
    }
    checks = result["checks"]

    def check(name, passed, actual=None, expected=None):
        entry = {"name": name, "passed": bool(passed)}
        if actual is not None:
            entry["actual"] = actual
        if expected is not None:
            entry["expected"] = expected
        checks.append(entry)

    try:
        if not path.is_file():
            raise VividError("file_missing", "Verification requires an existing regular file.")
        expectations = {
            "width": expected_width,
            "height": expected_height,
            "sample_rate": expected_sample_rate,
            "channels": expected_channels,
        }
        for name, value in expectations.items():
            if value is not None and (type(value) is not int or value < 1):
                raise VividError(
                    "invalid_arguments", f"expected_{name} must be a positive integer."
                )
        for name, value in (
            ("expected_duration", expected_duration),
            ("duration_tolerance", duration_tolerance),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise VividError("invalid_arguments", f"{name} must be finite and nonnegative.")
        if expected_audio is not None and type(expected_audio) is not bool:
            raise VividError("invalid_arguments", "expected_audio must be a boolean.")
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in expected_sha256)
        ):
            raise VividError("invalid_arguments", "expected_sha256 must contain 64 hex characters.")
        for name, value in (
            ("expected_video_codec", expected_video_codec),
            ("expected_audio_codec", expected_audio_codec),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise VividError("invalid_arguments", f"{name} must be a nonempty string.")
        context = (
            contextlib.nullcontext(None)
            if _immutable
            else tempfile.TemporaryDirectory(prefix="vivid-verify-")
        )
        with context as folder:
            snapshot = path if _immutable else Path(folder) / ("checked" + path.suffix)
            if not _immutable:
                with stage("verification_copy"):
                    _copy_file(path, snapshot)
            digest = sha256_file(snapshot)
            result.update(sha256=digest, checked_sha256=digest, bytes=snapshot.stat().st_size)
            check("nonempty_file", snapshot.stat().st_size > 0, snapshot.stat().st_size)
            if expected_sha256 is not None:
                check("sha256", digest == expected_sha256.lower(), digest, expected_sha256.lower())
            if path.suffix.lower() == ".json":
                result["format"] = "json"
                if snapshot.stat().st_size > 16 * 1024 * 1024:
                    check("valid_json", False)
                    result["json_error"] = "JSON exceeds the 16 MiB verification limit."
                else:

                    def reject_constant(value):
                        raise ValueError("Nonfinite JSON value: " + value)

                    try:
                        json.loads(
                            snapshot.read_text(encoding="utf-8"), parse_constant=reject_constant
                        )
                        check("valid_json", True)
                    except (ValueError, UnicodeError) as exc:
                        check("valid_json", False)
                        result["json_error"] = str(exc)[:1000]
            media_expected = any(
                value is not None
                for value in (
                    *expectations.values(),
                    expected_duration,
                    expected_audio,
                    expected_video_codec,
                    expected_audio_codec,
                )
            )
            if path.suffix.lower() in MEDIA_EXTENSIONS or media_expected:
                metadata = probe_media(snapshot)
                video = next(
                    (
                        s
                        for s in metadata["streams"]
                        if s.get("codec_type") == "video"
                        and not s.get("disposition", {}).get("attached_pic")
                    ),
                    {},
                )
                audio = next((s for s in metadata["streams"] if s.get("codec_type") == "audio"), {})
                result["media"] = {
                    "kind": metadata["kind"],
                    "duration": metadata["duration"],
                    "width": video.get("width"),
                    "height": video.get("height"),
                    "has_audio": bool(audio),
                    "video_codec": video.get("codec_name"),
                    "audio_codec": audio.get("codec_name"),
                    "sample_rate": int(audio.get("sample_rate", 0)) or None,
                    "channels": audio.get("channels"),
                }
                check("readable_media", bool(video or audio))
                for name, expected in expectations.items():
                    if expected is not None:
                        check(
                            name,
                            result["media"].get(name) == expected,
                            result["media"].get(name),
                            expected,
                        )
                if expected_duration is not None:
                    check(
                        "duration",
                        abs(metadata["duration"] - expected_duration) <= duration_tolerance,
                        metadata["duration"],
                        expected_duration,
                    )
                for name, expected in (
                    ("has_audio", expected_audio),
                    ("video_codec", expected_video_codec),
                    ("audio_codec", expected_audio_codec),
                ):
                    if expected is not None:
                        check(
                            name, result["media"][name] == expected, result["media"][name], expected
                        )
                try:
                    command = [
                        "ffmpeg",
                        "-hide_banner",
                        "-nostdin",
                        "-v",
                        "error",
                        "-xerror",
                        "-i",
                        str(snapshot),
                        "-map",
                        "0:v?",
                        "-map",
                        "0:a?",
                        "-f",
                        "null",
                        "-",
                    ]
                    executable = Path(shutil.which("ffmpeg") or "/missing-ffmpeg")
                    stat = executable.stat() if executable.exists() else None
                    decoder = (
                        [str(executable.resolve()), stat.st_size, stat.st_mtime_ns]
                        if stat
                        else None
                    )
                    cache_path = None
                    if cache_dir is not None:
                        cache_path = managed_path(Path(cache_dir) / (digest + ".json"))
                    lock = (
                        bounded_lock(cache_path.with_suffix(".lock"))
                        if cache_path
                        else contextlib.nullcontext()
                    )
                    with lock:
                        cached = _read_object(cache_path) if cache_path else {}
                        hit = (
                            cached.get("version") == 1
                            and cached.get("sha256") == digest
                            and cached.get("decoder") == decoder
                            and cached.get("full_decode") is True
                        )
                        result["decode_cache_hit"] = hit
                        with stage("full_decode", cache_hit=hit):
                            process = (
                                subprocess.CompletedProcess(command, 0, "", "")
                                if hit
                                else run_process(
                                    command,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=120,
                                )
                            )
                        if cache_path and not hit and process.returncode == 0:
                            atomic_json(
                                cache_path,
                                {
                                    "version": 1,
                                    "sha256": digest,
                                    "decoder": decoder,
                                    "full_decode": True,
                                },
                            )
                    # Every request still checks its own explicit expectations and
                    # source identity. Only the successful full decode is reused.
                except subprocess.TimeoutExpired as exc:
                    raise VividError(
                        "verification_timeout",
                        "Full media decode exceeded 120 seconds; "
                        "decode verification is incomplete.",
                    ) from exc
                check("full_decode", process.returncode == 0)
                if process.returncode:
                    result["decode_error"] = process.stderr[-2000:]
            check("source_unchanged", sha256_file(path) == digest)
        result["ok"] = all(item["passed"] for item in checks)
        if not result["ok"]:
            result["error"] = {
                "code": "verification_failed",
                "message": "One or more deterministic checks failed.",
            }
    except VividError as exc:
        result["error"] = exc.as_dict()["error"]
    except (OSError, ValueError, TypeError) as exc:
        result["error"] = {"code": "verification_failed", "message": str(exc)[:2000]}
    return result
