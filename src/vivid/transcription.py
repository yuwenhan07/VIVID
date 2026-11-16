"""Reuse one bounded Whisper process per task/interpreter instead of reloading per window."""

from __future__ import annotations

import atexit
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from vivid.runtime import check_budget, timeout_seconds

_workers = {}
_lock = threading.Lock()


def _request_timeout():
    return max(1, min(180, float(os.getenv("VIVID_ASR_TIMEOUT", "60"))))


def close_workers():
    for worker in list(_workers.values()):
        worker.close()
    _workers.clear()


atexit.register(close_workers)


class _Worker:
    def __init__(self, interpreter):
        self.process = subprocess.Popen(
            [interpreter, str(Path(__file__).with_name("asr_worker.py"))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
        self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()

    def transcribe(self, path, language, word_timestamps=False):
        check_budget()
        deadline = time.monotonic() + timeout_seconds(_request_timeout())
        self.process.stdin.write(
            (
                json.dumps(
                    {"path": str(path), "language": language, "word_timestamps": word_timestamps}
                )
                + "\n"
            ).encode()
        )
        self.process.stdin.flush()
        data = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                check_budget()
                if time.monotonic() >= deadline:
                    raise TimeoutError("Whisper transcription exceeded its remaining time budget")
                if not selector.select(timeout=min(0.25, deadline - time.monotonic())):
                    continue
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("Whisper worker exited without a transcript")
                data.extend(chunk)
                if len(data) > 8 * 1024 * 1024:
                    raise RuntimeError("Whisper transcript exceeds response limit")
                if b"\n" in data:
                    result = json.loads(data)
                    if "error" in result:
                        raise RuntimeError(result["error"])
                    return result


def transcribe(path: Path, start: float, language: str | None, word_timestamps=False):
    configured = os.getenv("VIVID_MEDIA_PYTHON")
    candidates = [configured] if configured else [sys.executable, shutil.which("python3")]
    errors, attempted = [], set()
    while not _lock.acquire(timeout=0.1):
        check_budget()
    try:
        from vivid.transcript_cache import profile

        model_identity = json.dumps(profile()["weights"], sort_keys=True)
        for interpreter in candidates:
            if not interpreter or interpreter in attempted or not Path(interpreter).is_file():
                continue
            attempted.add(interpreter)
            key = tuple(
                [interpreter]
                + [
                    os.getenv(name, "")
                    for name in (
                        "VIVID_WHISPER_MODEL",
                        "XDG_CACHE_HOME",
                        "HF_HOME",
                        "HF_HUB_CACHE",
                        "HUGGINGFACE_HUB_CACHE",
                        "VIVID_ALLOW_MODEL_DOWNLOAD",
                        "PYTHONPATH",
                        "VIVID_ASR_BACKEND",
                        "VIVID_ASR_THREADS",
                    )
                ]
                + [model_identity]
            )
            for old_key, old_worker in list(_workers.items()):
                if old_key[0] == interpreter and old_key != key:
                    old_worker.close()
                    del _workers[old_key]
            worker = _workers.get(key)
            try:
                if worker is None or worker.process.poll() is not None:
                    worker = _workers[key] = _Worker(interpreter)
                result = worker.transcribe(path, language, word_timestamps)
                for segment in result.get("segments", []):
                    segment["source_start"] = start + segment["start"]
                    segment["source_end"] = start + segment["end"]
                    for word in segment.get("words", []):
                        word["source_start"] = start + word["start"]
                        word["source_end"] = start + word["end"]
                return {
                    "requested": True,
                    "available": True,
                    "device": "cpu",
                    "model": os.getenv("VIVID_WHISPER_MODEL", "base"),
                    "python": interpreter,
                    "word_timestamps_requested": word_timestamps,
                    "word_timestamps_available": word_timestamps
                    and any(s.get("words") for s in result.get("segments", [])),
                    "timing_semantics": "ASR estimates; not forced alignment. Confirm boundaries locally.",
                    **result,
                }
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append({"python": interpreter, "error": str(exc)})
                if worker is not None:
                    worker.close()
                    _workers.pop(key, None)
                if isinstance(exc, TimeoutError):
                    break  # Do not repeat an expensive timeout in another interpreter.
            except BaseException:
                if worker is not None:
                    worker.close()
                    _workers.pop(key, None)
                raise
    finally:
        _lock.release()
    return {
        "requested": True,
        "available": False,
        "errors": errors,
        "retryable": any("time" in e["error"].lower() for e in errors),
        "error_kind": "asr_timeout"
        if any("time budget" in e["error"] for e in errors)
        else "asr_unavailable",
        "hint": "Use VIVID_MEDIA_PYTHON with the configured ASR backend and cached weights. Retry only the pending short intervals or use native tools. "
        "Transcription is derived evidence; native audio was not delivered.",
    }
