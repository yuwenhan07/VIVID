"""Task-local Whisper worker; also executable by an isolated ASR Python environment."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time


def _faster_audio(path):
    import numpy as np

    # Decode explicitly: avoids PyAV/FFmpeg ABI differences observed in baseline runs.
    r = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        timeout=20,
        check=True,
    )
    return np.frombuffer(r.stdout, dtype="<f4").copy()


def main():
    model = None
    engine = os.getenv("VIVID_ASR_BACKEND", "faster-whisper")
    threads = max(1, min(8, int(os.getenv("VIVID_ASR_THREADS", "2"))))
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                started = time.monotonic()
                if engine == "faster-whisper":
                    if model is None:
                        from faster_whisper import WhisperModel

                        model = WhisperModel(
                            os.getenv("VIVID_WHISPER_MODEL", "base"),
                            device="cpu",
                            compute_type="int8",
                            cpu_threads=threads,
                            num_workers=1,
                            local_files_only=os.getenv("VIVID_ALLOW_MODEL_DOWNLOAD") != "1",
                        )
                    segments, info = model.transcribe(
                        _faster_audio(request["path"]),
                        language=request.get("language"),
                        beam_size=1,
                        temperature=0,
                        condition_on_previous_text=False,
                        word_timestamps=request.get("word_timestamps", False),
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=1000, speech_pad_ms=400),
                    )
                    rows = []
                    for segment in segments:
                        row = {"start": segment.start, "end": segment.end, "text": segment.text}
                        if request.get("word_timestamps"):
                            row["words"] = [
                                {
                                    "start": w.start,
                                    "end": w.end,
                                    "word": w.word,
                                    "probability": w.probability,
                                }
                                for w in (segment.words or [])
                            ]
                        rows.append(row)
                    result = {
                        "text": " ".join(s["text"].strip() for s in rows),
                        "segments": rows,
                        "language": info.language,
                    }
                elif engine == "whisper":
                    if model is None:
                        import whisper

                        name = os.getenv("VIVID_WHISPER_MODEL", "base")
                        cache = os.path.join(
                            os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "whisper"
                        )
                        if name in whisper._MODELS:
                            weights = os.path.join(cache, whisper._MODELS[name].rsplit("/", 1)[-1])
                            if (
                                not os.path.isfile(weights)
                                and os.getenv("VIVID_ALLOW_MODEL_DOWNLOAD") != "1"
                            ):
                                raise RuntimeError(
                                    "Whisper weights are not cached; downloads require "
                                    "VIVID_ALLOW_MODEL_DOWNLOAD=1"
                                )
                        elif not os.path.isfile(name):
                            raise RuntimeError("Unknown Whisper model")
                        import torch

                        torch.set_num_threads(threads)
                        model = whisper.load_model(name, device="cpu", download_root=cache)
                    options = dict(language=request.get("language"), fp16=False, verbose=None)
                    if request.get("word_timestamps"):
                        options["word_timestamps"] = True
                    result = model.transcribe(request["path"], **options)
                else:
                    raise ValueError("VIVID_ASR_BACKEND must be faster-whisper or whisper")
            response = {
                "backend": engine,
                "inference_seconds": time.monotonic() - started,
                "text": result.get("text", ""),
                "language": result.get("language"),
                "segments": [
                    {
                        "start": s["start"],
                        "end": s["end"],
                        "text": s["text"],
                        **({"words": s["words"]} if "words" in s else {}),
                    }
                    for s in result.get("segments", [])
                ],
            }
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
