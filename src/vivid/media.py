"""Bounded, timestamped media inspection; originals are always retained."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .errors import VividError
from .memory import MemoryStore
from .runtime import check_budget, run_process, stage

MAX_VIDEO_SECONDS = 600.0
MAX_AUDIO_SECONDS = 120.0
MAX_FRAMES = 24
MAX_NATIVE_PIXELS = 40_000_000
MAX_AUDIO_VALUES = 16_000_000


def _number(
    value: Any, label: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise VividError("invalid_argument", f"{label} must be a finite number.")
    value = float(value)
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise VividError(
            "invalid_argument",
            f"{label} is outside its permitted bounds.",
            {"minimum": minimum, "maximum": maximum},
        )
    return value


def _run(
    command: list[str], *, timeout: float = 120, stage_name="media_process"
) -> subprocess.CompletedProcess[str]:
    try:
        with stage(stage_name):
            process = run_process(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout
            )
    except FileNotFoundError as error:
        raise VividError(
            "missing_dependency", f"{command[0]} is required in the task environment."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise VividError(
            "media_timeout", f"{command[0]} exceeded its {timeout:g}-second execution limit."
        ) from error
    if process.returncode:
        raise VividError(
            "media_processing_failed",
            f"{command[0]} could not process the selected material.",
            {"stderr": process.stderr[-4000:]},
        )
    return process


def _request_key(kind: str, asset: dict[str, Any], arguments: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"version": 1, "kind": kind, "source_sha256": asset["sha256"], **arguments},
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _range(
    asset: dict[str, Any], start: float, end: float | None, maximum: float, default: float
) -> tuple[float, float]:
    start = _number(start, "start", minimum=0)
    duration = asset["duration"]
    if asset["kind"] == "image":
        if start != 0 or end not in {None, 0, 1}:
            raise VividError("invalid_time_range", "A still image has only its frame at time zero.")
        return 0.0, 0.0
    if duration <= 0:
        raise VividError("unknown_duration", "The material has no reliable finite duration.")
    if start >= duration:
        raise VividError(
            "invalid_time_range",
            "Start must precede the end of the material.",
            {"duration": duration},
        )
    end = min(duration, start + default) if end is None else _number(end, "end", minimum=0)
    if end <= start:
        raise VividError(
            "invalid_time_range",
            "Use an ordered range inside the material.",
            {"duration": duration},
        )
    end = min(end, duration)
    if end - start > maximum + 1e-6:
        raise VividError(
            "inspection_too_long",
            "Narrow the window before inspecting it.",
            {"max_seconds": maximum},
        )
    return start, end


def _roi(
    value: list[int] | dict[str, int] | None, width: int, height: int
) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        try:
            value = [value["x"], value["y"], value["width"], value["height"]]
        except KeyError as error:
            raise VividError("invalid_roi", "ROI object needs x, y, width and height.") from error
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(type(v) is not int for v in value)
    ):
        raise VividError(
            "invalid_roi", "ROI must be [x,y,width,height] in native decoded pixel coordinates."
        )
    x, y, w, h = value
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise VividError(
            "invalid_roi",
            "ROI must fit inside the native decoded frame.",
            {"width": width, "height": height},
        )
    return x, y, w, h


def _contact_sheet(
    items: list[tuple[Path, str]], destination: Path, *, tile_width: int = 480
) -> None:
    columns = min(3, len(items))
    thumb_height = 300
    label_height = 44
    sheet = Image.new(
        "RGB",
        (columns * tile_width, math.ceil(len(items) / columns) * (thumb_height + label_height)),
        "#171a20",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(items):
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        image.thumbnail((tile_width - 12, thumb_height - 12), Image.Resampling.LANCZOS)
        x = index % columns * tile_width
        y = index // columns * (thumb_height + label_height)
        sheet.paste(
            image, (x + (tile_width - image.width) // 2, y + (thumb_height - image.height) // 2)
        )
        draw.text((x + 8, y + thumb_height + 3), label[:75], fill="white")
    sheet.save(destination)


class MediaTools:
    def __init__(self, store: MemoryStore):
        self.store = store

    def inspect_video(
        self,
        asset_id: str,
        start: float = 0,
        end: float | None = None,
        fps: float = 1,
        max_frames: int | None = None,
        roi: list[int] | dict[str, int] | None = None,
        width: int | None = 1280,
        mode: str = "window",
        sheet_size: int | None = None,
    ) -> dict[str, Any]:
        if mode not in {"window", "overview"}:
            raise VividError("invalid_argument", "mode must be window or overview")
        if max_frames is None:
            max_frames = 24 if mode == "overview" else 12
        if sheet_size is None:
            sheet_size = 8 if mode == "overview" else 24
        if type(sheet_size) is not int or not 1 <= sheet_size <= 24:
            raise VividError("invalid_argument", "sheet_size must be 1–24")
        if mode == "overview":
            return self._overview(asset_id, start, end, max_frames, roi, width, sheet_size)
        asset = self.store.asset(asset_id)
        streams = [
            stream
            for stream in asset["streams"]
            if stream.get("codec_type") == "video"
            and not stream.get("disposition", {}).get("attached_pic")
        ]
        if not streams:
            raise VividError("no_video", "The selected material has no video stream.")
        fps = _number(fps, "fps", minimum=0.01, maximum=60)
        if type(max_frames) is not int or not 1 <= max_frames <= MAX_FRAMES:
            raise VividError("invalid_argument", f"max_frames must be 1–{MAX_FRAMES}.")
        if width is not None and (type(width) is not int or not 64 <= width <= 8192):
            raise VividError(
                "invalid_argument", "Preview width must be 64–8192 or null for native resolution."
            )
        native_width, native_height = (
            int(streams[0].get("width", 0)),
            int(streams[0].get("height", 0)),
        )
        if (
            native_width <= 0
            or native_height <= 0
            or native_width * native_height > MAX_NATIVE_PIXELS
        ):
            raise VividError(
                "image_too_large",
                "Decoded frame dimensions are unsupported or exceed 40 megapixels.",
            )
        crop = _roi(roi, native_width, native_height)
        start, end = _range(asset, start, end, MAX_VIDEO_SECONDS, 30)
        effective_fps = min(fps, max_frames / (end - start)) if end > start else fps
        arguments = {
            "start": start,
            "end": end,
            "fps": fps,
            "max_frames": max_frames,
            "roi": crop,
            "width": width,
            "sheet_size": sheet_size,
        }
        key = _request_key("video", asset, arguments)
        cached = self.store.cached_evidence(asset["asset_id"], "video", key)
        if cached is not None:
            return cached
        if native_width * native_height * max_frames * 3 > 1_000_000_000:
            raise VividError(
                "inspection_too_large",
                "Reduce max_frames for this resolution to limit decoded storage.",
            )
        with tempfile.TemporaryDirectory(
            prefix="video-", dir=self.store.staging_directory()
        ) as folder:
            staging = Path(folder)
            destination = staging / "frame_%04d.png"
            base = float(asset.get("start_time", 0))
            if asset["kind"] == "image":
                filters = "showinfo"
                command = [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-loglevel",
                    "info",
                    "-noautorotate",
                    "-i",
                    asset["path"],
                    "-map",
                    f"0:{streams[0]['index']}",
                    "-vf",
                    filters,
                    "-frames:v",
                    "1",
                    "-pix_fmt",
                    "rgb24",
                    "-threads",
                    "1",
                    "-y",
                    str(destination),
                ]
            else:
                absolute_start, absolute_end = base + start, base + end
                select = f"gte(t,{absolute_start:.9f})*lt(t,{absolute_end:.9f})*(isnan(prev_selected_t)+gt(floor((t-{absolute_start:.9f})*{effective_fps:.9f}+0.0000001),floor((prev_selected_t-{absolute_start:.9f})*{effective_fps:.9f}+0.0000001)))"
                filters = "select=" + select.replace(",", "\\,") + ",showinfo"
                command = [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-loglevel",
                    "info",
                    "-ss",
                    f"{start:.9f}",
                    "-t",
                    f"{end - start:.9f}",
                    "-copyts",
                    "-noautorotate",
                    "-i",
                    asset["path"],
                    "-map",
                    f"0:{streams[0]['index']}",
                    "-an",
                    "-vf",
                    filters,
                    "-vsync",
                    "0",
                    "-frames:v",
                    str(max_frames),
                    "-pix_fmt",
                    "rgb24",
                    "-threads",
                    "1",
                    "-y",
                    str(destination),
                ]
            with stage("video_decode", mode=mode):
                process = _run(command, timeout=180)
            images = sorted(staging.glob("frame_*.png"))
            lines = re.findall(
                r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([-+\d.eE]+)", process.stderr
            )
            observations = [
                (int(index), int(pts), float(timestamp)) for index, pts, timestamp in lines
            ]
            if len(images) != len(observations) or not images:
                raise VividError(
                    "no_frames",
                    "No reliably timestamped frames were decoded. Choose a different interval.",
                    {
                        "decoded_images": len(images),
                        "timestamps": len(observations),
                        "stderr": process.stderr[-1500:],
                    },
                )
            frames = []
            for number, (raw_path, (_, pts, absolute_time)) in enumerate(zip(images, observations)):
                with Image.open(raw_path) as original:
                    preview = original.convert("RGB")
                    if crop:
                        x, y, w, h = crop
                        preview = preview.crop((x, y, x + w, y + h))
                    if width is not None and preview.width > width:
                        preview = preview.resize(
                            (width, max(1, round(preview.height * width / preview.width))),
                            Image.Resampling.LANCZOS,
                        )
                    preview_path = staging / f"preview_{number:04d}.png"
                    preview.save(preview_path)
                timestamp = max(0.0, absolute_time - base) if asset["kind"] != "image" else 0.0
                frame = self.store.register_evidence(
                    "video_frame",
                    asset["asset_id"],
                    {
                        "time_range": [timestamp, timestamp],
                        "timestamp": timestamp,
                        "pts": pts,
                        "pts_time": absolute_time,
                        "time_base": streams[0].get("time_base"),
                        "stream_index": streams[0]["index"],
                        "native_size": [native_width, native_height],
                        "pixel_format": "decoded_rgb8",
                        "source_pixel_format": streams[0].get("pix_fmt"),
                        "raw_path": str(raw_path),
                        "preview_path": str(preview_path),
                        "roi": list(crop) if crop else None,
                        "preview_size": list(preview.size),
                        "transform": {
                            "autorotate": False,
                            "crop": list(crop) if crop else None,
                            "preview_width_limit": width,
                        },
                        "images": [{"path": str(preview_path), "mime_type": "image/png"}],
                    },
                    [raw_path, preview_path],
                )
                frames.append(
                    {
                        name: frame[name]
                        for name in [
                            "evidence_id",
                            "timestamp",
                            "pts",
                            "pts_time",
                            "time_base",
                            "stream_index",
                            "native_size",
                            "pixel_format",
                            "raw_path",
                            "preview_path",
                            "roi",
                            "preview_size",
                        ]
                    }
                )
            files, groups = self._sheets(frames, staging, sheet_size)
            return self.store.register_evidence(
                "video",
                asset["asset_id"],
                {
                    "time_range": [start, end],
                    "requested_fps": fps,
                    "effective_fps_upper_bound": effective_fps,
                    "sampling_limited": effective_fps < fps,
                    "frame_count": len(frames),
                    "frames": frames,
                    "roi": list(crop) if crop else None,
                    "coverage": "sampled_frames_only",
                    "source_preserved": True,
                    "request_key": key,
                    "groups": groups,
                    "images": [{"path": str(p), "mime_type": "image/png"} for p in files],
                    "pixel_semantics": "Raw evidence PNGs preserve this decoded RGB8 frame, not original codec/YUV precision. The original source remains available. Rotation metadata is not applied.",
                },
                files,
            )

    @staticmethod
    def _sheets(frames, staging, sheet_size):
        files, groups = [], []
        for offset in range(0, len(frames), sheet_size):
            group = frames[offset : offset + sheet_size]
            sheet = staging / f"contact_sheet_{offset // sheet_size:02d}.png"
            if len(group) == 1:
                sheet = Path(group[0]["preview_path"])
            else:
                _contact_sheet(
                    [
                        (Path(f["preview_path"]), f"{offset + i + 1}: {f['timestamp']:.6f}s")
                        for i, f in enumerate(group)
                    ],
                    sheet,
                    tile_width=384,
                )
            files.append(sheet)
            groups.append(
                {
                    "image_index": len(files) - 1,
                    "frame_indices": list(range(offset, offset + len(group))),
                    "evidence_ids": [f["evidence_id"] for f in group],
                }
            )
        return files, groups

    def _overview(self, asset_id, start, end, max_frames, roi, width, sheet_size):
        """Seek independently to distributed samples; never decode the entire span."""
        asset = self.store.asset(asset_id)
        if asset["kind"] == "image":
            return self.inspect_video(
                asset_id, start=start, end=end, roi=roi, width=width, sheet_size=sheet_size
            )
        if type(max_frames) is not int or not 1 <= max_frames <= MAX_FRAMES:
            raise VividError("invalid_argument", "max_frames must be 1–24")
        start, end = _range(asset, start, end, float("inf"), asset["duration"])
        # Overview tiles intentionally use small previews. Individual native frames
        # remain retrievable by their evidence IDs, as in ordinary inspection.
        width = 640 if width == 1280 else width
        key = _request_key(
            "overview",
            asset,
            dict(
                start=start,
                end=end,
                max_frames=max_frames,
                roi=roi,
                width=width,
                sheet_size=sheet_size,
            ),
        )
        cached = self.store.cached_evidence(asset_id, "video", key)
        if cached is not None:
            return cached
        frames, missed = [], []
        for i in range(max_frames):
            check_budget()
            target = start + (i + 0.5) * (end - start) / max_frames
            try:
                result = self.inspect_video(
                    asset_id,
                    start=target,
                    end=min(end, target + 5),
                    max_frames=1,
                    roi=roi,
                    width=width,
                )
                frames.extend(result["frames"])
            except VividError as exc:
                if exc.code != "no_frames":
                    raise
                missed.append(target)
        if not frames:
            raise VividError("no_frames", "No overview samples decoded")
        with tempfile.TemporaryDirectory(dir=self.store.staging_directory()) as folder:
            files, groups = self._sheets(frames, Path(folder), sheet_size)
            return self.store.register_evidence(
                "video",
                asset_id,
                {
                    "mode": "overview",
                    "time_range": [start, end],
                    "frames": frames,
                    "frame_count": len(frames),
                    "groups": groups,
                    "missed_samples": missed,
                    "coverage": "sampled_frames_only",
                    "source_preserved": True,
                    "sampling": "independent_seeks_at_bin_centers",
                    "request_key": key,
                    "images": [{"path": str(p), "mime_type": "image/png"} for p in files],
                },
                files,
            )

    def inspect_audio(
        self,
        asset_id: str,
        start: float = 0,
        end: float | None = None,
        transcribe: bool = False,
        language: str | None = None,
        plots: bool = True,
        word_timestamps: bool = False,
    ) -> dict[str, Any]:
        asset = self.store.asset(asset_id)
        streams = [s for s in asset["streams"] if s.get("codec_type") == "audio"]
        if not streams:
            raise VividError("no_audio", "The selected material has no audio stream.")
        if type(transcribe) is not bool:
            raise VividError("invalid_argument", "transcribe must be true or false.")
        if type(word_timestamps) is not bool:
            raise VividError("invalid_argument", "word_timestamps must be true or false")
        if language is not None and (
            not isinstance(language, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z_-]{0,15}", language)
        ):
            raise VividError(
                "invalid_language", "Use a short Whisper language name/code, or null for detection."
            )
        start, end = _range(asset, start, end, MAX_AUDIO_SECONDS, 30)
        stream = streams[0]
        rate, channels = int(stream.get("sample_rate", 0)), int(stream.get("channels", 0))
        if rate <= 0 or not 1 <= channels <= 32:
            raise VividError(
                "unsupported_audio", "The audio sample rate or channel count is unsupported."
            )
        if (end - start) * rate * channels > MAX_AUDIO_VALUES:
            raise VividError(
                "inspection_too_large",
                "Narrow the audio window for this sample rate/channel count.",
                {"max_sample_values": MAX_AUDIO_VALUES},
            )
        asr_profile = self._asr_profile(word_timestamps) if transcribe else None
        key = _request_key(
            "audio",
            asset,
            {
                "start": start,
                "end": end,
                "transcribe": transcribe,
                "language": language,
                "whisper_model": os.getenv("VIVID_WHISPER_MODEL", "base"),
                "sample_timing_version": 2,
                "transcript_assembly_version": 2,
                "plots": plots,
                "asr_profile": asr_profile,
            },
        )
        cached = self.store.cached_evidence(asset["asset_id"], "audio", key)
        if cached is not None and (
            not transcribe
            or (
                asr_profile["weights"] is not None
                and self._asr_profile(word_timestamps) == asr_profile
                and cached.get("transcription", {}).get("available")
            )
        ):
            return cached
        base = float(asset.get("start_time", 0))
        try:
            audio_origin = float(stream.get("start_time", base))
        except (TypeError, ValueError):
            audio_origin = base
        actual_start = max(start, audio_origin - base)
        if end <= actual_start:
            raise VividError(
                "no_audio_in_window", "This interval precedes the start of the audio stream."
            )
        first_sample = max(0, round((base + actual_start - audio_origin) * rate))
        # PCM windows begin on sample boundaries, not arbitrary requested decimals.
        # Use that same origin for decoding, plots, transcripts and returned timing.
        actual_start = audio_origin - base + first_sample / rate
        frame_count = max(0, round((end - actual_start) * rate))
        with tempfile.TemporaryDirectory(
            prefix="audio-", dir=self.store.staging_directory()
        ) as folder:
            staging = Path(folder)
            wav_path = staging / "audio.wav"
            pcm_path = staging / "audio.pcm"
            _run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-v",
                    "error",
                    "-ss",
                    f"{actual_start:.9f}",
                    "-t",
                    f"{end - actual_start:.9f}",
                    "-i",
                    asset["path"],
                    "-map",
                    f"0:{stream['index']}",
                    "-vn",
                    "-af",
                    f"atrim=end_sample={frame_count},asetpts=PTS-STARTPTS",
                    "-c:a",
                    "pcm_s16le",
                    "-ar",
                    str(rate),
                    "-ac",
                    str(channels),
                    "-f",
                    "s16le",
                    "-threads",
                    "1",
                    "-y",
                    str(pcm_path),
                ],
                timeout=120,
                stage_name="audio_decode",
            )
            byte_count = pcm_path.stat().st_size
            if byte_count % (2 * channels) or byte_count > MAX_AUDIO_VALUES * 2:
                raise VividError("unsupported_audio", "Unexpected decoded audio format or length.")
            actual_rate, actual_channels = rate, channels
            nframes = byte_count // (2 * channels)
            raw = pcm_path.read_bytes()
            # A raw intermediate avoids FFmpeg's extensible WAV headers, which
            # Python 3.11 rejects even for PCM16 when there are >2 channels.
            with wave.open(str(wav_path), "wb") as recording:
                recording.setnchannels(channels)
                recording.setsampwidth(2)
                recording.setframerate(rate)
                recording.writeframes(raw)
            if nframes == 0:
                raise VividError(
                    "no_audio_in_window", "The requested interval decoded no audio samples."
                )
            samples = (
                np.frombuffer(raw, dtype="<i2").reshape(-1, actual_channels).astype(np.float32)
                / 32768.0
            )
            waveform_path = staging / "waveform.png"
            spectrogram_path = staging / "spectrogram.png"
            if plots:
                self._waveform(samples, actual_rate, actual_start, waveform_path)
                self._spectrogram(samples, actual_rate, actual_start, spectrogram_path)
            stats = {
                "peak": float(np.max(np.abs(samples))),
                "rms": float(np.sqrt(np.mean(np.square(samples), dtype=np.float64))),
                "dc_offset": [float(value) for value in samples.mean(axis=0)],
                "channel_rms": [
                    float(value)
                    for value in np.sqrt(np.mean(np.square(samples), axis=0, dtype=np.float64))
                ],
                "near_full_scale_fraction": float(np.mean(np.abs(samples) >= 32760 / 32768)),
                "near_zero_fraction": float(np.mean(np.abs(samples) < 1e-4)),
            }
            transcript = (
                self._cached_transcript(
                    asset,
                    wav_path,
                    actual_start,
                    actual_start + nframes / actual_rate,
                    language,
                    word_timestamps,
                )
                if transcribe
                else {"requested": False, "available": False}
            )
            return self.store.register_evidence(
                "audio",
                asset["asset_id"],
                {
                    "time_range": [actual_start, actual_start + nframes / actual_rate],
                    "requested_range": [start, end],
                    "stream_index": stream["index"],
                    "sample_rate": actual_rate,
                    "channels": actual_channels,
                    "sample_range": [first_sample, first_sample + nframes],
                    "sample_count": nframes,
                    "audio_path": str(wav_path),
                    "audio_mime_type": "audio/wav",
                    "statistics": stats,
                    "transcription": transcript,
                    "planner_input": "analysis_and_transcript"
                    if transcript.get("available")
                    else "analysis_only",
                    "source_preserved": True,
                    "request_key": key,
                    "images": [
                        {"path": str(waveform_path), "mime_type": "image/png"},
                        {"path": str(spectrogram_path), "mime_type": "image/png"},
                    ]
                    if plots
                    else [],
                    "audio_semantics": "This is a PCM16 inspection derivative at the source sample rate/channel count. Original compressed/native-precision audio remains in the source. Waveforms, spectra and ASR do not establish that the planner heard the audio.",
                },
                [wav_path, waveform_path, spectrogram_path] if plots else [wav_path],
            )

    @staticmethod
    def _asr_profile(word_timestamps=False):
        from vivid.transcript_cache import profile

        return profile(word_timestamps)

    def _cached_transcript(self, asset, path, start, end, language, word_timestamps=False):
        from vivid.transcript_cache import TranscriptCache

        def load(lo, hi):
            if abs(lo - start) < 1e-6 and abs(hi - end) < 1e-6:
                if word_timestamps:
                    return self._transcribe(path, lo, language, word_timestamps=True)
                return self._transcribe(path, lo, language)
            with wave.open(str(path), "rb") as recording:
                rate = recording.getframerate()
                first = round((lo - start) * rate)
                recording.setpos(first)
                raw = recording.readframes(round((hi - lo) * rate))
                with tempfile.NamedTemporaryFile(suffix=".wav", dir=path.parent) as temp:
                    with wave.open(temp.name, "wb") as clip:
                        clip.setparams(recording.getparams())
                        clip.writeframes(raw)
                    actual_lo = start + first / rate
                    if word_timestamps:
                        return self._transcribe(
                            Path(temp.name), actual_lo, language, word_timestamps=True
                        )
                    return self._transcribe(Path(temp.name), actual_lo, language)

        return TranscriptCache(self.store, asset, language, word_timestamps).resolve(
            start, end, load
        )

    @staticmethod
    def _waveform(samples: np.ndarray, rate: int, start: float, path: Path) -> None:
        width, channel_height, shown = 1280, 150, min(samples.shape[1], 8)
        image = Image.new("RGB", (width, 54 + shown * channel_height), "#151923")
        draw = ImageDraw.Draw(image)
        draw.text(
            (16, 12),
            f"Waveform | {start:.6f} to {start + len(samples) / rate:.6f}s | {rate} Hz | channel min/max envelope",
            fill="white",
        )
        if samples.shape[1] > shown:
            draw.text(
                (16, 30),
                f"Showing {shown}/{samples.shape[1]} channels; all channels retained in WAV and numerical statistics.",
                fill="#ffcc66",
            )
        bins = np.linspace(0, len(samples), width - 64 + 1).astype(int)
        for channel in range(shown):
            center = 54 + channel * channel_height + channel_height // 2
            draw.line((48, center, width - 16, center), fill="#485269")
            draw.text((8, center - 6), str(channel + 1), fill="white")
            for x in range(len(bins) - 1):
                block = samples[bins[x] : max(bins[x] + 1, bins[x + 1]), channel]
                if block.size:
                    high, low = float(block.max()), float(block.min())
                    draw.line(
                        (48 + x, center - high * 60, 48 + x, center - low * 60), fill="#65d4ec"
                    )
        image.save(path)

    @staticmethod
    def _spectrogram(samples: np.ndarray, rate: int, start: float, path: Path) -> None:
        nfft = min(2048, max(64, 2 ** int(math.floor(math.log2(max(len(samples), 64))))))
        bins = min(1024, max(1, math.ceil(len(samples) / (nfft // 4))))
        padded = np.pad(samples, ((nfft // 2, nfft // 2), (0, 0)))
        centers = np.linspace(0, max(len(samples) - 1, 0), bins).astype(int)
        indices = centers[:, None] + np.arange(nfft)[None, :]
        window = np.hanning(nfft).astype(np.float32)
        power = np.zeros((bins, nfft // 2 + 1), dtype=np.float64)
        for channel in range(samples.shape[1]):
            values = padded[indices, channel] * window
            power += np.abs(np.fft.rfft(values, axis=1)) ** 2
        power /= samples.shape[1]
        db = 10 * np.log10(np.maximum(power, 1e-12))
        db -= max(float(db.max()), -120)
        intensity = np.clip((db + 90) / 90, 0, 1).T[::-1]
        rgb = np.stack(
            (
                np.clip(2.5 * intensity - 0.7, 0, 1),
                np.clip(2.5 * intensity - 1.2, 0, 1),
                np.clip(2.5 * intensity, 0, 1) * (1 - intensity),
            ),
            axis=-1,
        )
        chart = Image.fromarray((rgb * 255).astype(np.uint8)).resize(
            (1200, 460), Image.Resampling.BILINEAR
        )
        image = Image.new("RGB", (1280, 530), "#151923")
        image.paste(chart, (60, 40))
        draw = ImageDraw.Draw(image)
        draw.text(
            (12, 12),
            f"Spectrogram | mean channel power | linear frequency 0–{rate / 2000:g} kHz | relative dB [-90, 0]",
            fill="white",
        )
        draw.text((8, 40), f"{rate / 2000:g}k", fill="white")
        draw.text((24, 485), "0", fill="white")
        draw.text((60, 510), f"{start:.6f}s", fill="white")
        draw.text((1140, 510), f"{start + len(samples) / rate:.6f}s", fill="white")
        image.save(path)

    @staticmethod
    def _transcribe(
        path: Path, start: float, language: str | None, word_timestamps=False
    ) -> dict[str, Any]:
        from vivid.transcription import transcribe

        return transcribe(path, start, language, word_timestamps=word_timestamps)

    def read_pixels(
        self,
        evidence_id: str,
        roi: list[int] | dict[str, int] | None = None,
        max_samples: int = 4096,
    ) -> dict[str, Any]:
        evidence = self.store.evidence(evidence_id)
        if evidence["kind"] == "video" and evidence.get("frame_count") == 1:
            evidence = self.store.evidence(evidence["frames"][0]["evidence_id"])
        if evidence["kind"] != "video_frame" or "raw_path" not in evidence:
            choices = [frame["evidence_id"] for frame in evidence.get("frames", [])]
            raise VividError(
                "ambiguous_frame",
                "Use a single video_frame evidence ID to read native decoded pixels.",
                {"frame_evidence_ids": choices},
            )
        if type(max_samples) is not int or not 1 <= max_samples <= 65536:
            raise VividError("invalid_argument", "max_samples must be 1–65536.")
        path = self.store._artifact_path(Path(evidence["raw_path"]))
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        crop = _roi(roi, image.width, image.height)
        x, y, w, h = crop or (0, 0, image.width, image.height)
        if w * h > max_samples:
            raise VividError(
                "too_many_pixels",
                "Request a smaller native pixel ROI.",
                {"requested_pixels": w * h, "max_samples": max_samples},
            )
        values = np.asarray(image.crop((x, y, x + w, y + h))).tolist()
        self.store.record_event(
            "pixels_read", {"evidence_id": evidence["evidence_id"], "roi": [x, y, w, h]}
        )
        return {
            "evidence_id": evidence["evidence_id"],
            "asset_id": evidence["asset_id"],
            "timestamp": evidence["timestamp"],
            "roi": [x, y, w, h],
            "pixel_format": "decoded_rgb8",
            "values": values,
            "semantics": "Exact values of the archived RGB8 decode, not original codec YUV samples.",
        }

    def compare(self, evidence_ids: list[str]) -> dict[str, Any]:
        if (
            not isinstance(evidence_ids, list)
            or not 1 <= len(evidence_ids) <= 16
            or any(not isinstance(value, str) for value in evidence_ids)
        ):
            raise VividError("invalid_argument", "Compare 1–16 evidence IDs.")
        evidence = [self.store.evidence(value) for value in evidence_ids]
        items = []
        for item in evidence:
            path = item.get("preview_path") or next(
                (
                    image["path"]
                    for image in item.get("images", [])
                    if image.get("mime_type", "").startswith("image/")
                ),
                None,
            )
            if path is None:
                raise VividError(
                    "no_visual_evidence", f"Evidence {item['evidence_id']} has no image preview."
                )
            items.append(
                (
                    self.store._artifact_path(Path(path)),
                    f"{item['evidence_id']} | {item.get('time_range')}",
                )
            )
        with tempfile.TemporaryDirectory(
            prefix="compare-", dir=self.store.staging_directory()
        ) as folder:
            destination = Path(folder) / "comparison.png"
            _contact_sheet(items, destination)
            return self.store.register_evidence(
                "comparison",
                evidence[0]["asset_id"],
                {
                    "time_range": evidence[0]["time_range"],
                    "derived_from": evidence_ids,
                    "description": "Side-by-side visual comparison. Each label identifies its own source evidence and time.",
                    "images": [{"path": str(destination), "mime_type": "image/png"}],
                },
                [destination],
            )
