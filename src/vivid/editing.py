"""Optional deterministic edit compilation over the native job runner."""

from __future__ import annotations

import hashlib
import json

from vivid.errors import VividError
from vivid.native_jobs import NativeJobs
from vivid.temporal import compose_segments, map_captions
from vivid.workspace import atomic_json, atomic_text


def _tempo(speed):
    filters = []
    while speed > 2:
        filters.append("atempo=2")
        speed /= 2
    while speed < 0.5:
        filters.append("atempo=0.5")
        speed *= 2
    return ",".join([*filters, f"atempo={speed:.12g}"])


def _srt_time(seconds):
    ms = round(seconds * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def edit(service, args):
    args = dict(args)
    action = args.pop("action")
    jobs = NativeJobs(service.workspace)
    if action in {"status", "cancel"}:
        if set(args) != {"job_id"}:
            raise VividError("invalid_arguments", "status/cancel requires only job_id")
        return getattr(jobs, action)(args["job_id"])
    if action == "start":
        return jobs.start(**args)
    segments = args.pop("segments", None)
    timeline = compose_segments(segments)
    captions = map_captions(timeline, args.pop("captions", []))
    if action == "timing":
        if args:
            raise VividError("invalid_arguments", "timing accepts segments and captions only")
        return {"ok": True, "timeline": timeline, "captions": captions}
    if "output" not in args:
        raise VividError("invalid_arguments", "render requires output")
    if any(c["requires_review"] for c in captions):
        return {
            "ok": False,
            "error": {
                "code": "caption_boundary",
                "message": "Caption text straddles a cut. Supply word timing or revised cues before burning it.",
            },
            "timeline": timeline,
            "captions": captions,
        }
    assets = [service.register_path(s["source"]) for s in segments]
    video = [
        next((v for v in a["streams"] if v.get("codec_type") == "video"), None) for a in assets
    ]
    if any(v is None for v in video):
        raise VividError(
            "no_video", "render segments must contain video; native commands support other edits"
        )
    for segment, asset in zip(segments, assets):
        if segment["source_range"][1] > asset["duration"] + 1e-6:
            raise VividError("invalid_time_range", "An edit range exceeds its source")
        if not 0.01 <= segment.get("speed", 1) <= 100:
            raise VividError("invalid_time", "Compiled render supports speed 0.01–100")
    width = args.pop("width", video[0]["width"])
    height = args.pop("height", video[0]["height"])
    if width % 2 or height % 2:
        raise VividError("invalid_dimensions", "H.264 yuv420p needs even dimensions")
    fps = video[0].get("avg_frame_rate", "30/1")
    if fps in {None, "0/0", "0/1"}:
        fps = "30/1"
    audio = args.pop(
        "audio", any(any(s.get("codec_type") == "audio" for s in a["streams"]) for a in assets)
    )
    command = ["ffmpeg", "-v", "error"]
    for segment, a in zip(segments, assets):
        start, end = segment["source_range"]
        command.extend(["-ss", str(start), "-t", str(end - start), "-i", a["path"]])
    filters, nodes = [], []
    for i, (segment, asset) in enumerate(zip(segments, assets)):
        start, end = segment["source_range"]
        speed = segment.get("speed", 1)
        duration = (end - start) / speed
        filters.append(
            f"[{i}:v:0]trim=duration={end - start},setpts=(PTS-STARTPTS)/{speed},"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"tpad=stop_mode=clone:stop_duration={duration},fps={fps}:round=up,trim=duration={duration},"
            f"setpts=PTS-STARTPTS,format=yuv420p[v{i}]"
        )
        nodes.append(f"[v{i}]")
        if audio:
            if any(s.get("codec_type") == "audio" for s in asset["streams"]):
                filters.append(
                    f"[{i}:a:0]atrim=duration={end - start},asetpts=PTS-STARTPTS,"
                    f"{_tempo(speed)},aresample=48000,aformat=channel_layouts=stereo,"
                    f"apad,atrim=duration={duration}[a{i}]"
                )
            else:
                filters.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={duration}[a{i}]")
            nodes.append(f"[a{i}]")
    filters.append(
        "".join(nodes)
        + f"concat=n={len(segments)}:v=1:a={int(audio)}[video]"
        + ("[audio]" if audio else "")
    )
    digest = hashlib.sha256(
        json.dumps(
            [timeline, captions, width, height, audio, fps, [a["sha256"] for a in assets]],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    root = service.workspace.state / "edit-specs" / digest
    root.mkdir(parents=True, exist_ok=True)
    dependencies = [a["path"] for a in assets]
    if captions:
        srt = root / "captions.srt"
        atomic_text(
            srt,
            "\n\n".join(
                f"{i + 1}\n{_srt_time(c['start'])} --> {_srt_time(c['end'])}\n{c['text']}"
                for i, c in enumerate(captions)
            )
            + "\n",
        )
        escaped = str(srt).replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''")
        filters.append(f"[video]subtitles=filename='{escaped}'[captioned]")
        dependencies.append(str(srt))
    graph = root / "filters.txt"
    atomic_text(graph, ";\n".join(filters))
    command.extend(
        ["-filter_complex_script", str(graph), "-map", "[captioned]" if captions else "[video]"]
    )
    if audio:
        command.extend(["-map", "[audio]", "-c:a", "aac", "-b:a", "160k"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "{output}",
        ]
    )
    dependencies.append(str(graph))
    atomic_json(root / "mapping.json", {"timeline": timeline, "captions": captions})
    expected = {
        "expected_duration": timeline["duration"],
        "expected_width": width,
        "expected_height": height,
        "expected_video_codec": "h264",
        "expected_audio": audio,
        **args.pop("expected", {}),
    }
    result = jobs.start(command, args.pop("output"), dependencies, expected=expected, **args)
    result.update(timeline=timeline, captions=captions, mapping_path=str(root / "mapping.json"))
    return result
