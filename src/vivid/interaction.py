"""Small media interactions; native planning and execution remain the agent's choice."""

from __future__ import annotations

import tempfile
from pathlib import Path

from vivid.errors import VividError


def route(service, name, args):
    if name == "retrieve":
        if "image_offset" in args:
            if set(args) != {"evidence_id", "image_offset"}:
                raise VividError("invalid_arguments", "image_offset requires only evidence_id")
            result = service._call("recall", {"evidence_id": args["evidence_id"]})
            if args["image_offset"] >= len(result.get("images", [])):
                raise VividError("invalid_arguments", "image_offset is outside this evidence")
            result["image_offset"] = args["image_offset"]
            return result
        if "evidence_id" in args or (
            "query" not in args and set(args) <= {"path", "asset_id", "limit"}
        ):
            return service._call("recall", args)
        return service._call("search", args)
    if name == "edit":
        from vivid.editing import edit

        return edit(service, args)
    if name == "transcribe":
        target = {k: args[k] for k in ("path", "asset_id") if k in args}
        asset = service._input(target)
        start, end = (
            args.get("start", 0),
            args.get("end", min(asset["duration"], args.get("start", 0) + 30)),
        )
        if end <= start or start >= asset["duration"]:
            raise VividError("invalid_time_range", "Use an ordered source interval")
        context = args.get("context_seconds", 0)
        actual = {
            "asset_id": asset["asset_id"],
            "start": max(0, start - context),
            "end": min(asset["duration"], end + context),
            "transcribe": True,
            "word_timestamps": args.get("precision", "segment") == "word",
            "plots": False,
        }
        if "language" in args:
            actual["language"] = args["language"]
        result = service._call("inspect_audio", actual)
        transcript = result["transcription"]
        available = transcript.get("available", False)
        complete = available and result.get("complete", True)
        pending = transcript.get("pending_ranges", []) + result.get("pending_ranges", [])
        return {
            **transcript,
            "ok": available or transcript.get("partial", False),
            "status": "completed"
            if complete
            else ("partial" if available or transcript.get("partial") else "unavailable"),
            "complete": complete,
            "pending_ranges": pending,
            "evidence_id": result["evidence_id"],
            "version_id": result["version_id"],
            "requested_range": [start, end],
            "actual_range": result["time_range"],
            "error": None
            if available
            else {
                "code": transcript.get("error_kind", "asr_unavailable"),
                "message": "ASR incomplete; completed chunks are retained. Retry pending ranges only.",
            },
        }
    if "evidence_id" in args:
        if set(args) != {"evidence_id"}:
            raise VividError("invalid_arguments", "Use evidence_id alone for exact recall")
        return service._call("recall", args)
    if "reference" in args:
        from vivid.review import compare_frames

        reference = args.pop("reference")
        if "path" not in args or set(args) - {"path", "start", "roi"}:
            raise VividError(
                "invalid_arguments", "Comparison uses path/start/reference and optional roi"
            )
        return compare_frames(
            service,
            reference["path"],
            args["path"],
            reference["time"],
            args.get("start", 0),
            args.get("roi"),
        )
    paths = args.pop("paths", None)
    if paths is not None:
        if "path" in args or "asset_id" in args:
            raise VividError("invalid_arguments", "Use paths or one path/asset_id")
        items = []
        for path in paths:
            try:
                options = {**args, "path": path}
                if len(paths) > 3 and "max_frames" not in options:
                    options["max_frames"] = 3
                items.append(route(service, "inspect", options))
            except VividError as exc:
                items.append({"path": path, **exc.as_dict()})
        result = {
            "ok": all(i.get("ok", True) for i in items),
            "items": items,
            "images": [image for i in items for image in i.get("images", [])],
            "image_sources": [
                {
                    "path": item.get("origin_path", paths[index]),
                    "evidence_id": item.get("evidence_id"),
                    "image_count": len(item.get("images", [])),
                }
                for index, item in enumerate(items)
            ],
            "coverage": "sampled evidence only; each item has its own source clock",
        }
        frames = [(frame, item) for item in items for frame in item.get("frames", [])]
        if len(result["images"]) > 3 and frames:
            from vivid.media import _contact_sheet

            with tempfile.TemporaryDirectory(dir=service.memory.staging_directory()) as folder:
                files, groups = [], []
                for offset in range(0, len(frames), 12):
                    group = frames[offset : offset + 12]
                    sheet = Path(folder) / f"comparison_{offset // 12}.png"
                    _contact_sheet(
                        [
                            (
                                Path(f["preview_path"]),
                                f"{Path(item['origin_path']).name} | {f['timestamp']:.3f}s",
                            )
                            for f, item in group
                        ],
                        sheet,
                        tile_width=336,
                    )
                    files.append(sheet)
                    groups.append(
                        {
                            "image_index": len(files) - 1,
                            "samples": [
                                {
                                    "path": item["origin_path"],
                                    "timestamp": f["timestamp"],
                                    "evidence_id": f["evidence_id"],
                                }
                                for f, item in group
                            ],
                        }
                    )
                saved = service.memory.register_evidence(
                    "comparison",
                    frames[0][1]["asset_id"],
                    {
                        "time_range": [0, 0],
                        "image_groups": groups,
                        "source_versions": sorted({item["asset_id"] for _, item in frames}),
                        "coverage": "sampled frames from separate source clocks; not continuous coverage",
                        "images": [{"path": str(p), "mime_type": "image/png"} for p in files],
                    },
                    files,
                )
                result.update(
                    images=saved["images"], image_groups=groups, evidence_id=saved["evidence_id"]
                )
                result.pop("image_sources", None)
        return result
    asset = service._input(args)
    args.pop("path", None)
    args.pop("asset_id", None)
    mode = args.pop("mode", "auto")
    if mode == "audio" or (mode == "auto" and asset["kind"] == "audio"):
        if set(args) - {"start", "end", "plots"}:
            raise VividError(
                "invalid_arguments",
                "Audio observation accepts start/end/plots; use transcribe for ASR",
            )
        args.setdefault("plots", False)
        return service._call("inspect_audio", {"asset_id": asset["asset_id"], **args})
    args.pop("plots", None)
    span = min(args.get("end", asset["duration"]), asset["duration"]) - args.get("start", 0)
    if mode == "auto":
        mode = "overview" if span > 60 else "window"
        if mode == "window" and "end" not in args:
            args["end"] = asset["duration"] if asset["kind"] != "image" else 1
    args.setdefault("max_frames", 3 if span <= 15 else (6 if span <= 60 else 24))
    args["mode"] = mode
    return service._call("inspect_video", {"asset_id": asset["asset_id"], **args})
