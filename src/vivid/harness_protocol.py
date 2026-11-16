"""The six public tools of the native-editing evidence harness."""

from __future__ import annotations


def obj(properties, required=()):
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


PATH = {"type": "string", "minLength": 1, "maxLength": 4096}
TEXT = {"type": "string", "maxLength": 4096}
TIME = {"type": "number", "minimum": 0}
BOOL = {"type": "boolean"}
LIMIT = {"type": "integer", "minimum": 1, "maximum": 100}
INPUT = {"path": PATH, "asset_id": PATH}
EXPECTED = {
    "expected_width": {"type": "integer", "minimum": 1},
    "expected_height": {"type": "integer", "minimum": 1},
    "expected_duration": TIME,
    "duration_tolerance": {"type": "number", "minimum": 0, "maximum": 60},
    "expected_audio": BOOL,
    "expected_video_codec": PATH,
    "expected_audio_codec": PATH,
    "expected_sample_rate": {"type": "integer", "minimum": 1},
    "expected_channels": {"type": "integer", "minimum": 1},
    "expected_sha256": {"type": "string", "pattern": "^[a-fA-F0-9]{64}$"},
}
SPECS = [
    (
        "catalog",
        "List current workspace media and version identities. A path registers any native "
        "output or intermediate automatically; old versions remain available through recall.",
        obj(
            {
                "path": PATH,
                "refresh": BOOL,
                "limit": LIMIT,
                "offset": {"type": "integer", "minimum": 0},
            }
        ),
    ),
    (
        "inspect_video",
        "Inspect timestamped frames. mode=overview seeks across the entire source (or range), "
        "up to 24 frames grouped into sheets of 8; use this first for long sources. "
        "Default window mode covers at most 600s (default 30s). Sampling is not full coverage.",
        obj(
            {
                **INPUT,
                "mode": {"type": "string", "enum": ["window", "overview"]},
                "sheet_size": {"type": "integer", "minimum": 1, "maximum": 24},
                "start": TIME,
                "end": TIME,
                "fps": {"type": "number", "minimum": 0.01, "maximum": 60},
                "max_frames": {"type": "integer", "minimum": 1, "maximum": 24},
                "roi": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "width": {"type": ["integer", "null"], "minimum": 64, "maximum": 8192},
            }
        ),
    ),
    (
        "inspect_audio",
        "Inspect audio statistics and optional waveform/spectrum images from any "
        "workspace path, at most 120s (default 30s). Optional local transcription is derived text, "
        "not direct listening; plots=false avoids images.",
        obj(
            {
                **INPUT,
                "start": TIME,
                "end": TIME,
                "transcribe": BOOL,
                "word_timestamps": BOOL,
                "language": TEXT,
                "plots": BOOL,
            }
        ),
    ),
    (
        "search",
        "Literal text search of cached evidence, transcripts, and GUIDE/WORKING notes. "
        "This does not search visual content in raw media; a miss does not establish absence.",
        obj({"query": TEXT, **INPUT, "kind": TEXT, "start": TIME, "end": TIME, "limit": LIMIT}),
    ),
    (
        "recall",
        "Retrieve cached evidence with images, an archived media version, or bounded tool "
        "history. No IDs returns recent tool history. Use path for version history.",
        obj({"evidence_id": PATH, "asset_id": PATH, "path": PATH, "limit": LIMIT}),
    ),
    (
        "verify",
        "Check a concrete file and its exact hash, media streams and bounded full decode. "
        "Optional flat expected_* fields check concrete requirements only; no semantic approval gates.",
        obj({"path": PATH, **EXPECTED}, ("path",)),
    ),
]
EXAMPLES = {
    "catalog": {},
    "inspect_video": {"path": "materials/clip.mp4", "max_frames": 6},
    "inspect_audio": {"path": "materials/clip.mp4", "plots": False},
    "search": {"query": "dialogue", "limit": 10},
    "recall": {"evidence_id": "ev_example"},
    "verify": {"path": "output/final.mp4"},
}

# Keep the catalog flat while making arguments understandable without another call.
DESCRIPTIONS = {
    "mode": "window for local inspection; overview for distributed independent seeks over the full source or requested range. fps applies only to window mode.",
    "sheet_size": "Frames per timestamped contact sheet, default 8 for overview (24 frames produce 3 images), 24 for local windows. Individual frame evidence remains available through recall.",
    "path": "Existing file path, absolute or relative to the workspace root; accepts inputs, work intermediates, and output. Must stay inside the workspace.",
    "asset_id": "Exact immutable version ID from catalog/recall. Provide exactly one of path or asset_id for inspection/search filtering.",
    "start": "Inclusive requested start in seconds relative to the media origin; default 0.",
    "end": "Exclusive end in seconds. For inspections, defaults to 30s from start or media end; ends beyond the duration are clipped with a warning, and long requests return a bounded window and version-pinned next_request. For search, filters indexed evidence only.",
    "fps": "Requested sampling rate in frames per second; max_frames may reduce the actual rate. Returned frames retain actual decoded timestamps and PTS.",
    "max_frames": "Maximum sampled frames to decode, 1–24; default 12 in window mode or 24 in overview. This is sampled evidence, not continuous coverage.",
    "width": "Preview width limit in pixels, default 1280; null keeps native resolution. Native decoded frames remain archived.",
    "roi": "Crop [x,y,width,height] in native decoded source pixel coordinates before preview scaling; all four values must fit the frame.",
    "transcribe": "Request optional local Whisper transcription; default false. Transcripts do not establish that audio was heard.",
    "word_timestamps": "With transcribe=true, request optional word-level ASR timing for local cut/subtitle work. Estimates, not guaranteed forced alignment; default false.",
    "language": "Optional short Whisper language code/name such as en; omitted means detection.",
    "plots": "Return waveform and spectrum images; default true. False returns statistics/transcripts without images.",
    "query": "Case-insensitive literal terms that must all occur in indexed text; empty query lists bounded recent evidence and nonempty notes.",
    "kind": "Filter cached evidence by its stored kind, such as video, video_frame, or audio; excludes note results.",
    "limit": "Maximum returned items, 1–100; defaults to 20 (catalog 100). Totals and truncation indicate further records.",
    "offset": "Number of catalog entries to skip for bounded pagination; default 0.",
    "refresh": "Rescan workspace media and snapshot changed versions; default true. False returns the last catalogued versions.",
    "evidence_id": "Exact cached evidence ID; returns its archived images and source version metadata. Mutually exclusive with path/asset_id.",
    "expected_width": "Required decoded video/image width in pixels.",
    "expected_height": "Required decoded video/image height in pixels.",
    "expected_duration": "Required media duration in seconds, compared using duration_tolerance.",
    "duration_tolerance": "Allowed absolute duration difference in seconds; default 0.1.",
    "expected_audio": "Require an audio stream when true, or require none when false.",
    "expected_video_codec": "Required ffprobe video codec name, such as h264 or png.",
    "expected_audio_codec": "Required ffprobe audio codec name, such as aac or pcm_s16le.",
    "expected_sample_rate": "Required audio sample rate in Hz.",
    "expected_channels": "Required audio channel count.",
    "expected_sha256": "Required exact SHA-256 file identity, as 64 hexadecimal characters.",
}
for _name, _description, _schema in SPECS:
    _schema["properties"] = {
        key: {**value, "description": DESCRIPTIONS[key]}
        for key, value in _schema["properties"].items()
    }

# Legacy calls remain readable for archived scripts; only the following six are advertised.
LEGACY_SPECS = SPECS
VIDEO = dict(next(s for n, _, s in LEGACY_SPECS if n == "inspect_video")["properties"])
VIDEO["mode"] = {"type": "string", "enum": ["auto", "window", "overview", "audio"]}
VIDEO["plots"] = BOOL
VIDEO["evidence_id"] = PATH
VIDEO["paths"] = {"type": "array", "items": PATH, "minItems": 1, "maxItems": 15}
VIDEO["reference"] = obj({"path": PATH, "time": TIME}, ("path", "time"))
SEGMENT = obj(
    {
        "source": PATH,
        "source_range": {"type": "array", "items": TIME, "minItems": 2, "maxItems": 2},
        "speed": {"type": "number", "exclusiveMinimum": 0},
    },
    ("source", "source_range"),
)
SPECS = [
    LEGACY_SPECS[0],
    (
        "inspect",
        "Observe visual samples, audio statistics or declared before/after pairs. "
        "Accepts paths for batch comparison; auto uses few frames for short clips and grouped overview for long media. "
        "evidence_id recalls exact archived observations. Sampling is not continuous coverage.",
        obj(VIDEO),
    ),
    (
        "transcribe",
        "Optional local ASR. precision=segment for text lookup; word for short cut/subtitle windows. "
        "Returns source timestamps, completed and pending ranges, including partial results on failure. "
        "Word timestamps are estimates, not forced alignment. Native ASR remains usable.",
        obj(
            {
                **INPUT,
                "start": TIME,
                "end": TIME,
                "language": TEXT,
                "precision": {"type": "string", "enum": ["segment", "word"]},
                "context_seconds": {"type": "number", "minimum": 0, "maximum": 10},
            }
        ),
    ),
    (
        "retrieve",
        "Search indexed text or recall exact evidence/version/history. "
        "A miss does not establish absence in raw media. Evidence IDs take precedence over text search. "
        "Use image_offset with evidence_id to retrieve another page of archived images.",
        obj(
            {
                **INPUT,
                "evidence_id": PATH,
                "image_offset": {"type": "integer", "minimum": 0},
                "query": TEXT,
                "kind": TEXT,
                "start": TIME,
                "end": TIME,
                "limit": LIMIT,
            }
        ),
    ),
    (
        "edit",
        "Optional deterministic trim/concat/speed/caption helper and recoverable job runner. "
        "Does not approve plans or replace native editing. render compiles declared segments; timing only maps them. "
        "start accepts native argv with {output}; status/cancel use job_id.",
        obj(
            {
                "action": {
                    "type": "string",
                    "enum": ["render", "timing", "start", "status", "cancel"],
                },
                "segments": {"type": "array", "items": SEGMENT, "minItems": 1, "maxItems": 100},
                "captions": {"type": "array", "items": {"type": "object"}, "maxItems": 10000},
                "output": PATH,
                "command": {"type": "array", "items": PATH, "minItems": 1, "maxItems": 4096},
                "inputs": {"type": "array", "items": PATH, "minItems": 1, "maxItems": 256},
                "job_id": PATH,
                "estimated_seconds": {"type": "number", "exclusiveMinimum": 0},
                "retry": BOOL,
                "expected": obj(EXPECTED),
                "audio": BOOL,
                "width": {"type": "integer", "minimum": 2, "maximum": 8192},
                "height": {"type": "integer", "minimum": 2, "maximum": 8192},
            },
            ("action",),
        ),
    ),
    (
        "verify",
        "Verify exact output bytes, concrete media requirements and full decode (reused by hash). "
        "Optional timeline/captions/requirements plans targeted output review windows; "
        "missing or invalid review information is reported without discarding technical checks. "
        "Review windows do not establish semantic success.",
        obj(
            {
                "path": PATH,
                **EXPECTED,
                "timeline": {
                    "type": "object",
                    "description": "Optional declared edit timeline returned by edit timing/render. "
                    "Required only to plan edit-boundary windows.",
                },
                "captions": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Output-time cues with start/end seconds and text; unplannable cues are reported.",
                },
                "requirements": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional output-time review requests. Include start/end seconds to plan a window; "
                    "other requests remain unplanned and are not verified.",
                },
            },
            ("path",),
        ),
    ),
]
EXAMPLES.update(
    inspect={"path": "materials/clip.mp4"},
    transcribe={"path": "materials/clip.mp4", "start": 30, "end": 40, "precision": "segment"},
    retrieve={"query": "door"},
    edit={"action": "status", "job_id": "native_..."},
    verify={"path": "output/final.mp4"},
)
