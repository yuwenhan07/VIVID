"""Persistent, source-addressed evidence with an explicitly lossy search index."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from .errors import VividError
from .runtime import check_budget, run_process
from .workspace import managed_path

MEDIA_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".ts",
    ".mts",
    ".wav",
    ".mp3",
    ".flac",
    ".aac",
    ".m4a",
    ".ogg",
    ".opus",
    ".aiff",
    ".aif",
    ".wma",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
}
DOCUMENT_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".srt", ".vtt", ".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}


def sha256_file(path: Path) -> str:
    check_budget()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            check_budget()
            digest.update(block)
    check_budget()
    return digest.hexdigest()


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise VividError("invalid_json", "Evidence must contain finite JSON values.") from error


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, encoding="utf-8", delete=False
    ) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def probe_media(path: Path) -> dict[str, Any]:
    try:
        process = run_process(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=45,
        )
    except FileNotFoundError as error:
        raise VividError(
            "missing_dependency", "ffprobe is required. Install FFmpeg in the task environment."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise VividError("probe_timeout", f"ffprobe timed out for {path.name}.") from error
    if process.returncode:
        raise VividError(
            "invalid_media", f"Unable to read {path.name}.", {"stderr": process.stderr[-3000:]}
        )
    try:
        data = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise VividError("invalid_media", "ffprobe returned invalid JSON.") from error
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    durations = []
    for candidate in [fmt.get("duration"), *(s.get("duration") for s in streams)]:
        try:
            number = float(candidate)
            if math.isfinite(number) and number >= 0:
                durations.append(number)
        except (TypeError, ValueError):
            pass
    try:
        start = float(fmt.get("start_time", 0))
        start = start if math.isfinite(start) else 0.0
    except (ValueError, TypeError):
        start = 0.0
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        kind = "image"
    elif any(
        s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        for s in streams
    ):
        kind = "video"
    elif any(s.get("codec_type") == "audio" for s in streams):
        kind = "audio"
    else:
        raise VividError("invalid_media", f"{path.name} has no supported video or audio stream.")
    stream_ends = []
    for stream in streams:
        try:
            length = float(stream.get("duration", 0))
            origin = float(stream.get("start_time", start))
            if math.isfinite(length) and length > 0 and math.isfinite(origin):
                stream_ends.append(max(0.0, origin - start) + length)
        except (TypeError, ValueError):
            pass
    duration = max(stream_ends) if stream_ends else max(durations, default=0.0)
    return {
        "kind": kind,
        "duration": duration,
        "format_duration": fmt.get("duration"),
        "start_time": start,
        "streams": streams,
        "format": fmt,
    }


def _file_stat(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev]


class MemoryStore:
    """All raw sources stay in materials; all generated evidence stays in root."""

    def __init__(self, root: Path, materials: Path):
        self.root = managed_path(Path(root).expanduser())
        self.materials = Path(materials).expanduser().resolve()
        if not self.materials.is_dir():
            raise VividError(
                "missing_materials", f"Materials directory does not exist: {self.materials}"
            )
        if self.root == self.materials or self.root.is_relative_to(self.materials):
            raise VividError(
                "invalid_memory_root", "Memory must be outside the read-only materials directory."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence_dir = managed_path(self.root / "evidence")
        self.blob_dir = managed_path(self.evidence_dir / "blobs")
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        managed_path(self.root / "staging").mkdir(exist_ok=True)
        self.db_path = managed_path(self.root / "memory.sqlite")
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY, relative_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL, metadata TEXT NOT NULL,
                    stat_json TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
                    first_seen REAL NOT NULL, last_seen REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS assets_path ON assets(relative_path,active);
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                    asset_id TEXT NOT NULL REFERENCES assets(asset_id),
                    created_at REAL NOT NULL, start REAL, end REAL,
                    payload TEXT NOT NULL, files TEXT NOT NULL,
                    search_text TEXT NOT NULL, request_key TEXT
                );
                CREATE INDEX IF NOT EXISTS evidence_asset_time ON evidence(asset_id,start,end);
                CREATE INDEX IF NOT EXISTS evidence_cache ON evidence(asset_id,kind,request_key);
                CREATE TABLE IF NOT EXISTS annotations (
                    annotation_id TEXT PRIMARY KEY,
                    evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
                    text TEXT NOT NULL, tags TEXT NOT NULL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL
                );
            """)

    @contextlib.contextmanager
    def _connect(self):
        for path in (
            self.root,
            self.db_path,
            self.db_path.with_name("memory.sqlite-wal"),
            self.db_path.with_name("memory.sqlite-shm"),
        ):
            managed_path(path)
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _source_path(self, relative: str) -> Path:
        path = (self.materials / relative).resolve()
        if not path.is_relative_to(self.materials) or path == self.materials:
            raise VividError("path_escape", "Source paths must stay inside materials.")
        if not path.is_file():
            raise VividError("source_missing", f"Source is no longer available: {relative}")
        return path

    def _artifact_path(self, path: Path) -> Path:
        managed_path(self.root)
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.root) or resolved == self.root:
            raise VividError("path_escape", "Evidence files must stay inside the memory directory.")
        if not resolved.is_file():
            raise VividError(
                "evidence_missing", f"Evidence artifact does not exist: {resolved.name}"
            )
        return resolved

    def staging_directory(self) -> Path:
        return managed_path(self.root / "staging")

    def _asset_row(self, value: str) -> sqlite3.Row:
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 1024
            or Path(value).is_absolute()
            or ".." in Path(value).parts
        ):
            raise VividError(
                "invalid_asset_id", "Use an asset ID or an unambiguous relative material filename."
            )
        with self._connect() as connection:
            direct = connection.execute(
                "SELECT * FROM assets WHERE asset_id=?", (value,)
            ).fetchone()
            if direct is not None:
                return direct
            rows = connection.execute("SELECT * FROM assets WHERE active=1").fetchall()
        matches = [
            row
            for row in rows
            if value
            in {
                row["relative_path"],
                Path(row["relative_path"]).name,
                Path(row["relative_path"]).stem,
            }
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise VividError(
                "ambiguous_asset", "Filename matches multiple assets; use the canonical asset ID."
            )
        raise VividError("unknown_asset", f"Unknown asset: {value}. Call catalog first.")

    def _asset_dict(self, row: sqlite3.Row, *, validate: bool = True) -> dict[str, Any]:
        metadata = json.loads(row["metadata"])
        derived = metadata.pop("_derived_path", None)
        if derived:
            path = managed_path(Path(derived))
            if not path.is_relative_to(self.root / "project" / "media") or not path.is_file():
                raise VividError("invalid_media", "Registered derived media is unavailable")
        else:
            path = self._source_path(row["relative_path"])
        if validate and _file_stat(path) != json.loads(row["stat_json"]):
            if sha256_file(path) != row["sha256"]:
                raise VividError(
                    "source_changed",
                    "Original material changed. Refresh catalog and use its new asset ID.",
                    {"asset_id": row["asset_id"]},
                )
            with self._connect() as connection:
                connection.execute(
                    "UPDATE assets SET stat_json=? WHERE asset_id=?",
                    (_json(_file_stat(path)), row["asset_id"]),
                )
        return {
            **metadata,
            "asset_id": row["asset_id"],
            "id": row["asset_id"],
            "path": str(path),
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "active": bool(row["active"]),
        }

    def register_derived(self, identity, path, sha256, metadata):
        path = managed_path(path)
        if not path.is_relative_to(self.root / "project" / "media"):
            raise VividError("invalid_media", "Derived media must use managed version storage")
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO assets VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(asset_id) DO NOTHING",
                (
                    identity,
                    path.name,
                    sha256,
                    _json({**metadata, "_derived_path": str(path)}),
                    _json(_file_stat(path)),
                    2,
                    now,
                    now,
                ),
            )

    def catalog(self, refresh: bool = False) -> dict[str, Any]:
        with self._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM assets WHERE active=1").fetchone()[0]
        errors = []
        if refresh or count == 0:
            seen = []
            for candidate in sorted(self.materials.rglob("*")):
                check_budget()
                if (
                    candidate.suffix.lower() not in MEDIA_EXTENSIONS | DOCUMENT_EXTENSIONS
                    or not candidate.is_file()
                ):
                    continue
                try:
                    resolved = candidate.resolve()
                    if not resolved.is_relative_to(self.materials):
                        errors.append(
                            {
                                "file": str(candidate.relative_to(self.materials)),
                                "error": "symlink_outside_materials",
                            }
                        )
                        continue
                    relative = str(candidate.relative_to(self.materials))
                    stat = _file_stat(resolved)
                    with self._connect() as connection:
                        previous = connection.execute(
                            "SELECT * FROM assets WHERE relative_path=? AND active=1", (relative,)
                        ).fetchone()
                    if previous is not None and stat == json.loads(previous["stat_json"]):
                        seen.append(previous["asset_id"])
                        continue
                    digest = sha256_file(resolved)
                    if previous is not None and digest == previous["sha256"]:
                        with self._connect() as connection:
                            connection.execute(
                                "UPDATE assets SET stat_json=?,last_seen=? WHERE asset_id=?",
                                (_json(stat), time.time(), previous["asset_id"]),
                            )
                        seen.append(previous["asset_id"])
                        continue
                    if candidate.suffix.lower() in DOCUMENT_EXTENSIONS:
                        metadata = {
                            "kind": "document",
                            "duration": 0.0,
                            "start_time": 0.0,
                            "streams": [],
                            "format": {},
                            "bytes": stat[0],
                        }
                    else:
                        metadata = {**probe_media(resolved), "bytes": stat[0]}
                    identity = hashlib.sha256((relative + "\0" + digest).encode()).hexdigest()[:24]
                    asset_id = "asset_" + identity
                    now = time.time()
                    with self._connect() as connection:
                        connection.execute(
                            "UPDATE assets SET active=0 WHERE relative_path=?", (relative,)
                        )
                        connection.execute(
                            "INSERT INTO assets VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(asset_id) DO UPDATE SET active=1,stat_json=excluded.stat_json,last_seen=excluded.last_seen",
                            (asset_id, relative, digest, _json(metadata), _json(stat), 1, now, now),
                        )
                    seen.append(asset_id)
                except VividError as error:
                    if error.code in {"time_budget", "operation_cancelled"}:
                        raise
                    errors.append(
                        {"file": str(candidate.relative_to(self.materials)), "error": str(error)}
                    )
                except OSError as error:
                    errors.append(
                        {"file": str(candidate.relative_to(self.materials)), "error": str(error)}
                    )
            with self._connect() as connection:
                if seen:
                    connection.execute(
                        "UPDATE assets SET active=0 WHERE asset_id NOT IN ("
                        + ",".join("?" for _ in seen)
                        + ")",
                        seen,
                    )
                else:
                    connection.execute("UPDATE assets SET active=0")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM assets WHERE active=1 ORDER BY relative_path"
            ).fetchall()
        assets = []
        for row in rows:
            try:
                assets.append(self._asset_dict(row))
            except VividError as error:
                errors.append({"file": row["relative_path"], "error": str(error)})
        return {
            "assets": assets,
            "count": len(assets),
            "errors": errors,
            "materials": str(self.materials),
            "memory_root": str(self.root),
            "semantics": "Catalogued sources are accessible, not necessarily observed.",
        }

    def asset(self, asset_id: str) -> dict[str, Any]:
        try:
            row = self._asset_row(asset_id)
        except VividError as error:
            if getattr(error, "code", None) == "unknown_asset":
                with self._connect() as connection:
                    any_assets = connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
                if not any_assets:
                    self.catalog()
                    row = self._asset_row(asset_id)
                else:
                    raise
            else:
                raise
        return self._asset_dict(row)

    def register_evidence(
        self, kind: str, asset_id: str, payload: dict[str, Any], files: list[Path] | None = None
    ) -> dict[str, Any]:
        if not isinstance(kind, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", kind):
            raise VividError("invalid_evidence_kind", "Use a short lowercase evidence kind.")
        if not isinstance(payload, dict):
            raise VividError("invalid_evidence", "Evidence payload must be an object.")
        asset = self.asset(asset_id)
        managed_path(self.blob_dir)
        encoded = _json(payload)
        if len(encoded.encode()) > 2 * 1024 * 1024:
            raise VividError(
                "evidence_too_large", "Store large binary data in evidence files, not JSON."
            )
        copied = []
        replacements = {}
        for value in files or []:
            source = self._artifact_path(Path(value))
            digest = sha256_file(source)
            suffix = (
                source.suffix.lower()
                if re.fullmatch(r"\.[a-z0-9]{1,8}", source.suffix.lower())
                else ".bin"
            )
            destination = self.blob_dir / (digest + suffix)
            if not destination.exists():
                with tempfile.NamedTemporaryFile(dir=self.blob_dir, delete=False) as stream:
                    tmp = Path(stream.name)
                try:
                    shutil.copyfile(source, tmp)
                    if sha256_file(tmp) != digest:
                        raise VividError(
                            "evidence_changed", "Evidence changed while being archived."
                        )
                    tmp.chmod(0o444)
                    os.replace(tmp, destination)
                finally:
                    tmp.unlink(missing_ok=True)
            elif sha256_file(destination) != digest:
                raise VividError("evidence_corrupt", "An archived evidence blob has been modified.")
            replacements[str(source)] = str(destination)
            copied.append(
                {
                    "path": str(destination),
                    "sha256": digest,
                    "bytes": source.stat().st_size,
                    "mime_type": mimetypes.guess_type(source.name)[0] or "application/octet-stream",
                    "name": source.name,
                }
            )

        def remap(value: Any) -> Any:
            if isinstance(value, str):
                return replacements.get(value, value)
            if isinstance(value, list):
                return [remap(item) for item in value]
            if isinstance(value, dict):
                return {key: remap(item) for key, item in value.items()}
            return value

        payload = remap(json.loads(encoded))
        interval = payload.get("time_range", payload.get("range", [None, None]))
        start, end = interval if isinstance(interval, list) and len(interval) == 2 else [None, None]
        for number in [start, end]:
            if number is not None and (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
            ):
                raise VividError("invalid_time_range", "Evidence times must be finite seconds.")
        if start is not None and end is not None and (start < 0 or end < start):
            raise VividError(
                "invalid_time_range", "Evidence time ranges must be ordered and nonnegative."
            )
        evidence_id = "ev_" + uuid.uuid4().hex[:24]
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    evidence_id,
                    kind,
                    asset["asset_id"],
                    now,
                    start,
                    end,
                    _json(payload),
                    _json(copied),
                    _json(payload).casefold(),
                    payload.get("request_key"),
                ),
            )
            connection.execute(
                "INSERT INTO events(kind,payload,created_at) VALUES(?,?,?)",
                (
                    "evidence_registered",
                    _json(
                        {"evidence_id": evidence_id, "kind": kind, "asset_id": asset["asset_id"]}
                    ),
                    now,
                ),
            )
        result = self.evidence(evidence_id)
        atomic_json(self.evidence_dir / (evidence_id + ".json"), result)
        return result

    def evidence(self, evidence_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT e.*,a.sha256 AS source_sha256 FROM evidence e JOIN assets a ON a.asset_id=e.asset_id WHERE evidence_id=?",
                (evidence_id,),
            ).fetchone()
            annotations = connection.execute(
                "SELECT * FROM annotations WHERE evidence_id=? ORDER BY created_at", (evidence_id,)
            ).fetchall()
        if row is None:
            raise VividError("unknown_evidence", f"Unknown evidence: {evidence_id}")
        files = json.loads(row["files"])
        for file in files:
            path = self._artifact_path(Path(file["path"]))
            if sha256_file(path) != file["sha256"]:
                raise VividError(
                    "evidence_corrupt",
                    "Archived evidence no longer matches its hash.",
                    {"evidence_id": evidence_id},
                )
        source_status = "current"
        try:
            self.asset(row["asset_id"])
        except VividError as error:
            source_status = getattr(error, "code", "unavailable")
        payload = json.loads(row["payload"])
        return {
            **payload,
            "evidence_id": row["evidence_id"],
            "id": row["evidence_id"],
            "kind": row["kind"],
            "asset_id": row["asset_id"],
            "source_sha256": row["source_sha256"],
            "source_status": source_status,
            "created_at": row["created_at"],
            "time_range": [row["start"], row["end"]],
            "payload": payload,
            "files": files,
            "annotations": [
                {
                    "annotation_id": a["annotation_id"],
                    "text": a["text"],
                    "tags": json.loads(a["tags"]),
                    "created_at": a["created_at"],
                }
                for a in annotations
            ],
        }

    def cached_evidence(self, asset_id: str, kind: str, request_key: str) -> dict[str, Any] | None:
        asset = self.asset(asset_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT evidence_id FROM evidence WHERE asset_id=? AND kind=? AND request_key=? ORDER BY created_at DESC LIMIT 1",
                (asset["asset_id"], kind, request_key),
            ).fetchone()
        if row is None:
            return None
        result = self.evidence(row["evidence_id"])
        result["cache_hit"] = True
        return result

    def observation_summary(self, asset_id):
        """Bounded navigation: sample points are never reported as full coverage."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT kind,start,end,payload FROM evidence WHERE asset_id=? "
                "AND kind IN ('video_frame','audio') ORDER BY created_at DESC LIMIT 257",
                (asset_id,),
            ).fetchall()
        samples = sorted({r[1] for r in rows[:256] if r[0] == "video_frame"})
        audio = [
            {
                "time_range": [r[1], r[2]],
                "transcribed": bool(json.loads(r[3]).get("transcription", {}).get("available")),
            }
            for r in rows[:256]
            if r[0] == "audio"
        ]
        return {
            "sample_timestamps": samples,
            "audio_windows": audio,
            "truncated": len(rows) > 256,
            "semantics": "Observed sample points and audio windows only. Gaps and sampled spans have not been continuously checked; candidates still need confirmation.",
        }

    def annotate(
        self, evidence_id: str, text: str, tags: list[str] | None = None
    ) -> dict[str, Any]:
        self.evidence(evidence_id)
        if not isinstance(text, str) or not text.strip() or len(text) > 64000:
            raise VividError("invalid_annotation", "Annotation must contain 1–64000 characters.")
        tags = tags or []
        if (
            not isinstance(tags, list)
            or len(tags) > 64
            or any(not isinstance(t, str) or len(t) > 128 for t in tags)
        ):
            raise VividError("invalid_tags", "Provide at most 64 short string tags.")
        annotation_id = "ann_" + uuid.uuid4().hex[:20]
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO annotations VALUES(?,?,?,?,?)",
                (annotation_id, evidence_id, text, _json(tags), time.time()),
            )
        self.record_event(
            "evidence_annotated", {"evidence_id": evidence_id, "annotation_id": annotation_id}
        )
        return {
            "annotation_id": annotation_id,
            "evidence_id": evidence_id,
            "text": text,
            "tags": tags,
            "status": "agent_interpretation_not_ground_truth",
        }

    def search(
        self,
        query: str = "",
        asset_id: str | None = None,
        kind: str | None = None,
        start: float | None = None,
        end: float | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        if not isinstance(query, str) or len(query) > 4096:
            raise VividError(
                "invalid_query", "Search query must be a string of at most 4096 characters."
            )
        if type(limit) is not int or not 1 <= limit <= 100:
            raise VividError("invalid_limit", "Search limit must be 1–100.")
        for value in [start, end]:
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise VividError(
                    "invalid_time_range", "Search times must be finite nonnegative seconds."
                )
        if start is not None and end is not None and end < start:
            raise VividError("invalid_time_range", "Search end must not precede start.")
        clauses, parameters = [], []
        if asset_id is not None:
            clauses.append("e.asset_id=?")
            parameters.append(self.asset(asset_id)["asset_id"])
        if kind is not None:
            clauses.append("e.kind=?")
            parameters.append(kind)
        if start is not None:
            clauses.append("(e.end IS NULL OR e.end>=?)")
            parameters.append(start)
        if end is not None:
            clauses.append("(e.start IS NULL OR e.start<=?)")
            parameters.append(end)
        terms = query.casefold().split()
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT e.*,COALESCE((SELECT GROUP_CONCAT(text || " " || tags," ") FROM annotations a WHERE a.evidence_id=e.evidence_id),"") AS annotation_text FROM evidence e'
                + (" WHERE " + " AND ".join(clauses) if clauses else "")
                + " ORDER BY e.created_at DESC",
                parameters,
            ).fetchall()
        results = []
        total = 0
        for row in rows:
            text = row["search_text"] + " " + row["annotation_text"].casefold()
            if not all(term in text for term in terms):
                continue
            total += 1
            if len(results) < limit:
                payload = json.loads(row["payload"])
                transcription = payload.get("transcription")
                transcript_text = (
                    transcription.get("text", "") if isinstance(transcription, dict) else ""
                )
                description = (
                    row["annotation_text"]
                    or payload.get("text")
                    or payload.get("description")
                    or transcript_text
                    or row["search_text"]
                )
                if not isinstance(description, str):
                    description = _json(description)
                results.append(
                    {
                        "evidence_id": row["evidence_id"],
                        "asset_id": row["asset_id"],
                        "kind": row["kind"],
                        "time_range": [row["start"], row["end"]],
                        "created_at": row["created_at"],
                        "excerpt": description[:800],
                    }
                )
        return {
            "matches": results,
            "total": total,
            "truncated": total > limit,
            "semantics": "Search covers indexed observations and annotations only. No match does not establish absence in raw media.",
        }

    def record_event(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if (
            not isinstance(kind, str)
            or not kind
            or len(kind) > 128
            or not isinstance(payload, dict)
        ):
            raise VividError(
                "invalid_event", "Events require a short kind and JSON object payload."
            )
        encoded = _json(payload)
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO events(kind,payload,created_at) VALUES(?,?,?)", (kind, encoded, now)
            )
            event_id = cursor.lastrowid
        return {"event_id": event_id, "kind": kind, "payload": payload, "created_at": now}

    def history(self, limit: int = 50) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise VividError("invalid_limit", "History limit must be 1–1000.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events ORDER BY event_id DESC LIMIT ?", (limit,)
            ).fetchall()
            total = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {
            "events": [
                {
                    "event_id": r["event_id"],
                    "kind": r["kind"],
                    "payload": json.loads(r["payload"]),
                    "created_at": r["created_at"],
                }
                for r in reversed(rows)
            ],
            "total": total,
            "truncated": total > limit,
        }

    def coverage(self, asset_id: str | None = None) -> dict[str, Any]:
        assets = [self.asset(asset_id)] if asset_id is not None else self.catalog()["assets"]
        result = []
        with self._connect() as connection:
            for asset in assets:
                rows = connection.execute(
                    "SELECT evidence_id,kind,payload FROM evidence WHERE asset_id=? ORDER BY created_at",
                    (asset["asset_id"],),
                ).fetchall()
                video_frames, audio_windows = [], []
                for row in rows:
                    payload = json.loads(row["payload"])
                    if row["kind"] == "video_frame":
                        video_frames.append(
                            {
                                "evidence_id": row["evidence_id"],
                                "time": payload.get("timestamp"),
                                "pts": payload.get("pts"),
                                "roi": payload.get("roi"),
                                "pixel_format": payload.get("pixel_format"),
                            }
                        )
                    if row["kind"] == "audio":
                        audio_windows.append(
                            {
                                "evidence_id": row["evidence_id"],
                                "time_range": payload.get("time_range"),
                                "sample_range": payload.get("sample_range"),
                                "planner_input": payload.get("planner_input", "analysis_only"),
                                "transcribed": bool(
                                    isinstance(payload.get("transcription"), dict)
                                    and payload["transcription"].get("available")
                                ),
                            }
                        )
                result.append(
                    {
                        "asset_id": asset["asset_id"],
                        "duration": asset["duration"],
                        "video_frames": video_frames,
                        "audio_windows": audio_windows,
                        "fully_observed": False,
                    }
                )
        return {
            "assets": result,
            "semantics": "These are retrieved samples, not proof of complete perception. Audio windows are analysis/transcription unless the backend explicitly confirms native audio delivery.",
        }
