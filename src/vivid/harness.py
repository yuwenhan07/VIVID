"""Small evidence service for agents that edit media with native tools."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from jsonschema import Draft202012Validator

from vivid.errors import VividError
from vivid.harness_protocol import EXAMPLES, LEGACY_SPECS, SPECS
from vivid.media import MediaTools
from vivid.memory import (
    MEDIA_EXTENSIONS,
    MemoryStore,
    _file_stat,
    _json,
    probe_media,
    sha256_file,
)
from vivid.runtime import check_budget, operation_budget, remaining, snapshot_matches, stage
from vivid.verification import _copy_file, verify_file
from vivid.workspace import Workspace, _read_object, atomic_json, file_lock, managed_path


class HarnessMemory(MemoryStore):
    """MemoryStore decoding/cache contracts backed by immutable workspace versions."""

    def __init__(self, root, workspace):
        super().__init__(root, workspace.materials)
        self.workspace = workspace
        self.versions = managed_path(self.root / "versions")
        self.versions.mkdir(exist_ok=True)

    def _asset_dict(self, row, *, validate=True):
        metadata = json.loads(row["metadata"])
        if "_harness_snapshot" not in metadata:
            return super()._asset_dict(row, validate=validate)
        snapshot = managed_path(Path(metadata.pop("_harness_snapshot")))
        origin = metadata.pop("_harness_origin")
        if not snapshot.is_relative_to(self.versions) or not snapshot.is_file():
            raise VividError("version_missing", "The archived media version is unavailable.")
        if validate and not snapshot_matches(snapshot, row["sha256"]):
            raise VividError(
                "version_corrupt", "The archived media version no longer matches its hash."
            )
        return {
            **metadata,
            "asset_id": row["asset_id"],
            "id": row["asset_id"],
            "version_id": row["asset_id"],
            "path": str(snapshot),
            "origin_path": origin,
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "active": bool(row["active"]),
            "archived": not bool(row["active"]),
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
        }

    def register_path(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.workspace.root) or path == self.workspace.root:
            raise VividError("path_escape", "Files must stay inside the workspace.")
        if not path.is_file():
            raise VividError("file_missing", "File does not exist inside the workspace.", str(path))
        relative = str(path.relative_to(self.workspace.root))
        with file_lock(self.root / ".versions.lock"):
            # Never infer identity from a mutable filename or its timestamp alone.
            digest = sha256_file(path)
            identity = (
                "asset_" + hashlib.sha256((relative + "\0" + digest).encode()).hexdigest()[:24]
            )
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT * FROM assets WHERE asset_id=?", (identity,)
                ).fetchone()
            if existing is not None:
                asset = self._asset_dict(existing)
                with self._connect() as connection:
                    connection.execute(
                        "UPDATE assets SET active=0 WHERE relative_path=?", (relative,)
                    )
                    connection.execute(
                        "UPDATE assets SET active=1,last_seen=? WHERE asset_id=?",
                        (time.time(), identity),
                    )
                return {**asset, "active": True, "archived": False}
            suffix = path.suffix.lower() if len(path.suffix) <= 12 else ".bin"
            destination = managed_path(self.versions / (digest + suffix))
            if not destination.exists():
                with tempfile.NamedTemporaryFile(dir=self.versions, delete=False) as stream:
                    temporary = Path(stream.name)
                try:
                    _copy_file(path, temporary)
                    if sha256_file(temporary) != digest or sha256_file(path) != digest:
                        raise VividError(
                            "file_changed", "File changed while its version was captured; retry."
                        )
                    temporary.chmod(0o444)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            elif sha256_file(destination) != digest:
                raise VividError(
                    "version_corrupt", "An archived version no longer matches its hash."
                )
            if path.suffix.lower() in MEDIA_EXTENSIONS:
                try:
                    metadata = probe_media(destination)
                except VividError as exc:
                    if exc.code != "invalid_media":
                        raise
                    metadata = {
                        "kind": "unreadable_media",
                        "duration": 0.0,
                        "start_time": 0.0,
                        "streams": [],
                        "format": {},
                        "probe_error": exc.as_dict()["error"],
                    }
            else:
                metadata = {
                    "kind": "document",
                    "duration": 0.0,
                    "start_time": 0.0,
                    "streams": [],
                    "format": {},
                }
            metadata.update(
                bytes=destination.stat().st_size,
                _harness_snapshot=str(destination),
                _harness_origin=str(path),
                mapping={"type": "identity", "source_version_id": identity}
                if path.is_relative_to(self.workspace.materials)
                else {"type": "unknown"},
            )
            now = time.time()
            with self._connect() as connection:
                connection.execute("UPDATE assets SET active=0 WHERE relative_path=?", (relative,))
                connection.execute(
                    "INSERT INTO assets VALUES(?,?,?,?,?,?,?,?)",
                    (
                        identity,
                        relative,
                        digest,
                        _json(metadata),
                        _json(_file_stat(destination)),
                        1,
                        now,
                        now,
                    ),
                )
            self.record_event(
                "media_version_registered",
                {
                    "asset_id": identity,
                    "path": str(path),
                    "sha256": digest,
                    "mapping": metadata["mapping"],
                },
            )
            return self.asset(identity)

    def catalog(self, refresh=False, limit=100, offset=0):
        errors = []
        if refresh:
            candidates = 0
            for folder in (
                self.workspace.materials,
                self.workspace.root / "inputs",
                self.workspace.root / "work",
                self.workspace.output,
            ):
                if not folder.is_dir():
                    continue
                for candidate in sorted(folder.rglob("*")):
                    if candidate.is_relative_to(self.workspace.state):
                        continue
                    if not candidate.is_file() or candidate.suffix.lower() not in MEDIA_EXTENSIONS:
                        continue
                    check_budget()
                    candidates += 1
                    if candidates > 5000:
                        errors.append({"error": "catalog_scan_limit", "limit": 5000})
                        break
                    try:
                        self.register_path(candidate)
                    except (VividError, OSError) as exc:
                        if isinstance(exc, VividError) and exc.code in {
                            "time_budget",
                            "operation_cancelled",
                        }:
                            raise
                        if len(errors) < 100:
                            errors.append(
                                {
                                    "file": str(candidate.relative_to(self.workspace.root)),
                                    "error": str(exc)[:500],
                                }
                            )
                if candidates > 5000:
                    break
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT asset_id,metadata FROM assets WHERE active=1"
                ).fetchall()
                for row in rows:
                    origin = json.loads(row["metadata"]).get("_harness_origin")
                    if origin and not Path(origin).is_file():
                        connection.execute(
                            "UPDATE assets SET active=0 WHERE asset_id=?", (row["asset_id"],)
                        )
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM assets WHERE active=1").fetchone()[0]
            rows = connection.execute(
                "SELECT * FROM assets WHERE active=1 ORDER BY relative_path LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        assets = []
        for row in rows:
            try:
                assets.append(self._asset_dict(row))
            except VividError as exc:
                errors.append({"asset_id": row["asset_id"], "error": str(exc)[:500]})
        return {
            "assets": assets,
            "count": len(assets),
            "total": total,
            "offset": offset,
            "truncated": offset + len(rows) < total,
            "errors": errors,
            "memory_root": str(self.root),
            "semantics": "Catalogued files are accessible, not observed.",
        }


class HarnessService:
    def __init__(self, workspace: Workspace | Path | str, max_calls=None, persistent=True):
        self.workspace = (
            workspace if isinstance(workspace, Workspace) else Workspace.open(workspace)
        )
        self._explicit_max_calls = max_calls is not None
        max_calls = 10000 if max_calls is None else max_calls
        if type(max_calls) is not int or max_calls < 1 or type(persistent) is not bool:
            raise VividError(
                "invalid_arguments", "max_calls must be positive and persistent a boolean."
            )
        self.max_calls = max_calls
        self._temporary = None
        root = self.workspace.state / "lean-memory"
        if not persistent:
            self._temporary = tempfile.TemporaryDirectory(
                prefix="lean-memory-", dir=self.workspace.state
            )
            root = Path(self._temporary.name)
        self.memory = HarnessMemory(root, self.workspace)
        self.store = self.memory
        self.media = MediaTools(self.memory)
        self.validators = {
            name: Draft202012Validator(schema) for name, _, schema in [*LEGACY_SPECS, *SPECS]
        }
        self.calls = 0

    def definitions(self):
        return [
            {"name": name, "description": description, "inputSchema": copy.deepcopy(schema)}
            for name, description, schema in SPECS
        ]

    def _path(self, value):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.workspace.root / path
        path = path.resolve()
        if not path.is_relative_to(self.workspace.root) or path == self.workspace.root:
            raise VividError("path_escape", "Files must stay inside the workspace.")
        return path

    def register_path(self, path):
        return self.memory.register_path(self._path(path))

    def _input(self, arguments):
        path, identity = arguments.get("path"), arguments.get("asset_id")
        if bool(path) == bool(identity):
            raise VividError(
                "invalid_arguments",
                "Provide exactly one of path or asset_id.",
                {"paths": [["path"], ["asset_id"]], "example": EXAMPLES["inspect_video"]},
            )
        if path:
            return self.register_path(path)
        return self.memory.asset(identity)

    @staticmethod
    def _safe_json(value, maximum=16384):
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
            return (
                value
                if len(encoded) <= maximum
                else {"truncated": True, "excerpt": encoded[:maximum]}
            )
        except (ValueError, TypeError, RecursionError):
            return {"invalid_json": True}

    def dispatch(self, name, arguments=None):
        started = time.monotonic()
        arguments = {} if arguments is None else arguments
        try:
            with file_lock(self.workspace.state / ".tool_budget.lock"):
                configured = _read_object(
                    self.workspace.state / "tool_budget.json", maximum=16384
                ).get("max_calls", self.max_calls)
                if type(configured) is not int or configured < 1:
                    raise VividError("invalid_state", "Invalid harness tool budget.")
                call_limit = (
                    min(configured, self.max_calls) if self._explicit_max_calls else configured
                )
                usage = _read_object(self.workspace.state / "tool_usage.json", maximum=16384).get(
                    "calls", 0
                )
                if type(usage) is not int or usage < 0:
                    raise VividError("invalid_state", "Invalid harness tool count.")
                self.calls = usage + 1
                atomic_json(self.workspace.state / "tool_usage.json", {"calls": self.calls})
            if self.calls > call_limit:
                raise VividError("tool_budget", "Harness tool-call budget exhausted.")
            if not isinstance(name, str) or name not in self.validators:
                raise VividError(
                    "unknown_tool",
                    "Unknown tool. Use one of the six published tools.",
                    {"tools": list(self.validators)},
                )
            try:
                json.dumps(arguments, allow_nan=False)
            except (ValueError, TypeError, RecursionError) as exc:
                raise VividError(
                    "invalid_arguments",
                    "Arguments must contain finite JSON values.",
                    {"example": EXAMPLES[name]},
                ) from exc
            errors = list(self.validators[name].iter_errors(arguments))
            if errors:
                details = [
                    {
                        "path": list(error.absolute_path),
                        "schema_path": list(error.absolute_schema_path),
                        "message": error.message[:500],
                    }
                    for error in errors[:32]
                ]
                raise VividError(
                    "invalid_arguments",
                    "Arguments do not match the published schema.",
                    {
                        "errors": details,
                        "total_errors": len(errors),
                        "truncated": len(errors) > 32,
                        "allowed_properties": sorted(self.validators[name].schema["properties"]),
                        "example": EXAMPLES[name],
                    },
                )
            with operation_budget(self.workspace.state):
                with stage("tool_" + name):
                    result = self._call(name, copy.deepcopy(arguments))
                check_budget()
            result.setdefault("ok", True)
        except VividError as exc:
            result = exc.as_dict()
        except (ValueError, TypeError, KeyError, OSError) as exc:
            result = VividError("tool_failed", str(exc)[:2000]).as_dict()
        self.memory.record_event(
            "tool_call",
            {
                "name": name if isinstance(name, str) else "<invalid>",
                "arguments": self._safe_json(arguments),
                "success": result.get("ok", False),
                "error": result.get("error"),
                "evidence_id": result.get("evidence_id"),
                "checked_sha256": result.get("checked_sha256"),
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "cache_hit": result.get("cache_hit", result.get("decode_cache_hit", False)),
            },
        )
        from vivid.notes import RecoveryNotes

        try:
            RecoveryNotes(self.workspace).refresh(store=self.memory)
        except (VividError, OSError, ValueError) as exc:
            result["memory_index_error"] = str(exc)[:500]
        result["elapsed_seconds"] = round(time.monotonic() - started, 6)
        seconds = remaining(self.workspace.state)
        if seconds is not None:
            result["budget"] = {
                "remaining_seconds": round(seconds, 3),
                "guidance": "Reserve time for export and final output checks; do not restart running jobs.",
            }
        return result

    def _bound_evidence(self, evidence):
        asset = self.memory.asset(evidence["asset_id"])
        return {
            **evidence,
            "version_id": asset["version_id"],
            "origin_path": asset["origin_path"],
            "mapping": asset["mapping"],
            "archived": asset["archived"],
            "source_status": "archived_version" if asset["archived"] else "version_snapshot",
            "source_snapshot": asset["path"],
        }

    def _call(self, name, args):
        if name in {"inspect", "transcribe", "retrieve", "edit"}:
            from vivid.interaction import route

            return route(self, name, args)
        if name == "catalog":
            if args.get("path"):
                asset = self.register_path(args["path"])
                result = {"assets": [asset], "count": 1, "total": 1}
            else:
                result = self.memory.catalog(
                    refresh=args.get("refresh", True),
                    limit=args.get("limit", 100),
                    offset=args.get("offset", 0),
                )
            result["capabilities"] = {
                "asr": {
                    "backend": os.getenv("VIVID_ASR_BACKEND", "faster-whisper"),
                    "model": os.getenv("VIVID_WHISPER_MODEL", "base"),
                    "availability": "configured, checked when requested",
                    "word_timing": "ASR estimate; no forced alignment model configured",
                },
                "native_tools": "Use available native tools directly; Vivid helpers are optional",
            }
            sidecars = []
            for index, path in enumerate(self.workspace.materials.rglob("*")):
                if index >= 2048:
                    result["subtitle_listing_truncated"] = True
                    break
                if (
                    path.suffix.lower() in {".srt", ".vtt", ".ass"}
                    and path.is_file()
                    and path.resolve().is_relative_to(self.workspace.materials)
                ):
                    sidecars.append(str(path.relative_to(self.workspace.root)))
                    if len(sidecars) >= 64:
                        break
            result["subtitle_files"] = sidecars
            return result
        if name in {"inspect_video", "inspect_audio"}:
            asset = self._input(args)
            args.pop("path", None)
            args.pop("asset_id", None)
            method = (
                self.media.inspect_video if name == "inspect_video" else self.media.inspect_audio
            )
            maximum = (
                float("inf")
                if args.get("mode") == "overview"
                else (600 if name == "inspect_video" else 120)
            )
            start = args.get("start", 0)
            requested_end = args.get("end")
            original_end = requested_end
            adjusted = (
                requested_end is not None
                and asset["kind"] != "image"
                and requested_end > asset["duration"]
            )
            if adjusted:
                requested_end = args["end"] = asset["duration"]
            continuation = None
            if (
                requested_end is not None
                and asset["kind"] != "image"
                and requested_end <= asset["duration"] + 0.001
                and requested_end - start > maximum
            ):
                args["end"] = start + maximum
                continuation = {
                    "asset_id": asset["asset_id"],
                    **args,
                    "start": start + maximum,
                    "end": requested_end,
                }
            budget = _read_object(self.workspace.state / "runtime_budget.json")
            reserve = budget.get("observation_reserve_seconds", 0)
            # Final/intermediate output inspection must remain possible during the
            # reserve. Only new source exploration consumes the observation budget.
            origin = Path(asset.get("origin_path", asset["path"]))
            if not origin.is_relative_to(self.workspace.materials):
                reserve = 0
            with operation_budget(self.workspace.state, reserve_seconds=reserve):
                result = self._bound_evidence(method(asset["asset_id"], **args))
            result["observation_summary"] = self.memory.observation_summary(asset["asset_id"])
            result["complete"] = continuation is None
            result["pending_ranges"] = [[start + maximum, requested_end]] if continuation else []
            if continuation:
                result["requested_time_range"] = [start, original_end]
                result["next_request"] = continuation
            if adjusted:
                result["requested_range"] = [start, original_end]
                result["range_adjusted"] = True
                result["warnings"] = [
                    {
                        "code": "end_clamped",
                        "message": "Observation end exceeded the material duration; clipped to its end.",
                        "requested_end": original_end,
                        "actual_end": requested_end,
                    }
                ]
            return result
        if name == "search":
            return self._search(args)
        if name == "recall":
            if sum(key in args for key in ("evidence_id", "asset_id", "path")) > 1:
                raise VividError(
                    "invalid_arguments", "Recall accepts one of evidence_id, asset_id, path."
                )
            if "evidence_id" in args:
                return self._bound_evidence(self.memory.evidence(args["evidence_id"]))
            if "asset_id" in args:
                return {"asset": self.memory.asset(args["asset_id"])}
            if "path" in args:
                relative = str(self._path(args["path"]).relative_to(self.workspace.root))
                with self.memory._connect() as connection:
                    rows = connection.execute(
                        "SELECT * FROM assets WHERE relative_path=? ORDER BY first_seen DESC LIMIT ?",
                        (relative, args.get("limit", 20)),
                    ).fetchall()
                    total = connection.execute(
                        "SELECT COUNT(*) FROM assets WHERE relative_path=?", (relative,)
                    ).fetchone()[0]
                return {
                    "versions": [self.memory._asset_dict(row) for row in rows],
                    "total": total,
                    "truncated": total > len(rows),
                }
            return self.memory.history(args.get("limit", 20))
        if name == "verify":
            review = {k: args.pop(k) for k in ("timeline", "captions", "requirements") if k in args}
            path = self._path(args.pop("path"))
            asset = self.register_path(path)
            with stage("verification"):
                result = verify_file(
                    asset["path"],
                    cache_dir=self.workspace.state / "verification",
                    _immutable=True,
                    **args,
                )
            result.update(
                path=str(path),
                version_id=asset["version_id"],
                checked_snapshot=asset["path"],
                mapping=asset["mapping"],
            )
            if path.is_file() and sha256_file(path) != asset["sha256"]:
                result.update(
                    ok=False,
                    error={"code": "file_changed", "message": "File changed during verification."},
                )
            elif not path.is_file():
                result.update(
                    ok=False,
                    error={
                        "code": "file_missing",
                        "message": "File disappeared during verification.",
                    },
                )
            if review:
                from vivid.review import plan_output_checks

                result["review"] = plan_output_checks(
                    **review, duration=result.get("media", {}).get("duration")
                )
            return result
        raise VividError("unknown_tool", "Unknown tool.")

    def _search(self, args):
        if "path" in args or "asset_id" in args:
            asset = self._input(args)
            args.pop("path", None)
            args["asset_id"] = asset["asset_id"]
        limit = args.get("limit", 20)
        result = self.memory.search(**args)
        matches = result["matches"]
        total = result["total"]
        terms = args.get("query", "").casefold().split()
        if not any(key in args for key in ("asset_id", "kind", "start", "end")):
            for filename in ("GUIDE.md", "WORKING.md"):
                for folder in (self.workspace.state, self.workspace.root):
                    path = folder / filename
                    if not path.is_file():
                        continue
                    path = self._path(str(path))
                    with path.open("r", encoding="utf-8", errors="replace") as stream:
                        text = stream.read(65536)
                    if not text.strip() or not all(term in text.casefold() for term in terms):
                        continue
                    total += 1
                    if len(matches) < limit:
                        folded = text.casefold()
                        position = folded.find(terms[0]) if terms else 0
                        start = max(0, position - 160)
                        matches.append(
                            {
                                "kind": "note",
                                "path": str(path),
                                "excerpt": text[start : start + 800],
                                "truncated": len(text) > 800,
                                "sha256": sha256_file(path),
                            }
                        )
        result.update(
            total=total,
            truncated=total > limit,
            semantics="Literal search of cached evidence/transcripts and GUIDE/WORKING notes only; "
            "no visual semantic search. A miss does not establish absence in raw media.",
        )
        if args.get("asset_id"):
            result["observation_summary"] = self.memory.observation_summary(args["asset_id"])
        return result
