"""Small, file-backed continuation notes and a generated evidence index.

The original brief and model notes have different authority. MEMORY.md is only a
bounded navigation aid; SQLite and archived evidence remain the durable record.
"""

from __future__ import annotations

import json
from typing import Any

from vivid.errors import VividError
from vivid.workspace import Workspace, atomic_text, file_lock, managed_path


class RecoveryNotes:
    LIMITS = {"GUIDE.md": 16384, "WORKING.md": 8192, "MEMORY.md": 16384}
    INDEX_LIMIT = LIMITS["MEMORY.md"]

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.root = workspace.state

    def initialize(self, instruction: str) -> None:
        """Preserve the exact public brief; never silently replace it on resume."""
        with file_lock(self.root / ".recovery-notes.lock"):
            path = managed_path(self.root / "TASK.md")
            if path.exists():
                if path.read_text(encoding="utf-8") != instruction:
                    raise VividError(
                        "task_mismatch",
                        "This memory belongs to another task; TASK.md differs from the original task brief",
                    )
            else:
                atomic_text(path, instruction)
            for name in ("GUIDE.md", "WORKING.md"):
                path = managed_path(self.root / name)
                if not path.exists():
                    atomic_text(path, "")

    @staticmethod
    def _short(value: Any, limit: int = 256) -> str:
        return str(value)[:limit]

    def refresh(self, store=None) -> dict[str, Any]:
        """Regenerate a bounded index after tool calls; never overwrite model notes."""
        if store is None:
            from vivid.harness import HarnessMemory

            store = HarnessMemory(self.root / "lean-memory", self.workspace)
        # Query summaries rather than full payloads, blobs, transcripts or coverage.
        with store._connect() as connection:
            assets = connection.execute(
                "SELECT asset_id,relative_path,sha256 FROM assets WHERE active!=0 ORDER BY relative_path LIMIT 33"
            ).fetchall()
            evidence = connection.execute(
                "SELECT evidence_id,kind,asset_id,start,end FROM evidence ORDER BY created_at DESC LIMIT 25"
            ).fetchall()
            evidence_count = connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
        index = {
            "generated": True,
            "semantics": "Navigation only. Recall evidence by ID. Indexed samples are incomplete; absence is not evidence of absence.",
            "assets": [
                {
                    "asset_id": a["asset_id"],
                    "path": self._short(a["relative_path"]),
                    "sha256": a["sha256"],
                }
                for a in assets[:32]
            ],
            "assets_truncated": len(assets) > 32,
            "recent_evidence": [
                {
                    "evidence_id": e["evidence_id"],
                    "kind": e["kind"],
                    "asset_id": e["asset_id"],
                    "time_range": [e["start"], e["end"]],
                }
                for e in evidence[:24]
            ],
            "evidence_total": evidence_count,
            "evidence_truncated": evidence_count > 24,
            "storage": {"database": str(store.db_path), "evidence": str(store.evidence_dir)},
        }
        # Long state paths must not make this an unbounded prompt injection.
        content = json.dumps(index, ensure_ascii=False, indent=2)
        while len(content) > self.INDEX_LIMIT and (index["assets"] or index["recent_evidence"]):
            key = (
                "assets"
                if len(index["assets"]) >= len(index["recent_evidence"])
                else "recent_evidence"
            )
            index[key].pop()
            index["assets_truncated" if key == "assets" else "evidence_truncated"] = True
            content = json.dumps(index, ensure_ascii=False, indent=2)
        if len(content) > self.INDEX_LIMIT:
            raise VividError("state_limit", "Memory index metadata exceeds its bounded limit")
        with file_lock(self.root / ".recovery-notes.lock"):
            atomic_text(self.root / "MEMORY.md", content)
        return index

    def _note(self, name: str) -> dict[str, Any]:
        path = managed_path(self.root / name)
        maximum = self.LIMITS[name]
        if not path.is_file():
            return {"path": str(path), "content": "", "truncated": False}
        try:
            with path.open(encoding="utf-8") as stream:
                content = stream.read(maximum + 1)
        except UnicodeError as exc:
            raise VividError("invalid_note", f"{name} must contain UTF-8 text") from exc
        return {
            "path": str(path),
            "content": content[:maximum],
            "truncated": len(content) > maximum,
        }

    def pending_jobs(self) -> list[dict[str, str]]:
        """Only report job records. Never execute or restart an inferred job."""
        records = []
        for directory in (self.root / "jobs",):
            if not directory.is_dir() or directory.is_symlink():
                continue
            for path in directory.glob("*.json"):
                if len(records) >= 16:
                    return records
                try:
                    with managed_path(path).open(encoding="utf-8") as stream:
                        text = stream.read(32769)
                    if len(text) > 32768:
                        continue
                    value = json.loads(text)
                except (OSError, ValueError, VividError):
                    continue
                if isinstance(value, dict) and value.get("status") in {
                    "running",
                    "pending",
                    "queued",
                    "started",
                }:
                    records.append(
                        {
                            "path": str(path),
                            "status": str(value["status"]),
                            "meaning": "Recorded status only; inspect actual process and logs before starting work.",
                        }
                    )
        native_logs = self.root / "native_logs"
        if native_logs.is_dir() and not native_logs.is_symlink():
            for directory in native_logs.iterdir():
                if len(records) >= 16:
                    break
                invocation = directory / "invocation.json"
                if invocation.is_file() and not (directory / "result.json").exists():
                    records.append(
                        {
                            "path": str(invocation),
                            "status": "native_result_unknown",
                            "meaning": "Incomplete native invocation may still have child jobs. Inspect logs/processes; do not replay automatically.",
                        }
                    )
        return records

    def snapshot(self) -> dict[str, Any]:
        files = []
        for path in self.workspace.output.rglob("*"):
            if len(files) >= 32:
                break
            if (
                path.is_file()
                and not path.is_symlink()
                and path.resolve().is_relative_to(self.workspace.output)
            ):
                stat = path.stat()
                files.append(
                    {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                )
        return {
            "task_file": str(self.root / "TASK.md"),
            "notes": {name: self._note(name) for name in self.LIMITS},
            "output_files": files,
            "output_inventory_limit": 32,
            "pending_jobs": self.pending_jobs(),
            "recovery_rule": "Reconcile actual files, evidence IDs and job status before editing. Notes are revisable model understanding; TASK.md is the original brief. Do not replay completed edits or restart a pending native job without checking it.",
        }
