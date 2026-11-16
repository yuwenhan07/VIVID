from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from vivid.errors import VividError


def managed_path(path: Path) -> Path:
    """Managed state may not follow links installed by native editing scripts."""
    path = Path(path).absolute()
    if path.resolve() != path:
        raise VividError("invalid_state", "Managed state paths may not contain symlinks", str(path))
    return path


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def atomic_text(path: Path, value: str) -> None:
    path = managed_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".vivid-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path = managed_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@dataclass(frozen=True)
class Workspace:
    root: Path
    materials: Path
    state: Path
    output: Path

    @classmethod
    def open(
        cls,
        root: Path | str,
        materials: Path | str | None = None,
        state: Path | str | None = None,
        output: Path | str | None = None,
    ) -> Workspace:
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        def resolve(value: Path | str | None, default: str) -> Path:
            path = Path(value or default).expanduser()
            path = (path if path.is_absolute() else root / path).resolve()
            if not path.is_relative_to(root) or path == root:
                raise VividError(
                    "workspace_path", "Task paths must be inside the workspace", str(path)
                )
            return path

        paths = [
            resolve(materials, "materials"),
            resolve(state, "work/vivid"),
            resolve(output, "output"),
        ]
        for index, path in enumerate(paths):
            for other in paths[index + 1 :]:
                if path.is_relative_to(other) or other.is_relative_to(path):
                    raise VividError(
                        "overlapping_paths", "Materials, state and output must be disjoint"
                    )
            path.mkdir(parents=True, exist_ok=True)
        return cls(root, *paths)

    def manifest(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in ("root", "materials", "state", "output")}


def _read_object(path: Path, *, maximum: int = 2 * 1024 * 1024) -> dict[str, Any]:
    path = managed_path(path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            content = stream.read(maximum + 1)
    except UnicodeError as exc:
        raise VividError("invalid_state", "State JSON must contain UTF-8 text", str(path)) from exc
    if len(content) > maximum:
        raise VividError("state_limit", "State JSON exceeds its bounded read limit", str(path))
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise VividError(
            "invalid_state", "State file does not contain valid JSON", str(path)
        ) from exc
    if not isinstance(value, dict):
        raise VividError("invalid_state", "State JSON must contain an object", str(path))
    return value

