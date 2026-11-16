from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import shutil
import subprocess
from pathlib import Path


def software_manifest() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": {}}
    for name in ("vivid-harness", "mcp", "numpy", "Pillow", "jsonschema"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    result["source_sha256"] = digest.hexdigest()
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name):
            try:
                process = subprocess.run(
                    [name, "-version"], capture_output=True, text=True, timeout=10
                )
                result[name] = process.stdout.splitlines()[0] if process.stdout else None
            except (OSError, subprocess.SubprocessError):
                result[name] = None
        else:
            result[name] = None
    return result
