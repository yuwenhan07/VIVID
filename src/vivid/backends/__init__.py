from __future__ import annotations

from .base import BackendConfig, BackendResult, NativeBackend


def create_backend(config: BackendConfig) -> NativeBackend:
    if config.backend == "codex" and config.codex_transport == "app-server":
        from .app_server import CodexAppServerBackend

        return CodexAppServerBackend(config)
    if config.backend == "codex":
        from .codex import CodexBackend

        return CodexBackend(config)
    if config.backend == "claude":
        from .claude import ClaudeBackend

        return ClaudeBackend(config)
    if config.backend == "opencode":
        from .opencode import OpenCodeBackend

        return OpenCodeBackend(config)
    raise ValueError(f"Unknown backend: {config.backend}")


__all__ = ["BackendConfig", "BackendResult", "NativeBackend", "create_backend"]
