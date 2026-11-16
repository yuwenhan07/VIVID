from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import NativeBackend, StreamState, token_count
from .provider import api_provider_overrides


class CodexBackend(NativeBackend):
    name = "codex"

    def command(self, session_id: str | None, segment_dir: Path) -> list[str]:
        config = self.config
        args = [self.executable(), "exec"]
        if session_id:
            args.append("resume")
        model = config.model if "/" in config.model else f"openai/{config.model}"
        args.extend(["--json", "--skip-git-repo-check", "--model", model])
        if config.isolate_user_config:
            args.append("--ignore-user-config")
        overrides: dict[str, Any] = {
            "approval_policy": "never",
            "sandbox_mode": config.codex_sandbox,
            "mcp_servers.vivid.command": config.mcp_command[0],
            "mcp_servers.vivid.args": config.mcp_command[1:],
            "mcp_servers.vivid.startup_timeout_sec": 60,
            "mcp_servers.vivid.tool_timeout_sec": config.tool_timeout_seconds,
            "mcp_servers.vivid.required": True,
            "mcp_servers.vivid.env_vars": [
                "VIVID_MEDIA_PYTHON",
                "VIVID_WHISPER_MODEL",
                "VIVID_ALLOW_MODEL_DOWNLOAD",
                "VIVID_ASR_BACKEND",
                "VIVID_ASR_THREADS",
                "VIVID_ASR_TIMEOUT",
            ],
        }
        overrides.update(api_provider_overrides(self.environment()))
        if config.effort:
            overrides["model_reasoning_effort"] = config.effort
        for key, value in overrides.items():
            args.extend(["-c", f"{key}={json.dumps(value, ensure_ascii=False)}"])
        args.extend(config.extra_args)
        if session_id:
            args.append(session_id)
        args.append("-")
        return args

    def consume(self, event: dict[str, Any], state: StreamState) -> None:
        kind = event.get("type")
        if kind == "thread.started":
            state.session_id = event.get("thread_id") or state.session_id
        elif kind == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                state.final_text = item["text"]
        elif kind == "turn.completed":
            state.completed = True
            state.failed = False
            state.error = None
            usage = event.get("usage") or {}
            for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
                state.usage[key] += token_count(usage.get(key))
            if "reasoning_output_tokens" in usage:
                state.usage["reasoning_output_tokens"] = state.usage.get(
                    "reasoning_output_tokens", 0
                ) + token_count(usage["reasoning_output_tokens"])
            state.peak_context_tokens = max(
                state.peak_context_tokens,
                token_count(usage.get("input_tokens")),
            )
        elif kind in {"turn.failed", "error"}:
            state.failed = True
            error = event.get("error", event.get("message", "Native Codex reported an error"))
            state.error = (
                error.get("message", str(error)) if isinstance(error, dict) else str(error)
            )
