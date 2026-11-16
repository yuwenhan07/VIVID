from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .base import NativeBackend, StreamState, token_count
from .provider import api_provider_overrides


class ClaudeBackend(NativeBackend):
    name = "claude"

    def command(self, session_id: str | None, segment_dir: Path) -> list[str]:
        config = self.config
        mcp_path = segment_dir / "mcp.json"
        mcp_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "vivid": {
                            "type": "stdio",
                            "command": config.mcp_command[0],
                            "args": config.mcp_command[1:],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        mcp_path.chmod(0o600)
        args = [
            self.executable(),
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--model",
            config.model if "/" in config.model else f"anthropic/{config.model}",
            "--permission-mode",
            config.claude_permission_mode,
            "--mcp-config",
            str(mcp_path),
        ]
        if config.isolate_user_config:
            args.append("--strict-mcp-config")
        # These are permission grants, not a --tools restriction. Native tools remain available.
        if config.claude_permission_mode != "plan":
            args.extend(["--allowedTools", "Read,Write,Edit,Bash,mcp__vivid__*"])
        if config.effort:
            args.extend(["--effort", config.effort])
        if session_id:
            args.extend(["--resume", session_id])
        if config.max_turns is not None:
            args.extend(["--max-turns", str(config.max_turns)])
        args.extend(config.extra_args)
        return args

    def environment(self) -> dict[str, str]:
        env = super().environment()
        provider = api_provider_overrides(env)
        env["ANTHROPIC_BASE_URL"] = provider["model_providers.vivid_api.base_url"].removesuffix(
            "/v1"
        )
        env["ANTHROPIC_AUTH_TOKEN"] = env["OPENROUTER_API_KEY"]
        env["ANTHROPIC_API_KEY"] = ""
        env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
        env.setdefault("MCP_TOOL_TIMEOUT", str(self.config.tool_timeout_seconds * 1000))
        return env

    def consume(self, event: dict[str, Any], state: StreamState) -> None:
        if event.get("session_id"):
            state.session_id = str(event["session_id"])
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            for server in event.get("mcp_servers") or []:
                if (
                    server.get("name") == "vivid"
                    and server.get("status") not in {"connected", "pending"}
                ):
                    state.failed = True
                    state.error = f"VIVID MCP did not connect: {server.get('status', 'unknown')}"
        elif kind == "assistant":
            message = event.get("message") or {}
            content = message.get("content") or []
            texts = [
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ]
            if texts:
                state.final_text = "\n".join(texts)
            usage = message.get("usage") or {}
            message_id = message.get("id")
            # Streaming SDKs may re-emit the same assistant message. Count it once.
            if message_id is None or str(message_id) not in state.message_ids:
                self._add_usage(usage, state)
                if message_id is not None:
                    state.message_ids.add(str(message_id))
        elif kind == "result":
            state.completed = event.get("subtype") == "success" and not event.get("is_error", False)
            state.failed = state.failed or not state.completed
            if isinstance(event.get("result"), str):
                state.final_text = event["result"]
            if not state.completed:
                errors = event.get("errors") or event.get("result") or event.get("subtype")
                state.error = str(errors)
            usage = event.get("usage")
            if isinstance(usage, dict):
                self._set_total_usage(usage, state)
            cost = event.get("total_cost_usd")
            if (
                isinstance(cost, (int, float))
                and not isinstance(cost, bool)
                and math.isfinite(cost)
                and cost >= 0
            ):
                state.cost_usd = float(cost)
        elif kind == "error":
            state.failed = True
            state.error = str(event.get("error") or event.get("message") or "Claude Code error")

    @staticmethod
    def _parts(usage: dict) -> dict[str, int]:
        cache_read = token_count(usage.get("cache_read_input_tokens"))
        cache_write = token_count(usage.get("cache_creation_input_tokens"))
        return {
            "input_tokens": token_count(usage.get("input_tokens")) + cache_read + cache_write,
            "output_tokens": token_count(usage.get("output_tokens")),
            "cached_input_tokens": cache_read,
            "cache_write_input_tokens": cache_write,
        }

    @classmethod
    def _add_usage(cls, usage: dict, state: StreamState) -> None:
        parts = cls._parts(usage)
        for key, value in parts.items():
            state.usage[key] = state.usage.get(key, 0) + value
        state.peak_context_tokens = max(state.peak_context_tokens, parts["input_tokens"])

    @classmethod
    def _set_total_usage(cls, usage: dict, state: StreamState) -> None:
        parts = cls._parts(usage)
        state.usage.update(parts)
        # Aggregate usage is the total billed across turns, not the current context size.
        if not state.peak_context_tokens:
            state.peak_context_tokens = parts["input_tokens"]
