from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .base import NativeBackend, StreamState, token_count
from .provider import api_provider_overrides


class OpenCodeBackend(NativeBackend):
    name = "opencode"

    def command(self, session_id: str | None, segment_dir: Path) -> list[str]:
        model = self.config.model
        if "/" not in model:
            model = f"openai/{model}"
        args = [self.executable(), "run", "--format", "json", "--model", f"openrouter/{model}"]
        if self.config.effort:
            args.extend(["--variant", self.config.effort])
        if session_id:
            args.extend(["--session", session_id])
        args.extend(self.config.extra_args)
        return args

    def environment(self) -> dict[str, str]:
        env = super().environment()
        provider = api_provider_overrides(env)
        env["OPENROUTER_BASE_URL"] = provider["model_providers.vivid_api.base_url"]
        config: dict[str, Any] = {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "share": "disabled",
            "permission": "allow",
            "mcp": {
                "vivid": {
                    "type": "local",
                    "command": self.config.mcp_command,
                    "enabled": True,
                    "timeout": self.config.tool_timeout_seconds * 1000,
                }
            },
        }
        model_id = self.config.model
        if "/" not in model_id:
            model_id = f"openai/{model_id}"
        options = {"baseURL": "{env:OPENROUTER_BASE_URL}", "apiKey": "{env:OPENROUTER_API_KEY}"}
        model_config = {}
        if model_id == "openai/gpt-5.5":
            model_config = {
                "reasoning": True,
                "attachment": True,
                "tool_call": True,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                # Conservative runtime caps also work with older CLI model catalogs.
                "limit": {"context": 100000, "output": 16384},
            }
        config["provider"] = {"openrouter": {"models": {model_id: model_config}, "options": options}}
        if self.config.max_turns is not None:
            config["agent"] = {"build": {"steps": self.config.max_turns}}
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
        env["OPENCODE_DISABLE_AUTOUPDATE"] = "true"
        if self.config.isolate_user_config:
            private = self.config.log_dir / ".opencode"
            env.pop("OPENCODE_CONFIG", None)
            env.pop("OPENCODE_CONFIG_DIR", None)
            env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "true"
            env["OPENCODE_DISABLE_CLAUDE_CODE"] = "true"
            env["XDG_CONFIG_HOME"] = str(private / "config")
            env["XDG_DATA_HOME"] = str(private / "data")
            env["XDG_CACHE_HOME"] = str(private / "cache")
        return env

    def consume(self, event: dict[str, Any], state: StreamState) -> None:
        if event.get("sessionID"):
            state.session_id = str(event["sessionID"])
        kind = event.get("type")
        part = event.get("part") or {}
        if kind == "step_start":
            state.completed = False
        elif kind == "text" and isinstance(part.get("text"), str):
            state.final_text = part["text"]
        elif kind == "step_finish":
            part_id = part.get("id")
            if part_id and str(part_id) in state.message_ids:
                return
            if part_id:
                state.message_ids.add(str(part_id))
            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            cached = token_count(cache.get("read"))
            written = token_count(cache.get("write"))
            inputs = token_count(tokens.get("input")) + cached + written
            for key, value in {
                "input_tokens": inputs,
                "cached_input_tokens": cached,
                "cache_write_input_tokens": written,
                "output_tokens": token_count(tokens.get("output"))
                + token_count(tokens.get("reasoning")),
            }.items():
                state.usage[key] = state.usage.get(key, 0) + value
            state.peak_context_tokens = max(state.peak_context_tokens, inputs)
            cost = part.get("cost")
            if (
                isinstance(cost, (int, float))
                and not isinstance(cost, bool)
                and math.isfinite(cost)
                and cost >= 0
            ):
                state.cost_usd = (state.cost_usd or 0) + cost
            state.completed = part.get("reason") == "stop"
        elif kind == "error":
            state.failed = True
            state.error = json.dumps(event.get("error") or event, ensure_ascii=False)
