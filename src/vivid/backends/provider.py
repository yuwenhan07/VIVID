from __future__ import annotations

from urllib.parse import urlsplit

from vivid.errors import VividError


def api_provider_overrides(env: dict[str, str]) -> dict:
    """Configure OpenRouter for the native agent without embedding credentials."""
    key = env.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise VividError("missing_api_key", "Set OPENROUTER_API_KEY before running an agent")
    base = env.get("OPENROUTER_BASE_URL", "").strip() or "https://openrouter.ai/api/v1"
    parsed = urlsplit(base)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise VividError(
            "invalid_api_endpoint",
            "OPENROUTER_BASE_URL must be an HTTP(S) API base URL without embedded credentials, query or fragment",
        )
    return {
        "model_provider": "vivid_api",
        "model_providers.vivid_api.name": "OpenRouter",
        "model_providers.vivid_api.base_url": base.rstrip("/"),
        "model_providers.vivid_api.env_key": "OPENROUTER_API_KEY",
        "model_providers.vivid_api.wire_api": "responses",
        "model_providers.vivid_api.requires_openai_auth": False,
        "model_providers.vivid_api.supports_websockets": False,
    }
