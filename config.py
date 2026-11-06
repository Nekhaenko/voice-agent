"""Configuration: environment-driven settings for the voice assistant.

All endpoint specifics live here; no module hard-codes them (design D).
Missing API key is a configuration error surfaced by the UI (spec
`conversation-agent`, scenario "Missing API key").
"""

import os


def settings() -> dict:
    """Return resolved settings, raising a clear error when the API key is absent."""
    api_key = os.environ.get("ORCAROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Configuration error: ORCAROUTER_API_KEY is not set. "
            "Export the key before starting the assistant."
        )
    return {
        "api_key": api_key,
        "base_url": os.environ.get(
            "ORCAROUTER_BASE_URL", "https://api.orcarouter.ai/v1"
        ),
        "model_name": os.environ.get("ORCAROUTER_MODEL", "orcarouter/free"),
        # router_mode: "web_tool" (bind a search tool) or "json_decision"
        # (structured yes/no decision; deterministic fallback for endpoints
        # with unreliable tool-call support, design D3).
        "router_mode": os.environ.get("ROUTER_MODE", "web_tool"),
        # Local VOSK model path; exact filename resolved at install time
        # (design Open Question). Override with VOSK_MODEL_PATH.
        "vosk_model_path": os.environ.get(
            "VOSK_MODEL_PATH", "models/vosk-model-small-ru-0.22"
        ),
        # DuckDuckGo results to fetch per query (search node).
        "search_max_results": int(os.environ.get("SEARCH_MAX_RESULTS", "5")),
        # Router-quality eval minimum accuracy (router-quality spec).
        "eval_threshold": float(os.environ.get("EVAL_THRESHOLD", "0.8")),
    }
