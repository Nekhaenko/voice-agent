import os

import config


def test_settings_raise_without_api_key(monkeypatch):
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    try:
        config.settings()
    except RuntimeError as exc:
        assert "ORCAROUTER_API_KEY" in str(exc)
    else:
        raise AssertionError("settings() must fail without the API key")


def test_settings_resolve_all_workers(monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "abc")
    monkeypatch.setenv("ROUTER_MODE", "json_decision")
    monkeypatch.setenv("VOSK_MODEL_PATH", "models/custom-ru")
    monkeypatch.setenv("EVAL_THRESHOLD", "0.9")
    cfg = config.settings()
    assert cfg["api_key"] == "abc"
    assert cfg["router_mode"] == "json_decision"
    assert cfg["vosk_model_path"] == "models/custom-ru"
    assert cfg["eval_threshold"] == 0.9
    assert isinstance(cfg["base_url"], str) and cfg["base_url"]
    assert cfg["model_name"] == "orcarouter/free"


def test_router_mode_default_is_web_tool(monkeypatch):
    monkeypatch.delenv("ROUTER_MODE", raising=False)
    assert os.environ.get("ROUTER_MODE") is None
    assert config.settings()["router_mode"] == "web_tool"
