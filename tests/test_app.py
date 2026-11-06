"""Interface-level behavior tests (tasks 2.3, 4.2, 4.3)."""

import app
from app import process_exchange, toggle_recording


def fixture_wav_bytes() -> bytes:
    from pathlib import Path

    return (Path(__file__).parent / "fixtures" / "sample_ru.wav").read_bytes()


def test_empty_recording_skips_agent(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("the agent must not run for empty recordings")

    monkeypatch.setattr(app, "_agent_answer", fail)
    history, status, _thread = process_exchange(None, [], None)
    assert "Nothing recognized" in status
    assert history == []


def test_unknown_audio_surfaces_config_missing_key(monkeypatch):
    """Missing ORCAROUTER_API_KEY must surface as a config error, no answer."""

    def fail_settings():
        raise RuntimeError(
            "Configuration error: ORCAROUTER_API_KEY is not set."
        )

    monkeypatch.setattr(app, "settings", fail_settings)
    history, status, _thread = process_exchange(fixture_wav_bytes(), [], None)
    assert "Error" in status
    assert "ORCAROUTER_API_KEY" in status
    assert history == []


def test_scripted_turns_accumulate_in_history(monkeypatch):
    answers = iter(["Ответ один.", "Ответ два."])
    monkeypatch.setattr(
        app, "_agent_answer", lambda tid, q: next(answers)
    )
    wav = fixture_wav_bytes()

    history1, _status, thread = process_exchange(wav, [], None)
    assert history1[0]["role"] == "user"
    assert "привет" in history1[0]["content"].lower()
    assert history1[1]["role"] == "assistant"
    assert history1[1]["content"] == "Ответ один."

    history2, _status2, thread2 = process_exchange(wav, history1, thread)
    assert len(history2) == 4, "prior exchanges must stay visible"
    assert thread2 == thread, "one session must keep one checkpointer thread"


def test_toggle_flips_state_and_label():
    state, label, status = toggle_recording(False)
    assert (state, label) == (True, app.TOGGLE_STOP)
    assert "Recording" in status
    state, label, _ = toggle_recording(True)
    assert (state, label) == (False, app.TOGGLE_START)


def test_blocks_builds():
    demo = app.build_block()
    assert demo is not None
