import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from asr import ASREmptyResult, _to_16k_mono_pcm, transcribe


def test_convert_44k1_stereo_to_16k_mono(stereo_44k1_wav_bytes):
    pcm = _to_16k_mono_pcm(stereo_44k1_wav_bytes)
    # 0.4 s at 44.1 kHz stereo -> ~0.4 s at 16 kHz mono, 2 bytes per sample.
    expected = 0.4 * 16000 * 2
    assert abs(len(pcm) - expected) < 0.05 * expected


def test_converted_pcm_is_16bit_mono_length_matches_duration(
    stereo_44k1_wav_bytes,
):
    pcm = _to_16k_mono_pcm(stereo_44k1_wav_bytes)
    samples = len(pcm) // 2
    assert abs(samples / 16000 - 0.4) < 0.02  # seconds of audio


def test_transcribe_speech_fixture_returns_expected_text():
    from config import settings

    fixture = Path(__file__).parent / "fixtures" / "sample_ru.wav"
    assert fixture.is_file(), "Speech fixture missing; see tasks 2.2 notes"
    text = transcribe(fixture.read_bytes(), settings()["vosk_model_path"])
    # The fixture says "Привет, как дела?" in Russian.
    for word in ("привет", "дела"):
        assert word in text.lower(), f"fixture should contain {word!r}, got {text!r}"


def test_transcribe_silence_raises_empty():
    rate = 16000
    silent = np.zeros(int(1.0 * rate), dtype=np.int16)
    buffer = io.BytesIO()
    sf.write(buffer, silent, rate, format="WAV", subtype="PCM_16")
    from config import settings

    with pytest.raises(ASREmptyResult):
        transcribe(buffer.getvalue(), settings()["vosk_model_path"])
