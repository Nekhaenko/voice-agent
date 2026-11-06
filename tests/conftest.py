import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# All graph/config paths require an API key; tests never hit the real
# endpoint (the LLM is faked), so a placeholder satisfies resolution.
os.environ.setdefault("ORCAROUTER_API_KEY", "test-key-for-pytest")
os.environ.setdefault("ORCAROUTER_BASE_URL", "http://localhost:9/v1")
os.environ.setdefault("VOSK_MODEL_PATH", "models/vosk-model-small-ru-0.22")


@pytest.fixture()
def stereo_44k1_wav_bytes() -> bytes:
    """A 0.4 s stereo 44.1 kHz PCM-16 WAV mimicking a browser recording."""
    duration = 0.4
    rate = 44100
    t = np.linspace(0.0, duration, int(rate * duration), endpoint=False)
    left = (np.sin(2 * np.pi * 220.0 * t) * 12000).astype(np.int16)
    right = (np.sin(2 * np.pi * 330.0 * t) * 12000).astype(np.int16)
    interleaved = np.stack([left, right], axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, interleaved, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@pytest.fixture()
def mono_16k_wav_bytes() -> bytes:
    """A 0.4 s silence WAV already at the ASR target format."""
    duration = 0.4
    rate = 16000
    silent = np.zeros(int(rate * duration), dtype=np.int16)
    buffer = io.BytesIO()
    sf.write(buffer, silent, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
