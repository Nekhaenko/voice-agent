"""Local speech recognition: WAV conversion + batch VOSK transcription.

Design D2: decode any WAV the browser produced via soundfile, convert to
16 kHz mono 16-bit PCM, and transcribe in one pass with VOSK (offline).

ASREmptyResult distinguishes "decoded fine, nothing recognized" from
errors: the interface must not send a non-recognition to the agent
(specs/voice-assistant, scenario "Empty recording").
"""

from __future__ import annotations

import io

import soundfile as sf

TARGET_RATE = 16000
TARGET_CHANNELS = 1


class ASREmptyResult(Exception):
    """Audio decoded successfully but VOSK produced no recognized text."""


def get_vosk_model(model_path: str):
    """Load the local VOSK model, explaining a missing install clearly."""
    import os

    from vosk import Model

    if not os.path.isdir(model_path):
        raise RuntimeError(
            f"VOSK model not found at '{model_path}'. Download the Russian "
            "model (vosk-model-small-ru-0.22) and set VOSK_MODEL_PATH."
        )
    return Model(model_path)


def _to_16k_mono_pcm(wav_bytes: bytes) -> bytes:
    """Decode browser WAV bytes and resample/remix to 16 kHz mono PCM."""
    import numpy as np

    data, samplerate = sf.read(
        io.BytesIO(wav_bytes), dtype="int16", always_2d=True
    )

    # Mix down any channel count to mono.
    mono = data.mean(axis=1).astype(np.int16)

    if samplerate == TARGET_RATE:
        converted = mono
    else:
        # Naive but adequate resampling by linear interpolation: the browser
        # clips are short utterances; quality impact is negligible for ASR.
        source = mono.astype(np.float64)
        length = max(1, int(round(source.size * TARGET_RATE / samplerate)))
        converted = (
            np.interp(
                np.linspace(0.0, source.size - 1, length),
                np.arange(source.size),
                source,
            )
            .round()
            .astype(np.int16)
        )

    return converted.tobytes()


def transcribe(wav_bytes: bytes, model_path: str) -> str:
    """Transcribe one recorded utterance (Russian) with the local VOSK model.

    Raises ASREmptyResult when the clip contains no recognizable speech.
    """
    import json

    from vosk import KaldiRecognizer

    pcm = _to_16k_mono_pcm(wav_bytes)
    recognizer = KaldiRecognizer(get_vosk_model(model_path), TARGET_RATE)

    # Batch: feed the whole utterance, then take the final result.
    recognizer.AcceptWaveform(pcm)
    final = json.loads(recognizer.FinalResult())

    text = (final.get("text") or "").strip()
    if not text:
        raise ASREmptyResult("Nothing recognized in the recording.")

    return text
