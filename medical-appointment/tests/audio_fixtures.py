"""Synthetic audio for tests: a WAV built in memory, no codec dependency."""

import base64
import io
import wave

import numpy as np


def wav_bytes(duration_s: float, sample_rate: int = 16000, freq_hz: float = 440.0) -> bytes:
    n = int(round(duration_s * sample_rate))
    t = np.arange(n) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((samples * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def wav_base64(duration_s: float, sample_rate: int = 16000) -> str:
    return base64.b64encode(wav_bytes(duration_s, sample_rate)).decode('ascii')
