"""Voice Activity Detection (VAD) — RMS energy gate.

Lightweight first-pass filter: any chunk whose RMS energy exceeds a threshold
is considered speech and passed through to Whisper. Silences are dropped
before reaching the (relatively expensive) transcription stage.

This is intentionally simple. It catches:
  - Long pauses (people typing, thinking, music between sections)
  - Empty audio (no system sound playing)

It does NOT distinguish speech from music/noise. For v2 we can swap in
Silero VAD (5 ONNX, ~2MB, much smarter) behind the same VADFilter API.
"""
from __future__ import annotations

import numpy as np


def rms_energy(samples: np.ndarray) -> float:
    """Root Mean Square energy of an audio buffer.

    For a float32 signal in [-1, 1], RMS returns a value in [0, 1] where
    1.0 means full-scale sine wave. Silence → 0.
    """
    if samples.size == 0:
        return 0.0
    # float32 arithmetic in numpy is sufficient; no need to upcast.
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


class VADFilter:
    """RMS-energy-based voice activity detector.

    Args:
        energy_threshold: Minimum RMS to count as speech. Default 0.01
            works well for typical laptop speaker volume. Tune via CLI in v2.
        min_chunk_samples: Ignore chunks smaller than this (defensive).
    """

    DEFAULT_THRESHOLD = 0.01

    def __init__(
        self,
        energy_threshold: float = DEFAULT_THRESHOLD,
        min_chunk_samples: int = 1600,  # 100ms @ 16kHz
    ) -> None:
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError(f"energy_threshold must be in (0, 1]; got {energy_threshold}")
        if min_chunk_samples <= 0:
            raise ValueError(f"min_chunk_samples must be > 0; got {min_chunk_samples}")
        self._threshold = energy_threshold
        self._min_samples = min_chunk_samples

    @property
    def threshold(self) -> float:
        return self._threshold

    def is_speech(self, audio: np.ndarray) -> bool:
        """Return True if `audio` contains enough energy to be speech.

        Side effect: pure zero arrays short-circuit without computing RMS.
        """
        if audio.size < self._min_samples:
            return False
        return rms_energy(audio) >= self._threshold
