"""Tests for VAD filter (T004)."""
from __future__ import annotations

import numpy as np
import pytest

from captions.audio.vad import VADFilter, rms_energy


def test_silence_chunk_returns_no_speech() -> None:
    """All-zero audio should be classified as silence."""
    vad = VADFilter(energy_threshold=0.01)
    silence = np.zeros(32000, dtype=np.float32)
    assert vad.is_speech(silence) is False


def test_sine_wave_chunk_returns_speech() -> None:
    """A loud sine wave should be detected as speech."""
    vad = VADFilter(energy_threshold=0.01)
    t = np.arange(32000, dtype=np.float32) / 16000.0
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    assert vad.is_speech(tone) is True


def test_quiet_noise_below_threshold() -> None:
    """Noise below threshold should not be speech."""
    vad = VADFilter(energy_threshold=0.05)
    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(32000) * 0.001).astype(np.float32)
    assert vad.is_speech(noise) is False


def test_loud_noise_above_threshold() -> None:
    """Noise above threshold should be classified as speech (false positive OK at this stage)."""
    vad = VADFilter(energy_threshold=0.05)
    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(32000) * 0.3).astype(np.float32)
    assert vad.is_speech(noise) is True


def test_rms_energy_zero_for_silence() -> None:
    silence = np.zeros(100, dtype=np.float32)
    assert rms_energy(silence) == 0.0


def test_rms_energy_matches_known_signal() -> None:
    """Constant signal of amplitude A should have RMS = |A|."""
    signal = np.full(1000, 0.5, dtype=np.float32)
    assert rms_energy(signal) == pytest.approx(0.5, abs=1e-6)


def test_threshold_is_configurable() -> None:
    """Different thresholds should produce different classifications on the same audio."""
    rng = np.random.default_rng(1)
    audio = (rng.standard_normal(32000) * 0.1).astype(np.float32)

    vad_low = VADFilter(energy_threshold=0.001)
    vad_high = VADFilter(energy_threshold=1.0)

    assert vad_low.is_speech(audio) is True
    assert vad_high.is_speech(audio) is False


def test_short_audio_handled_gracefully() -> None:
    """VAD should work on chunks shorter than the typical 2s (defensive)."""
    vad = VADFilter(energy_threshold=0.01)
    short = np.zeros(100, dtype=np.float32)
    assert vad.is_speech(short) is False


def test_negative_amplitudes_counted() -> None:
    """RMS should be positive regardless of sign (uses square)."""
    vad = VADFilter(energy_threshold=0.01)
    t = np.arange(32000, dtype=np.float32) / 16000.0
    wave = -0.2 * np.sin(2 * np.pi * 300.0 * t).astype(np.float32)
    assert vad.is_speech(wave) is True
