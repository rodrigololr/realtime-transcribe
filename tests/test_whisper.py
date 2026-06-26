"""Tests for Whisper engine wrapper (T005)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from captions.core.segment import Segment
from captions.stt.whisper_engine import WhisperEngine, WhisperEngineError


def _silent_chunk(seconds: float = 2.0) -> np.ndarray:
    return np.zeros(int(16000 * seconds), dtype=np.float32)


def _mock_whisper_model(transcript: str = "hello world", no_speech_prob: float = 0.1):
    """Build a mock faster_whisper.WhisperModel."""
    mock_model = MagicMock()
    seg = MagicMock()
    seg.text = transcript
    seg.start = 0.0
    seg.end = 2.0
    seg.avg_logprob = -0.1
    seg.no_speech_prob = no_speech_prob
    seg.compression_ratio = 1.0
    info = MagicMock()
    info.language = "en"
    info.language_probability = 0.99
    mock_model.transcribe.return_value = ([seg], info)
    return mock_model


def test_init_rejects_unknown_model() -> None:
    with pytest.raises(WhisperEngineError, match="model"):
        WhisperEngine(model_size="nonexistent-model")


def test_transcribe_returns_segment() -> None:
    """Transcription should return a typed Segment object."""
    with patch("captions.stt.whisper_engine._import_whisper") as mock_import:
        mock_import.return_value = MagicMock(WhisperModel=MagicMock(return_value=_mock_whisper_model("hello there")))
        engine = WhisperEngine(model_size="tiny.en", device="cpu")
        seg = engine.transcribe(_silent_chunk(), language="en")

    assert isinstance(seg, Segment)
    assert "hello" in seg.text


def test_transcribe_skips_high_no_speech_prob() -> None:
    """If Whisper thinks audio is silence (no_speech_prob > 0.6), engine returns empty Segment."""
    with patch("captions.stt.whisper_engine._import_whisper") as mock_import:
        mock_import.return_value = MagicMock(WhisperModel=MagicMock(return_value=_mock_whisper_model("...", 0.95)))
        engine = WhisperEngine(model_size="tiny.en", device="cpu")
        seg = engine.transcribe(_silent_chunk(), language="en")

    assert seg.text == ""
    assert seg.confidence == 0.0


def test_no_speech_threshold_is_configurable() -> None:
    """A higher threshold should let through results that a lower one rejects."""
    with patch("captions.stt.whisper_engine._import_whisper") as mock_import:
        mock_import.return_value = MagicMock(WhisperModel=MagicMock(return_value=_mock_whisper_model("hi", 0.5)))
        engine = WhisperEngine(model_size="tiny.en", device="cpu", no_speech_threshold=0.9)
        seg = engine.transcribe(_silent_chunk(), language="en")
        assert "hi" in seg.text


def test_transcribe_passes_language_to_model() -> None:
    """The language hint should be forwarded to faster-whisper."""
    model_mock = _mock_whisper_model("bonjour")
    with patch("captions.stt.whisper_engine._import_whisper") as mock_import:
        mock_import.return_value = MagicMock(WhisperModel=MagicMock(return_value=model_mock))
        engine = WhisperEngine(model_size="tiny.en", device="cpu")
        engine.transcribe(_silent_chunk(), language="fr")

    # transcribe called with positional (audio,) and keyword language
    call_kwargs = model_mock.transcribe.call_args.kwargs
    assert call_kwargs.get("language") == "fr"


def test_cuda_fallback_to_cpu_when_unavailable() -> None:
    """When device='auto' and CUDA is missing, engine should run on CPU."""
    model_mock = _mock_whisper_model("ok")
    with patch("captions.stt.whisper_engine._import_whisper") as mock_import, \
         patch("captions.stt.whisper_engine._cuda_available", return_value=False):
        mock_import.return_value = MagicMock(WhisperModel=MagicMock(return_value=model_mock))
        engine = WhisperEngine(model_size="tiny.en", device="auto")
        engine.transcribe(_silent_chunk(), language="en")

    # Check WhisperModel was instantiated with device="cpu"
    model_call = mock_import.return_value.WhisperModel.call_args
    assert model_call.kwargs.get("device") == "cpu"


def test_explicit_cpu_device_used_directly() -> None:
    """device='cpu' should not try to detect CUDA."""
    model_mock = _mock_whisper_model("ok")
    with patch("captions.stt.whisper_engine._import_whisper") as mock_import, \
         patch("captions.stt.whisper_engine._cuda_available") as cuda_check:
        mock_import.return_value = MagicMock(WhisperModel=MagicMock(return_value=model_mock))
        engine = WhisperEngine(model_size="tiny.en", device="cpu")
        engine.transcribe(_silent_chunk(), language="en")
        cuda_check.assert_not_called()


def test_missing_whisper_raises_friendly_error() -> None:
    """If faster-whisper is not installed, error explains how to fix."""
    with (
        patch(
            "captions.stt.whisper_engine._import_whisper",
            side_effect=WhisperEngineError("faster-whisper is not installed. Run: pip install -e .[stt]"),
        ),
        pytest.raises(WhisperEngineError, match="pip install"),
    ):
        WhisperEngine(model_size="tiny.en", device="cpu")
