"""Speech-to-text engine wrapper around `faster-whisper`.

Design notes:
  - Lazy import: faster-whisper + torch are heavy (1GB+). We import them inside
    a function so that `pip install -e .[dev]` works for CI without them, and
    `run.sh` installs them only when actually running the app.
  - Pure-Python wrapper: no global state, easy to mock for tests.
  - CUDA detection at construction time, with graceful CPU fallback.

API surface is intentionally tiny: `transcribe(audio, language) -> Segment`.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from captions.core.segment import Segment

if TYPE_CHECKING:
    pass


class WhisperEngineError(RuntimeError):
    """Raised when the engine cannot be initialized or used."""


# Models officially supported by faster-whisper. User can extend if needed.
VALID_MODELS: frozenset[str] = frozenset(
    {
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
        "distil-large-v3",
        # Aliases commonly used
        "turbo",
    }
)


def _import_whisper() -> Any:
    """Lazy import of faster_whisper with friendly error."""
    try:
        import faster_whisper
        return faster_whisper
    except ImportError as exc:
        raise WhisperEngineError(
            "faster-whisper is not installed. Run: "
            "`pip install -e .[stt]` or use `./run.sh` which installs it."
        ) from exc


def _cuda_available() -> bool:
    """Check if CUDA is actually usable by torch."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


class WhisperEngine:
    """Thin wrapper over faster_whisper.WhisperModel.

    Args:
        model_size: One of VALID_MODELS. Use `tiny.en` for CPU, `small.en`
            for CUDA, `medium.en` for high-accuracy CUDA.
        device: "cuda" | "cpu" | "auto". "auto" picks cuda if available.
        compute_type: "float16" (CUDA), "int8" (CPU), "float32" (precision).
        no_speech_threshold: faster-whisper returns segments whose
            `no_speech_prob` exceeds this are treated as silence.
    """

    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "auto",
        compute_type: str | None = None,
        no_speech_threshold: float = 0.6,
    ) -> None:
        if model_size not in VALID_MODELS:
            raise WhisperEngineError(
                f"Unknown model '{model_size}'. Valid: {sorted(VALID_MODELS)}"
            )
        if device not in {"cuda", "cpu", "auto"}:
            raise WhisperEngineError(f"device must be 'cuda'|'cpu'|'auto'; got '{device}'")

        resolved_device = device
        if device == "auto":
            resolved_device = "cuda" if _cuda_available() else "cpu"
            if resolved_device == "cpu":
                # Inform the user so they're not surprised by slow inference.
                # Use a stderr-friendly print rather than logging to keep deps zero.
                import sys
                print(
                    "[captions] CUDA not available — falling back to CPU. "
                    "Use 'tiny.en' for reasonable speed.",
                    file=sys.stderr,
                )

        if compute_type is None:
            compute_type = "float16" if resolved_device == "cuda" else "int8"

        whisper = _import_whisper()
        try:
            self._model = whisper.WhisperModel(
                model_size,
                device=resolved_device,
                compute_type=compute_type,
            )
        except Exception as exc:
            raise WhisperEngineError(f"Failed to load Whisper model '{model_size}': {exc}") from exc

        self._model_size = model_size
        self._device = resolved_device
        self._no_speech_threshold = no_speech_threshold

    @property
    def model_size(self) -> str:
        return self._model_size

    @property
    def device(self) -> str:
        return self._device

    def warmup(self) -> None:
        """Run a tiny dummy inference to amortize cold-start (CUDA kernel JIT)."""
        try:
            dummy = np.zeros(16000, dtype=np.float32)
            self.transcribe(dummy, language="en")
        except Exception:
            pass  # warmup is best-effort

    def transcribe(self, audio: np.ndarray, language: str = "en") -> Segment:
        """Transcribe one chunk of mono 16kHz float32 audio.

        Returns a Segment. If the chunk is judged as silence/no-speech by
        Whisper, returns an empty Segment (text="") so callers can skip.
        """
        if audio.ndim != 1:
            raise WhisperEngineError(f"audio must be 1-D mono; got shape {audio.shape}")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        try:
            segments_iter, info = self._model.transcribe(
                audio,
                language=language,
                beam_size=1,           # latency > accuracy for live captions
                vad_filter=False,      # we already filter upstream
                condition_on_previous_text=False,  # each chunk is independent
            )
            segments = list(segments_iter)
        except Exception as exc:
            raise WhisperEngineError(f"Transcription failed: {exc}") from exc

        if not segments:
            return Segment(text="", language=language)

        # faster-whisper can return multiple segments per chunk; concatenate.
        # But if any has high no_speech_prob, treat the whole thing as silence.
        max_no_speech = max(s.no_speech_prob for s in segments)
        if max_no_speech > self._no_speech_threshold:
            return Segment(text="", language=language, confidence=0.0)

        text_parts: list[str] = []
        avg_logprobs: list[float] = []
        start_ms = 0
        end_ms = 0
        for seg in segments:
            t = seg.text.strip()
            if t:
                text_parts.append(t)
            if seg.avg_logprob is not None and not math.isnan(seg.avg_logprob):
                avg_logprobs.append(seg.avg_logprob)
            start_ms = int(seg.start * 1000)
            end_ms = int(seg.end * 1000)

        text = " ".join(text_parts).strip()
        # Normalize log prob in [-inf, 0] → confidence in [0, 1] via exp.
        confidence = float(np.exp(np.mean(avg_logprobs))) if avg_logprobs else 0.5

        return Segment(
            text=text,
            start_ms=start_ms,
            end_ms=end_ms,
            confidence=confidence,
            language=getattr(info, "language", language),
        )


__all__ = ["VALID_MODELS", "WhisperEngine", "WhisperEngineError"]
