"""Domain models shared across pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Segment:
    """A transcribed chunk of speech.

    Carries the text plus minimal metadata for downstream filtering (dedup),
    display (timestamps), and future translation.

    Fields:
        text: Transcribed text. May be empty for pure-silence segments.
        start_ms: Start time in milliseconds (relative to capture start).
        end_ms: End time in milliseconds.
        confidence: Average normalized log probability in [0, 1]. Lower = less
            confident. Used to filter weak transcriptions.
        language: Detected or forced language code (e.g. "en", "pt").
    """

    text: str
    start_ms: int = 0
    end_ms: int = 0
    confidence: float = 1.0
    language: str = "en"

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


__all__ = ["Segment"]
