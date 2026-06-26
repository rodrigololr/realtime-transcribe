"""Fuzzy duplicate detection for transcribed segments.

Whisper chunks overlap, so the same utterance often comes back nearly identical
2-3 times before the speaker moves on. This buffer:

  - Rejects exact or near-exact duplicates (`difflib.SequenceMatcher` ratio).
  - For partial overlaps, extracts only the new suffix (avoids echo).
  - Bounds history so memory stays constant in long meetings.

The algorithm is intentionally simple and pure-stdlib — no ML.
"""
from __future__ import annotations

from collections import deque
from difflib import SequenceMatcher

from captions.core.segment import Segment


def _normalize(text: str) -> str:
    """Lowercase + strip surrounding punctuation/whitespace for comparison."""
    return text.lower().strip().strip(".,!?;:\"'()[]{}")


class DedupBuffer:
    """Decides whether a new segment is duplicate, partial-overlap, or fresh.

    Args:
        max_history: How many recent segments to remember for comparison.
        similarity_threshold: Above this ratio (0-1), treat as duplicate.
        suffix_min_chars: Minimum length of an extracted suffix to be returned.
            Prevents emitting tiny fragments like "es" or "the".
    """

    def __init__(
        self,
        max_history: int = 5,
        similarity_threshold: float = 0.75,
        suffix_min_chars: int = 8,
    ) -> None:
        if max_history <= 0:
            raise ValueError(f"max_history must be > 0; got {max_history}")
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be in (0, 1]; got {similarity_threshold}")
        if suffix_min_chars < 0:
            raise ValueError(f"suffix_min_chars must be >= 0; got {suffix_min_chars}")

        self._history: deque[Segment] = deque(maxlen=max_history)
        self._threshold = similarity_threshold
        self._suffix_min_chars = suffix_min_chars

    def reset(self) -> None:
        """Clear all remembered segments (e.g., on user hotkey)."""
        self._history.clear()

    def add(self, segment: Segment) -> Segment | None:
        """Decide what to do with `segment`.

        Returns:
          - None if the segment is empty or a duplicate (should be dropped).
          - A new Segment with only the fresh suffix if it partially overlaps.
          - The input segment unchanged if it's genuinely new.
        """
        text = segment.text.strip()
        if not text:
            return None

        normalized = _normalize(text)
        if not normalized:
            return None

        # Compare against all historical segments.
        for prev in self._history:
            prev_norm = _normalize(prev.text)
            ratio = SequenceMatcher(None, normalized, prev_norm).ratio()

            if ratio >= self._threshold:
                # Near-duplicate. Check if there's a meaningful suffix.
                suffix = self._extract_suffix(normalized, prev_norm)
                if suffix and len(suffix) >= self._suffix_min_chars:
                    # Store the new, longer version and return just the suffix.
                    self._history.append(segment)
                    return Segment(
                        text=suffix,
                        start_ms=segment.start_ms,
                        end_ms=segment.end_ms,
                        confidence=segment.confidence,
                        language=segment.language,
                    )
                # Otherwise it's a true duplicate → drop.
                self._history.append(segment)  # keep newest version
                return None

        # No match → fresh segment.
        self._history.append(segment)
        return segment

    @staticmethod
    def _extract_suffix(new_norm: str, prev_norm: str) -> str:
        """If `new_norm` extends `prev_norm`, return just the tail.

        Returns empty string if no clear extension relationship.
        """
        if not prev_norm or not new_norm:
            return ""

        # Common prefix length
        i = 0
        n = min(len(new_norm), len(prev_norm))
        while i < n and new_norm[i] == prev_norm[i]:
            i += 1

        # If new_norm is entirely contained in prev_norm → no suffix
        if i >= len(new_norm):
            return ""

        suffix = new_norm[i:].strip()
        # Heuristic: suffix must be substantial relative to previous text
        # to avoid spurious matches like "ok" after "ok let's go".
        if len(suffix) < max(3, len(prev_norm) * 0.15):
            return ""
        return suffix


__all__ = ["DedupBuffer"]
