"""Tests for Dedup buffer (T006)."""
from __future__ import annotations

import pytest

from captions.core.segment import Segment
from captions.stt.dedup import DedupBuffer


def _seg(text: str, conf: float = 1.0) -> Segment:
    return Segment(text=text, start_ms=0, end_ms=2000, confidence=conf, language="en")


def test_first_segment_accepted() -> None:
    buf = DedupBuffer()
    result = buf.add(_seg("hello world"))
    assert result is not None
    assert result.text == "hello world"


def test_exact_duplicate_rejected() -> None:
    buf = DedupBuffer()
    buf.add(_seg("hello world"))
    result = buf.add(_seg("hello world"))
    assert result is None


def test_near_duplicate_rejected() -> None:
    """Tiny wording differences still count as duplicates (similarity > threshold)."""
    buf = DedupBuffer(similarity_threshold=0.7)
    buf.add(_seg("the quick brown fox"))
    result = buf.add(_seg("the quick brown fox jumps"))
    # "fox" + " jumps" is a small suffix → still considered duplicate of the same utterance
    assert result is None


def test_overlap_extracts_suffix() -> None:
    """When new text mostly extends previous, return only the new suffix."""
    buf = DedupBuffer(similarity_threshold=0.7)
    buf.add(_seg("hello everyone today we will learn about"))
    # New chunk: previous text + " neural networks"
    result = buf.add(_seg("hello everyone today we will learn about neural networks"))
    assert result is not None
    assert "neural networks" in result.text
    # Should NOT echo the whole thing
    assert "hello everyone" not in result.text


def test_completely_new_segment_accepted_unchanged() -> None:
    buf = DedupBuffer()
    buf.add(_seg("hello world"))
    result = buf.add(_seg("goodbye sky"))
    assert result is not None
    assert result.text == "goodbye sky"


def test_history_is_bounded() -> None:
    """Buffer should not grow unbounded — only keep last N segments."""
    buf = DedupBuffer(max_history=3)
    buf.add(_seg("one"))
    buf.add(_seg("two"))
    buf.add(_seg("three"))
    buf.add(_seg("four"))
    # 'one' should be evicted; first-of-its-kind still works
    result = buf.add(_seg("one"))
    assert result is not None
    assert result.text == "one"


def test_empty_segment_always_rejected() -> None:
    """Empty transcriptions (silence, errors) should never be returned."""
    buf = DedupBuffer()
    assert buf.add(_seg("")) is None
    assert buf.add(_seg("   ")) is None


def test_reset_clears_history() -> None:
    buf = DedupBuffer()
    buf.add(_seg("hello"))
    buf.reset()
    # After reset, same text should be accepted again
    result = buf.add(_seg("hello"))
    assert result is not None


def test_confidence_propagated() -> None:
    """Returned Segment should keep original metadata."""
    buf = DedupBuffer()
    result = buf.add(_seg("hi", conf=0.85))
    assert result is not None
    assert result.confidence == pytest.approx(0.85)


def test_case_insensitive_dedup() -> None:
    """Hello vs HELLO should still be a duplicate."""
    buf = DedupBuffer(similarity_threshold=0.7)
    buf.add(_seg("Hello World"))
    result = buf.add(_seg("hello world"))
    assert result is None


def test_punctuation_normalized() -> None:
    """Trailing periods and commas shouldn't break dedup."""
    buf = DedupBuffer(similarity_threshold=0.7)
    buf.add(_seg("hello world"))
    result = buf.add(_seg("hello world."))
    assert result is None
