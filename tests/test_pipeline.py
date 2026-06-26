"""Tests for Pipeline orchestration (T007)."""
from __future__ import annotations

import threading
import time
from queue import Queue
from unittest.mock import MagicMock

import numpy as np

from captions.core.pipeline import Pipeline
from captions.core.segment import Segment

SAMPLE_RATE = 16000


def _chunk(seconds: float = 2.0, value: float = 0.0) -> np.ndarray:
    return np.full(int(SAMPLE_RATE * seconds), value, dtype=np.float32)


def _make_mock_capture(chunks: list[np.ndarray]) -> MagicMock:
    """Mock AudioCapture that emits the given chunks when start() is called."""
    capture = MagicMock()
    capture._last_error = None
    q_holder: list[Queue] = []

    def fake_start(out_q: Queue) -> None:
        q_holder.append(out_q)
        for c in chunks:
            out_q.put(c)

    capture.start.side_effect = fake_start
    capture.stop = MagicMock()
    return capture


def _make_mock_whisper(text_per_chunk: list[str]) -> MagicMock:
    """Mock WhisperEngine that returns predetermined text per call."""
    engine = MagicMock()
    iter_texts = iter(text_per_chunk)

    def fake_transcribe(audio, language="en"):
        try:
            return Segment(text=next(iter_texts), start_ms=0, end_ms=2000, language=language)
        except StopIteration:
            return Segment(text="", start_ms=0, end_ms=2000, language=language)

    engine.transcribe.side_effect = fake_transcribe
    engine.warmup = MagicMock()
    return engine


def test_pipeline_runs_all_stages() -> None:
    """A chunk pushed by capture → VAD passes → Whisper → dedup → display queue."""
    capture = _make_mock_capture([_chunk(value=0.3)])  # loud enough for VAD
    vad = MagicMock()
    vad.is_speech.return_value = True
    whisper = _make_mock_whisper(["hello world"])
    dedup = MagicMock()
    dedup.add.side_effect = lambda s: s  # accept all

    display: Queue = Queue()

    pipe = Pipeline(capture=capture, vad=vad, whisper=whisper, dedup=dedup)
    pipe.run(display, timeout=0.5)

    assert display.qsize() == 1
    seg = display.get_nowait()
    assert seg.text == "hello world"


def test_pipeline_drops_silence_via_vad() -> None:
    """VAD returns False → chunk skipped, never reaches Whisper."""
    capture = _make_mock_capture([_chunk(value=0.0)])
    vad = MagicMock()
    vad.is_speech.return_value = False
    whisper = _make_mock_whisper(["should not appear"])
    dedup = MagicMock()
    dedup.add.side_effect = lambda s: s

    display: Queue = Queue()
    pipe = Pipeline(capture=capture, vad=vad, whisper=whisper, dedup=dedup)
    pipe.run(display, timeout=0.3)

    assert display.qsize() == 0
    whisper.transcribe.assert_not_called()


def test_pipeline_drops_duplicates_via_dedup() -> None:
    """Dedup returns None → segment not shown."""
    capture = _make_mock_capture([_chunk(value=0.3), _chunk(value=0.3)])
    vad = MagicMock()
    vad.is_speech.return_value = True
    whisper = _make_mock_whisper(["hello", "hello"])
    dedup = MagicMock()
    dedup.add.side_effect = [Segment(text="hello", language="en"), None]

    display: Queue = Queue()
    pipe = Pipeline(capture=capture, vad=vad, whisper=whisper, dedup=dedup)
    pipe.run(display, timeout=0.5)

    assert display.qsize() == 1


def test_pipeline_shuts_down_cleanly() -> None:
    """stop() should terminate the pipeline within timeout."""
    capture = _make_mock_capture([])
    vad = MagicMock()
    vad.is_speech.return_value = True
    whisper = _make_mock_whisper([])
    dedup = MagicMock()

    pipe = Pipeline(capture=capture, vad=vad, whisper=whisper, dedup=dedup)

    t = threading.Thread(target=lambda: pipe.run(Queue(), timeout=2.0), daemon=True)
    t.start()
    time.sleep(0.1)
    pipe.stop()
    t.join(timeout=2.0)
    assert not t.is_alive(), "pipeline thread should exit cleanly"
    capture.stop.assert_called()


def test_pipeline_whisper_failure_does_not_crash() -> None:
    """If Whisper raises, pipeline should log and continue."""
    capture = _make_mock_capture([_chunk(value=0.3), _chunk(value=0.3)])
    vad = MagicMock()
    vad.is_speech.return_value = True
    whisper = MagicMock()
    whisper.transcribe.side_effect = [
        RuntimeError("GPU OOM"),
        Segment(text="recovered", language="en"),
    ]
    whisper.warmup = MagicMock()
    dedup = MagicMock()
    dedup.add.side_effect = lambda s: s

    display: Queue = Queue()
    pipe = Pipeline(capture=capture, vad=vad, whisper=whisper, dedup=dedup)
    pipe.run(display, timeout=0.5)

    assert display.qsize() == 1
    assert display.get_nowait().text == "recovered"
