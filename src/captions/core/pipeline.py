"""Pipeline orchestration: ties together capture → VAD → Whisper → dedup.

Stage composition:
    [capture thread] → chunks_queue → [whisper worker] → dedup → display_queue

Design choices:
  - Use threading (not asyncio) because Whisper is a blocking C-extension call;
    async would only add overhead.
  - Single Whisper worker thread is sufficient at ~2s/chunk latency.
    Whisper is the bottleneck; CPU/GPU-bound. No point parallelizing.
  - All inter-stage queues are bounded (maxsize=10) — backpressure drops oldest.
  - `run()` blocks until stop() is called or timeout expires.
"""
from __future__ import annotations

import contextlib
import logging
import queue
import threading
from types import TracebackType
from typing import Any

import numpy as np

from captions.audio.vad import VADFilter
from captions.core.segment import Segment
from captions.stt.dedup import DedupBuffer
from captions.stt.whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates audio capture through to display-ready segments.

    All collaborators are injected so tests can mock each stage independently.

    Args:
        capture: Any object with start(queue) / stop() / _last_error.
        vad: Any object with is_speech(ndarray) -> bool.
        whisper: Any object with transcribe(ndarray, language) -> Segment
                 and optional warmup().
        dedup: Any object with add(Segment) -> Segment | None.
    """

    def __init__(
        self,
        capture: Any,
        vad: VADFilter,
        whisper: WhisperEngine,
        dedup: DedupBuffer,
        language: str = "en",
    ) -> None:
        self._capture = capture
        self._vad = vad
        self._whisper = whisper
        self._dedup = dedup
        self._language = language

        # Bounded queues: backpressure drops oldest if downstream is slow.
        self._chunks_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=10)
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

    # --- lifecycle -------------------------------------------------------

    def run(self, display_queue: queue.Queue[Segment], timeout: float | None = None) -> None:
        """Start capture + worker, block until stop() or timeout.

        Args:
            display_queue: Final destination — UI / logger consumes from here.
            timeout: If given, exit after this many seconds. None = block forever.
        """
        if hasattr(self._whisper, "warmup"):
            try:
                self._whisper.warmup()
            except Exception:
                logger.exception("warmup failed; continuing anyway")

        self._capture.start(self._chunks_q)

        self._worker = threading.Thread(
            target=self._worker_loop,
            args=(display_queue,),
            name="Pipeline-worker",
            daemon=True,
        )
        self._worker.start()

        try:
            self._stop_event.wait(timeout=timeout)
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop capture, worker, and wait for clean shutdown."""
        self._stop_event.set()
        try:
            self._capture.stop()
        except Exception:
            logger.exception("capture.stop() failed")
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    # --- internals -------------------------------------------------------

    def _worker_loop(self, display_queue: queue.Queue[Segment]) -> None:
        """Consume chunks → VAD → Whisper → dedup → push to display."""
        while not self._stop_event.is_set():
            try:
                audio = self._chunks_q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                segment = self._process_chunk(audio)
            except Exception:
                logger.exception("chunk processing failed; skipping")
                continue

            if segment is None:
                continue

            try:
                display_queue.put_nowait(segment)
            except queue.Full:
                # Drop oldest, prefer recency
                with contextlib.suppress(queue.Empty):
                    display_queue.get_nowait()
                with contextlib.suppress(queue.Full):
                    display_queue.put_nowait(segment)

    def _process_chunk(self, audio: np.ndarray) -> Segment | None:
        """Run one chunk through the full pipeline. Returns displayable segment or None."""
        if not self._vad.is_speech(audio):
            return None

        try:
            raw = self._whisper.transcribe(audio, language=self._language)
        except Exception:
            logger.exception("whisper.transcribe() failed; dropping chunk")
            return None

        if raw.is_empty:
            return None

        return self._dedup.add(raw)

    # context-manager sugar
    def __enter__(self) -> Pipeline:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


__all__ = ["Pipeline"]
