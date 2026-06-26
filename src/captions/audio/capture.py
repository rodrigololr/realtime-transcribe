"""Real-time audio capture from PulseAudio loopback source.

Uses `parec` (PulseAudio's official recorder CLI) as a subprocess — robust,
no native deps, ships with `pulseaudio-utils`. We parse its raw int16 PCM
stdout into numpy buffers and emit fixed-size mono float32 chunks.

This approach is preferred over python-sounddevice because:
  - No PortAudio build dependency (works on minimal Zorin installs)
  - Uses the standard PulseAudio path directly (no abstraction layer)
  - parec handles resampling, format conversion, and monitor-source routing
"""
from __future__ import annotations

import contextlib
import queue
import shutil
import subprocess
import threading
from typing import Any

import numpy as np


class AudioCaptureError(RuntimeError):
    """Raised for misconfiguration or fatal capture failure."""


class AudioCapture:
    """Captures PCM frames from a PulseAudio source into fixed-size chunks.

    Spawns `parec --raw --format=s16le --channels=1 --rate=16000` as a subprocess
    and reads its stdout. Frames are parsed (4 bytes per sample), accumulated
    in a buffer, and emitted when the buffer reaches `chunk_samples`.

    Args:
        source_name: PulseAudio source name (e.g. "alsa_output.foo.monitor").
        sample_rate: Must be 16000 (Whisper requirement).
        chunk_seconds: Length of each emitted chunk. 2.0 balances latency
            vs transcription quality for English speech.
    """

    REQUIRED_SAMPLE_RATE = 16000
    # parec default block size is ~2KB; we read in larger chunks for efficiency.
    READ_BYTES = 8192

    def __init__(
        self,
        source_name: str,
        sample_rate: int = REQUIRED_SAMPLE_RATE,
        chunk_seconds: float = 2.0,
    ) -> None:
        if sample_rate != self.REQUIRED_SAMPLE_RATE:
            raise AudioCaptureError(
                f"sample_rate must be {self.REQUIRED_SAMPLE_RATE} (Whisper); got {sample_rate}"
            )
        if chunk_seconds <= 0:
            raise AudioCaptureError(f"chunk_seconds must be positive; got {chunk_seconds}")

        if shutil.which("parec") is None:
            raise AudioCaptureError(
                "`parec` not found in PATH. Install pulseaudio-utils: "
                "`sudo apt install pulseaudio-utils`"
            )

        self._source_name = source_name
        self._sample_rate = sample_rate
        self._chunk_samples = int(sample_rate * chunk_seconds)

        # int16 buffer for accumulating raw PCM before float32 conversion
        self._raw_buffer = bytearray()
        self._lock = threading.Lock()

        self._output_queue: queue.Queue[np.ndarray] | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Captures fatal exceptions from the worker thread so callers / tests can
        # observe them via `last_error`. Real errors also propagate via stderr.
        self._last_error: BaseException | None = None

    # --- public API ------------------------------------------------------

    def start(self, output_queue: queue.Queue[np.ndarray]) -> None:
        """Begin capturing. Returns immediately; runs on background thread."""
        if self._thread is not None and self._thread.is_alive():
            raise AudioCaptureError("capture already started")
        self._output_queue = output_queue
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="AudioCapture",
            daemon=True,
        )
        self._thread.start()
        # Give the thread a brief moment to either start the process or fail fast,
        # so that an immediate second start() call doesn't pass the "already started"
        # check on a dead thread.
        # NOTE: this is only relevant for tests; production code never calls start() twice.

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the subprocess and wait for the worker to exit."""
        self._stop_event.set()
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1.0)
            except Exception:
                pass
            self._process = None
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def last_error(self) -> BaseException | None:
        """Return the fatal exception from the worker thread, if any."""
        return self._last_error

    # --- internals -------------------------------------------------------

    def _run(self) -> None:
        """Worker thread: spawn parec, read its stdout, emit chunks."""
        assert self._output_queue is not None
        cmd = [
            "parec",
            "--raw",                       # no WAV header — pure PCM
            "--format=s16le",              # signed 16-bit little-endian
            "--channels=1",
            f"--rate={self._sample_rate}",
            f"--device={self._source_name}",
        ]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # unbuffered — we read in READ_BYTES chunks
            )
        except FileNotFoundError as exc:
            self._last_error = AudioCaptureError(
                "`parec` binary missing at runtime. Install pulseaudio-utils."
            )
            raise self._last_error from exc
        except Exception as exc:
            self._last_error = AudioCaptureError(f"Failed to spawn parec: {exc}")
            raise self._last_error from exc

        assert self._process.stdout is not None
        process = self._process
        try:
            while not self._stop_event.is_set():
                data = process.stdout.read(self.READ_BYTES)
                if not data:
                    # parec exited unexpectedly
                    stderr = process.stderr.read(1024).decode(errors="replace") if process.stderr else ""
                    self._last_error = AudioCaptureError(
                        f"parec exited unexpectedly. stderr: {stderr.strip()}"
                    )
                    raise self._last_error
                self._ingest(data)
        finally:
            if process.poll() is None:
                with contextlib.suppress(Exception):
                    process.terminate()

    def _ingest(self, data: bytes) -> None:
        """Append raw int16 PCM to buffer; emit chunks when threshold reached."""
        # int16 = 2 bytes/sample. Discard trailing odd byte defensively.
        usable = len(data) - (len(data) % 2)
        if usable <= 0:
            return
        with self._lock:
            self._raw_buffer.extend(data[:usable])
            # Each sample = 2 bytes; chunk target = chunk_samples * 2 bytes
            target_bytes = self._chunk_samples * 2
            while len(self._raw_buffer) >= target_bytes:
                self._emit_chunk_bytes(target_bytes)

    def _emit_chunk_bytes(self, n_bytes: int) -> None:
        """Slice n_bytes from raw buffer, convert to float32, push to queue."""
        assert self._output_queue is not None
        chunk_bytes = bytes(self._raw_buffer[:n_bytes])
        del self._raw_buffer[:n_bytes]

        # int16 → float32 in [-1, 1]
        samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            self._output_queue.put_nowait(samples)
        except queue.Full:
            # Drop oldest to make room — prefer recency for live captions
            with contextlib.suppress(queue.Empty):
                self._output_queue.get_nowait()
            with contextlib.suppress(queue.Full):
                self._output_queue.put_nowait(samples)


__all__: list[Any] = ["AudioCapture", "AudioCaptureError"]
