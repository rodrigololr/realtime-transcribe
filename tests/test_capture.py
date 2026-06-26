"""Tests for AudioCapture (T003)."""
from __future__ import annotations

import time
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from captions.audio.capture import AudioCapture, AudioCaptureError

SAMPLE_RATE = 16000
CHUNK_SAMPLES = SAMPLE_RATE * 2  # 32000


def _pcm_bytes(samples: int, value: float = 0.0) -> bytes:
    """Generate raw int16 PCM bytes (s16le format)."""
    int_val = int(value * 32767)
    return np.full(samples, int_val, dtype=np.int16).tobytes()


def test_init_validates_sample_rate() -> None:
    with pytest.raises(AudioCaptureError, match="16000"):
        AudioCapture(source_name="x.monitor", sample_rate=44100)


def test_init_validates_chunk_size() -> None:
    with pytest.raises(AudioCaptureError, match="positive"):
        AudioCapture(source_name="x.monitor", sample_rate=16000, chunk_seconds=0.0)


def test_init_requires_parec_binary() -> None:
    with patch("captions.audio.capture.shutil.which", return_value=None), pytest.raises(
        AudioCaptureError, match="pulseaudio-utils"
    ):
        AudioCapture(source_name="x.monitor")


def test_start_spawns_parec_with_correct_args() -> None:
    """parec command should include source name, format, channels, rate."""
    out_q: Queue = Queue()
    capture = AudioCapture(source_name="test.monitor", sample_rate=SAMPLE_RATE, chunk_seconds=2.0)

    popen_mock = MagicMock()
    popen_mock.stdout = MagicMock()
    popen_mock.stderr = MagicMock()
    popen_mock.poll.return_value = None  # still running

    # Make stdout.read block briefly to let stop() terminate
    popen_mock.stdout.read.side_effect = [_pcm_bytes(800), b""]

    captured_cmd: list = []

    def fake_popen(cmd, **kwargs):
        captured_cmd.extend(cmd)
        return popen_mock

    with patch("captions.audio.capture.subprocess.Popen", side_effect=fake_popen):
        capture.start(out_q)
        time.sleep(0.1)
        capture.stop()

    assert "parec" in captured_cmd
    assert "--raw" in captured_cmd
    assert "--format=s16le" in captured_cmd
    assert "--channels=1" in captured_cmd
    assert f"--rate={SAMPLE_RATE}" in captured_cmd
    assert "--device=test.monitor" in captured_cmd


def test_emits_float32_chunk_of_correct_size() -> None:
    """After ingesting chunk_samples * 2 bytes, one chunk should appear on queue."""
    out_q: Queue = Queue(maxsize=10)
    capture = AudioCapture(source_name="test.monitor", sample_rate=SAMPLE_RATE, chunk_seconds=2.0)

    # Build a stream that yields 3 small reads then "" (signals end → loop exits)
    # First read: 16000 samples (= 32000 bytes) — below threshold
    # Second read: another 16000 samples → 32000 total samples → emit
    # Third read: empty → exit
    reads = [_pcm_bytes(16000), _pcm_bytes(16000), b""]

    popen_mock = MagicMock()
    popen_mock.stdout = MagicMock()
    popen_mock.stderr = MagicMock()
    popen_mock.poll.return_value = None

    read_iter = iter(reads)

    def fake_read(_n):
        return next(read_iter)

    popen_mock.stdout.read.side_effect = fake_read

    with patch("captions.audio.capture.subprocess.Popen", return_value=popen_mock):
        capture.start(out_q)
        time.sleep(0.2)
        capture.stop()

    assert out_q.qsize() == 1
    chunk = out_q.get_nowait()
    assert chunk.shape == (CHUNK_SAMPLES,)
    assert chunk.dtype == np.float32
    assert -1.0 <= chunk.min() <= chunk.max() <= 1.0


def test_emits_chunks_for_max_value_audio() -> None:
    """Full-scale int16 should map to ~1.0 float32 without clipping."""
    out_q: Queue = Queue()
    capture = AudioCapture(source_name="test.monitor", sample_rate=SAMPLE_RATE, chunk_seconds=2.0)

    popen_mock = MagicMock()
    popen_mock.stdout = MagicMock()
    popen_mock.stderr = MagicMock()
    popen_mock.poll.return_value = None

    reads = [_pcm_bytes(CHUNK_SAMPLES, value=1.0), b""]
    read_iter = iter(reads)
    popen_mock.stdout.read.side_effect = lambda _: next(read_iter)

    with patch("captions.audio.capture.subprocess.Popen", return_value=popen_mock):
        capture.start(out_q)
        time.sleep(0.2)
        capture.stop()

    chunk = out_q.get_nowait()
    assert chunk.max() == pytest.approx(1.0, abs=1e-4)
    assert chunk.min() == pytest.approx(1.0, abs=1e-4)


def test_parec_crash_surfaces_via_last_error() -> None:
    """If parec dies, the fatal exception should be observable via last_error."""
    out_q: Queue = Queue()
    capture = AudioCapture(source_name="test.monitor", sample_rate=SAMPLE_RATE, chunk_seconds=2.0)

    popen_mock = MagicMock()
    popen_mock.stdout = MagicMock()
    popen_mock.stderr = MagicMock()
    popen_mock.poll.return_value = None

    # parec dies on first read → returns b""
    popen_mock.stdout.read.return_value = b""
    popen_mock.stderr.read.return_value = b"Connection refused"

    with patch("captions.audio.capture.subprocess.Popen", return_value=popen_mock):
        capture.start(out_q)
        # Wait for thread to process the empty read and surface error
        time.sleep(0.3)
        capture._thread.join(timeout=1.0)  # type: ignore[union-attr]

    assert capture.last_error is not None
    assert isinstance(capture.last_error, AudioCaptureError)
    assert "Connection refused" in str(capture.last_error)


def test_stop_terminates_parec_process() -> None:
    """stop() should terminate the subprocess."""
    out_q: Queue = Queue()
    capture = AudioCapture(source_name="test.monitor", sample_rate=SAMPLE_RATE, chunk_seconds=2.0)

    popen_mock = MagicMock()
    popen_mock.stdout = MagicMock()
    popen_mock.stderr = MagicMock()
    popen_mock.poll.return_value = None

    # Make read block forever unless stop() terminates process

    def slow_read(_):
        time.sleep(0.1)
        return b""

    popen_mock.stdout.read.side_effect = slow_read

    with patch("captions.audio.capture.subprocess.Popen", return_value=popen_mock):
        capture.start(out_q)
        time.sleep(0.05)
        capture.stop(timeout=2.0)

    popen_mock.terminate.assert_called()
    assert capture._thread is None or not capture._thread.is_alive()


def test_double_start_raises() -> None:
    """Starting an already-running capture must fail."""
    out_q: Queue = Queue()
    capture = AudioCapture(source_name="test.monitor", sample_rate=SAMPLE_RATE)

    popen_mock = MagicMock()
    popen_mock.stdout = MagicMock()
    popen_mock.stderr = MagicMock()
    popen_mock.poll.return_value = None

    # Slow read keeps thread alive long enough for the second start() to fire
    popen_mock.stdout.read.side_effect = lambda _: (time.sleep(0.5), b"")[1]

    with patch("captions.audio.capture.subprocess.Popen", return_value=popen_mock):
        capture.start(out_q)
        time.sleep(0.1)
        with pytest.raises(AudioCaptureError, match="already started"):
            capture.start(out_q)
        capture.stop()
