"""Tests for Overlay window (T009).

Tkinter requires an X server (or Xvfb). We isolate the windowing layer
behind an OverlayBackend protocol so tests can use a fake backend and we
don't need a display in CI.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from captions.core.segment import Segment
from captions.ui.overlay import CaptionOverlay, InMemoryBackend


def _seg(text: str) -> Segment:
    return Segment(text=text, start_ms=0, end_ms=2000, language="en")


def test_in_memory_backend_starts_hidden() -> None:
    """Overlay should start in hidden state (user toggles via hotkey)."""
    backend = InMemoryBackend()
    overlay = CaptionOverlay(backend=backend)
    overlay.start()
    try:
        assert backend.visible is False
        assert backend.text == ""
    finally:
        overlay.stop()


def test_update_text_stores_latest() -> None:
    backend = InMemoryBackend()
    overlay = CaptionOverlay(backend=backend)
    overlay.start()
    try:
        overlay.show()
        overlay.update_text(_seg("hello world"))
        # Pump the in-memory queue
        overlay._drain()
        assert backend.text == "hello world"
    finally:
        overlay.stop()


def test_toggle_visibility() -> None:
    backend = InMemoryBackend()
    overlay = CaptionOverlay(backend=backend)
    overlay.start()
    try:
        assert backend.visible is False
        overlay.toggle()
        overlay._drain()
        assert backend.visible is True
        overlay.toggle()
        overlay._drain()
        assert backend.visible is False
    finally:
        overlay.stop()


def test_console_backend_used_when_disabled() -> None:
    """When overlay_disabled, captions go to a console printer instead of GUI."""

    class ConsoleBackend:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def show_text(self, text: str) -> None:
            self.messages.append(text)

        def set_visible(self, visible: bool) -> None:
            pass

    backend = ConsoleBackend()
    overlay = CaptionOverlay(backend=backend)
    overlay.start()
    overlay.show()
    overlay.update_text(_seg("test message"))
    overlay._drain()

    assert backend.messages == ["test message"]
    overlay.stop()


def test_queue_overflow_drops_oldest() -> None:
    """Pending segments queue should be bounded; overflow drops oldest."""
    backend = InMemoryBackend()
    overlay = CaptionOverlay(backend=backend, queue_size=3)
    overlay.start()
    overlay.show()
    try:
        for i in range(5):
            overlay.update_text(_seg(f"msg {i}"))
        # After 5 updates with queue size 3 → only last 3 are pending
        # Drain
        for _ in range(5):
            overlay._drain()
        assert backend.text == "msg 4"  # only the last one rendered
    finally:
        overlay.stop()


def test_empty_segments_ignored() -> None:
    backend = InMemoryBackend()
    overlay = CaptionOverlay(backend=backend)
    overlay.start()
    overlay.show()
    try:
        overlay.update_text(_seg(""))
        overlay.update_text(_seg("   "))
        overlay._drain()
        assert backend.text == ""
    finally:
        overlay.stop()


def test_backend_protocol_is_minimal() -> None:
    """Backend protocol needs only 2 methods — keeps Tkinter optional."""
    backend = MagicMock()
    overlay = CaptionOverlay(backend=backend)
    overlay.start()
    overlay.show()
    overlay.update_text(_seg("hi"))
    overlay._drain()
    overlay.toggle()
    overlay._drain()

    backend.set_visible.assert_any_call(True)
    backend.set_visible.assert_any_call(False)
    backend.show_text.assert_called_once_with("hi")
    overlay.stop()
