"""Caption overlay window.

Architecture: backend protocol lets us swap Tkinter (real app) for any
test-friendly backend. The overlay itself just owns a worker thread that
drains a queue of segments and forwards them to the backend.

Tkinter-specific code lives in `TkOverlayBackend`. The default backend when
`--no-overlay` is passed is `ConsoleBackend`.
"""
from __future__ import annotations

import contextlib
import logging
import queue
import threading
from typing import Protocol

from captions.core.segment import Segment

logger = logging.getLogger(__name__)


class OverlayBackend(Protocol):
    """Minimal contract every overlay backend must satisfy."""

    def set_visible(self, visible: bool) -> None:
        ...

    def show_text(self, text: str) -> None:
        ...


class InMemoryBackend:
    """Reference backend for tests. Stores state, no actual UI."""

    def __init__(self) -> None:
        self.visible: bool = False
        self.text: str = ""

    def set_visible(self, visible: bool) -> None:
        self.visible = visible

    def show_text(self, text: str) -> None:
        self.text = text


class ConsoleBackend:
    """Plain stdout backend used when --no-overlay is passed."""

    def __init__(self) -> None:
        self.visible: bool = True

    def set_visible(self, visible: bool) -> None:
        self.visible = visible

    def show_text(self, text: str) -> None:
        print(text, flush=True)


class TkOverlayBackend:
    """Real Tkinter overlay window — always-on-top, transparent background.

    Requires python3-tk at runtime. The window is positioned at the bottom
    of the primary monitor, 80% width, ~140px tall. Background uses a
    translucent color so it composites nicely over videos / browsers.

    Run on main thread; calls from worker thread use `after()` for safety.
    """

    BG_COLOR = "#101010"
    FG_COLOR = "#ffffff"
    FONT_FAMILY = "DejaVu Sans Mono"
    FONT_SIZE = 22
    OPACITY = 0.78  # 0..1 window alpha (works on X11)

    def __init__(self) -> None:
        import tkinter as tk

        self._tk = tk
        self._root = tk.Tk()
        self._root.title("captions")
        self._root.configure(bg=self.BG_COLOR)
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", self.OPACITY)
        self._root.overrideredirect(False)  # keep window decorations minimal
        self._root.resizable(True, True)

        self._label = tk.Label(
            self._root,
            text="",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=(self.FONT_FAMILY, self.FONT_SIZE, "bold"),
            wraplength=1200,
            justify="center",
            padx=20,
            pady=18,
        )
        self._label.pack(expand=True, fill="both")

        self._visible = False
        self._withdraw()

        self._center_bottom()

    def _center_bottom(self) -> None:
        self._root.update_idletasks()
        w = int(self._root.winfo_screenwidth() * 0.8)
        h = 140
        x = (self._root.winfo_screenwidth() - w) // 2
        y = self._root.winfo_screenheight() - h - 60
        self._root.geometry(f"{w}x{h}+{x}+{y}")

    def set_visible(self, visible: bool) -> None:
        if visible and not self._visible:
            self._root.deiconify()
            self._visible = True
        elif not visible and self._visible:
            self._withdraw()
            self._visible = False

    def _withdraw(self) -> None:
        self._root.withdraw()
        self._visible = False

    def show_text(self, text: str) -> None:
        # Tk is not thread-safe; marshal to main thread.
        self._root.after(0, self._label.configure, {"text": text})

    def mainloop(self) -> None:
        self._root.mainloop()

    def quit(self) -> None:
        try:
            self._root.quit()
            self._root.destroy()
        except Exception:
            pass


class CaptionOverlay:
    """Owns the backend and a worker thread that drains pending segments.

    Threading model:
      - update_text() / show() / toggle() are called from any thread.
      - Worker loop calls backend methods on the calling thread (Tk thread
        must be the one calling Tk methods → use `after()` inside backend).

    Args:
        backend: Anything satisfying OverlayBackend protocol.
        queue_size: Max pending segments. Overflow drops oldest.
    """

    def __init__(
        self,
        backend: OverlayBackend | None = None,
        queue_size: int = 10,
    ) -> None:
        self._backend: OverlayBackend = backend or InMemoryBackend()
        self._pending: queue.Queue[Segment | None] = queue.Queue(maxsize=queue_size)
        self._visible = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def visible(self) -> bool:
        return self._visible

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._drain_loop,
            name="Overlay-drain",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with contextlib.suppress(queue.Full):
            self._pending.put_nowait(None)  # wake up the drain
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    # --- mutators (thread-safe) -----------------------------------------

    def show(self) -> None:
        self._visible = True
        try:
            self._backend.set_visible(True)
        except Exception:
            logger.exception("backend.set_visible(True) failed")

    def hide(self) -> None:
        self._visible = False
        try:
            self._backend.set_visible(False)
        except Exception:
            logger.exception("backend.set_visible(False) failed")

    def toggle(self) -> None:
        if self._visible:
            self.hide()
        else:
            self.show()

    def update_text(self, segment: Segment) -> None:
        """Push a segment for display. Drops oldest if queue is full."""
        try:
            self._pending.put_nowait(segment)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._pending.get_nowait()
            with contextlib.suppress(queue.Full):
                self._pending.put_nowait(segment)

    # --- internals -------------------------------------------------------

    def _drain_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                seg = self._pending.get(timeout=0.2)
            except queue.Empty:
                continue
            if seg is None:
                return
            if seg.is_empty:
                continue
            try:
                self._backend.show_text(seg.text)
            except Exception:
                logger.exception("backend.show_text() failed")

    def _drain(self) -> None:
        """For tests: synchronously drain queue once."""
        while True:
            try:
                seg = self._pending.get_nowait()
            except queue.Empty:
                return
            if seg is None:
                return
            if seg.is_empty:
                continue
            self._backend.show_text(seg.text)


__all__ = [
    "CaptionOverlay",
    "ConsoleBackend",
    "InMemoryBackend",
    "OverlayBackend",
    "TkOverlayBackend",
]
