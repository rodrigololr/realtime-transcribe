"""Application entry point.

Wires together:
  - Config parsing (CLI args)
  - PulseAudio source discovery
  - AudioCapture, VADFilter, WhisperEngine, DedupBuffer
  - CaptionOverlay (Tk or Console backend)
  - Pipeline (orchestrator)
  - Global hotkey listener (Ctrl+Shift+T/Q/R)

Run with `./run.sh` or `python -m captions`.
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
import queue
import shutil
import signal
import sys
import threading
from typing import Any

from captions.audio.capture import AudioCapture
from captions.audio.probe import find_default_monitor_source, list_sinks, list_sources
from captions.audio.vad import VADFilter
from captions.core.config import Config, parse_args
from captions.core.pipeline import Pipeline
from captions.core.segment import Segment
from captions.stt.dedup import DedupBuffer
from captions.stt.whisper_engine import WhisperEngine
from captions.ui.overlay import CaptionOverlay, ConsoleBackend, TkOverlayBackend

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Components:
    """Bundle of all wired-up collaborators (for testability)."""

    capture: AudioCapture
    vad: VADFilter
    whisper: WhisperEngine
    dedup: DedupBuffer
    overlay: CaptionOverlay
    source_name: str
    overlay_backend_type: str  # "tk" | "console"


def build_components(cfg: Config) -> Components:
    """Construct all pipeline stages from a Config.

    Pure factory: no side effects (capture.start, hotkey listener).
    """
    # 1. Resolve audio source
    source_name = cfg.source
    if source_name is None:
        source_name = find_default_monitor_source()
        logger.info("Auto-detected monitor source: %s", source_name)

    # 2. Audio capture
    if shutil.which("parec") is None:
        # Friendly exit BEFORE instantiating AudioCapture (which would also raise).
        # Points the user at the exact apt command to fix it.
        sys.stderr.write(
            "\n[captions] ERRO: o binário `parec` não foi encontrado no PATH.\n"
            "[captions] Ele vem do pacote `pulseaudio-utils`.\n\n"
            "[captions] Para instalar:\n"
            "    sudo apt update && sudo apt install -y pulseaudio-utils libpulse-dev\n\n"
            "[captions] Depois rode `./run.sh` de novo.\n"
        )
        sys.exit(2)
    capture = AudioCapture(
        source_name=source_name,
        sample_rate=16000,
        chunk_seconds=cfg.chunk_seconds,
    )

    # 3. VAD
    vad = VADFilter(energy_threshold=cfg.vad_threshold)

    # 4. Whisper
    whisper = WhisperEngine(
        model_size=cfg.model,
        device=cfg.device,
    )

    # 5. Dedup
    dedup = DedupBuffer()

    # 6. Overlay
    if cfg.no_overlay:
        backend: Any = ConsoleBackend()
        backend_type = "console"
    else:
        backend = _make_tk_backend()
        backend_type = "tk"
    overlay = CaptionOverlay(backend=backend)

    return Components(
        capture=capture,
        vad=vad,
        whisper=whisper,
        dedup=dedup,
        overlay=overlay,
        source_name=source_name,
        overlay_backend_type=backend_type,
    )


def _make_tk_backend() -> Any:
    """Lazy Tk backend constructor with friendly error if tkinter missing."""
    try:
        return TkOverlayBackend()
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Tkinter is required for the GUI overlay but is not installed. "
            "Install it: `sudo apt install python3-tk`. "
            "Or run with `--no-overlay` for console-only mode."
        ) from exc


def run_list_sources() -> None:
    """Print available sinks and sources, then exit."""
    print("=== Output sinks ===")
    for s in list_sinks():
        print(f"  {s}")
    print("\n=== Input sources (including .monitor loopbacks) ===")
    for s in list_sources():
        marker = "  [loopback]" if s.name.endswith(".monitor") else ""
        print(f"  {s}{marker}")
    sys.exit(0)


def _install_hotkeys(overlay: CaptionOverlay, dedup: DedupBuffer, stop_event: threading.Event) -> None:
    """Install global hotkey listener via pynput.

    Hotkeys:
      Ctrl+Shift+T → toggle overlay
      Ctrl+Shift+R → reset dedup history
      Ctrl+Shift+Q → quit
    """
    try:
        from pynput import keyboard
    except ImportError:
        logger.warning("pynput not available; hotkeys disabled")
        return

    pressed = set()

    def on_press(key: object) -> None:
        try:
            # Only care about modifier state
            from pynput.keyboard import Key, KeyCode

            if key in (Key.ctrl_l, Key.ctrl_r):
                pressed.add("ctrl")
            elif key in (Key.shift_l, Key.shift_r):
                pressed.add("shift")
            elif isinstance(key, KeyCode) and key.char:
                # Check combos
                mods = pressed
                if {"ctrl", "shift"}.issubset(mods):
                    if key.char.lower() == "t":
                        overlay.toggle()
                    elif key.char.lower() == "r":
                        dedup.reset()
                        logger.info("dedup history reset via hotkey")
                    elif key.char.lower() == "q":
                        logger.info("quit requested via Ctrl+Shift+Q")
                        stop_event.set()
        except Exception:
            logger.exception("hotkey handler failed")

    def on_release(key: object) -> None:
        from pynput.keyboard import Key

        if key in (Key.ctrl_l, Key.ctrl_r):
            pressed.discard("ctrl")
        elif key in (Key.shift_l, Key.shift_r):
            pressed.discard("shift")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release, daemon=True)
    listener.start()
    return listener


def run_app(cfg: Config, max_seconds: float = 0.0) -> None:
    """Build components, wire pipeline + overlay, install hotkeys, run.

    Args:
        cfg: Parsed configuration.
        max_seconds: If > 0, exit after this many seconds (for tests).
                     If 0, run until SIGINT or hotkey quit.
    """
    comps = build_components(cfg)
    logger.info(
        "captions starting | source=%s model=%s device=%s lang=%s",
        comps.source_name,
        cfg.model,
        comps.whisper.device,
        cfg.language,
    )

    # Display queue: pipeline → overlay
    display_q: queue.Queue[Segment] = queue.Queue(maxsize=20)
    drain_stop = threading.Event()

    def drain_display_to_overlay() -> None:
        while not drain_stop.is_set():
            try:
                seg = display_q.get(timeout=0.2)
            except queue.Empty:
                continue
            comps.overlay.update_text(seg)

    drain_thread = threading.Thread(target=drain_display_to_overlay, name="display-drain", daemon=True)
    drain_thread.start()

    # Pipeline
    pipeline = Pipeline(
        capture=comps.capture,
        vad=comps.vad,
        whisper=comps.whisper,
        dedup=comps.dedup,
        language=cfg.language,
    )

    # Hotkeys
    stop_event = threading.Event()
    if not cfg.no_overlay:
        comps.overlay.show()  # start visible
    listener = _install_hotkeys(comps.overlay, comps.dedup, stop_event)

    # SIGINT handler
    def _on_sigint(signum: int, frame: object) -> None:
        logger.info("SIGINT received; shutting down")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_sigint)

    # Run pipeline (blocks)
    pipe_thread: threading.Thread | None = None
    try:
        if comps.overlay_backend_type == "tk":
            # Tkinter must run on main thread. Run pipeline in worker thread,
            # and main thread runs the Tk event loop.
            pipe_thread = threading.Thread(
                target=lambda: pipeline.run(display_q),
                name="pipeline",
                daemon=True,
            )
            pipe_thread.start()
            try:
                if max_seconds > 0:
                    stop_event.wait(timeout=max_seconds)
                else:
                    tk_loop = comps.overlay._backend  # type: ignore[attr-defined]
                    if hasattr(tk_loop, "mainloop"):
                        tk_loop.mainloop()
            finally:
                stop_event.set()
        else:
            # Console: pipeline on main thread, stop on signal/hotkey
            if max_seconds > 0:
                pipeline.run(display_q, timeout=max_seconds)
            else:
                # Block until stop event; pipeline keeps running in worker thread
                pipe_thread = threading.Thread(
                    target=lambda: pipeline.run(display_q),
                    name="pipeline",
                    daemon=True,
                )
                pipe_thread.start()
                stop_event.wait()
    finally:
        stop_event.set()
        pipeline.stop()
        if pipe_thread is not None:
            pipe_thread.join(timeout=2.0)
        drain_stop.set()
        comps.overlay.stop()
        if listener is not None:
            with contextlib.suppress(Exception):
                listener.stop()
        logger.info("captions stopped")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    # Configure logging before anything else
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = parse_args(argv)
    if cfg.list_sources:
        run_list_sources()
        return 0

    try:
        run_app(cfg)
    except KeyboardInterrupt:
        logger.info("interrupted by user")
    return 0


if __name__ == "__main__":
    sys.exit(main())
