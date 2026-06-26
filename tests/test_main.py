"""Tests for main entry point (T010)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from captions.core.config import Config
from captions.main import build_components, run_app, run_list_sources


def _stub_config(**overrides: object) -> Config:
    defaults: dict[str, object] = {
        "model": "tiny.en",
        "language": "en",
        "device": "cpu",
        "source": "test.monitor",
        "no_overlay": False,
        "list_sources": False,
        "vad_threshold": 0.01,
        "chunk_seconds": 2.0,
    }
    defaults.update(overrides)
    return Config(**defaults)  # type: ignore[arg-type]


def test_build_components_no_overlay() -> None:
    """--no-overlay should pick ConsoleBackend, no Tk involved."""
    cfg = _stub_config(no_overlay=True, source=None)
    with (
        patch("captions.main.find_default_monitor_source", return_value="auto.monitor"),
        patch("captions.main.AudioCapture") as cap_cls,
        patch("captions.main.WhisperEngine") as whi_cls,
    ):
        cap_cls.return_value = MagicMock()
        whi_cls.return_value = MagicMock()
        components = build_components(cfg)

    assert components.source_name == "auto.monitor"
    assert components.overlay_backend_type == "console"


def test_build_components_with_overlay_picks_tk() -> None:
    cfg = _stub_config(no_overlay=False, source="test.monitor")
    with (
        patch("captions.main.TkOverlayBackend") as tk_cls,
        patch("captions.main.AudioCapture") as cap_cls,
        patch("captions.main.WhisperEngine") as whi_cls,
    ):
        cap_cls.return_value = MagicMock()
        whi_cls.return_value = MagicMock()
        tk_cls.return_value = MagicMock()
        components = build_components(cfg)

    assert components.overlay_backend_type == "tk"


def test_build_components_explicit_source() -> None:
    """If --source given, don't probe PulseAudio."""
    cfg = _stub_config(source="explicit.monitor", no_overlay=True)
    with (
        patch("captions.main.find_default_monitor_source") as probe,
        patch("captions.main.AudioCapture") as cap_cls,
        patch("captions.main.WhisperEngine") as whi_cls,
    ):
        cap_cls.return_value = MagicMock()
        whi_cls.return_value = MagicMock()
        components = build_components(cfg)
        probe.assert_not_called()

    assert components.source_name == "explicit.monitor"


def test_build_components_raises_friendly_error_when_parec_missing(capsys: pytest.CaptureFixture[str]) -> None:
    """If `parec` is missing, error must tell the user exactly what to install."""
    cfg = _stub_config(source="test.monitor", no_overlay=True)
    with (
        patch("captions.main.AudioCapture") as cap_cls,
        patch("captions.main.WhisperEngine") as whi_cls,
        patch("captions.main.shutil.which", return_value=None),
    ):
        cap_cls.return_value = MagicMock()
        whi_cls.return_value = MagicMock()

        with pytest.raises(SystemExit) as exc_info:
            build_components(cfg)

    # Exit code should be non-zero (2 = our chosen code for missing prereq)
    assert exc_info.value.code == 2
    # Stderr must contain actionable info
    captured = capsys.readouterr()
    msg = captured.err.lower()
    assert "parec" in msg
    assert "pulseaudio-utils" in msg
    assert "sudo apt install" in msg


def test_run_app_runs_pipeline_and_drains_to_overlay() -> None:
    """run_app() should call pipeline.run() and overlay.stop() at the end."""
    import sys

    cfg = _stub_config(no_overlay=True, source="test.monitor")

    overlay = MagicMock()
    pipeline = MagicMock()

    # When Pipeline.run is called, return immediately (mimic timeout-based run)
    pipeline.run.side_effect = lambda *a, **kw: None
    pipeline.stop = MagicMock()

    components = MagicMock(
        capture=MagicMock(),
        vad=MagicMock(),
        whisper=MagicMock(),
        dedup=MagicMock(),
        overlay=overlay,
        source_name="test.monitor",
        overlay_backend_type="console",
    )

    # Stub pynput so test never tries to access a real keyboard
    class _FakeListener:
        def __init__(self, *a, **kw) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    fake_pynput = MagicMock()
    fake_pynput.Listener = _FakeListener
    fake_pynput.Key = MagicMock()
    fake_pynput.KeyCode = MagicMock()

    with (
        patch.dict(sys.modules, {"pynput": fake_pynput, "pynput.keyboard": fake_pynput}),
        patch("captions.main.build_components", return_value=components),
        patch("captions.main.Pipeline", return_value=pipeline),
    ):
        run_app(cfg, max_seconds=0.2)

    pipeline.run.assert_called_once()
    pipeline.stop.assert_called()
    overlay.stop.assert_called()


def test_run_list_sources_prints() -> None:
    """--list-sources should print sinks + sources and exit."""
    from captions.audio.probe import AudioDevice

    with (
        patch("captions.main.list_sinks", return_value=[AudioDevice("alsa_output.1", "Speakers")]),
        patch("captions.main.list_sources", return_value=[AudioDevice("alsa_output.1.monitor", "Monitor")]),
        patch("builtins.print") as mock_print,
    ):
        try:
            run_list_sources()
        except SystemExit as e:
            assert e.code == 0

    printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list)
    assert "alsa_output.1" in printed
    assert "Monitor" in printed
