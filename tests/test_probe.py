"""Tests for PulseAudio environment probe (T002)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from captions.audio.probe import (
    AudioDevice,
    NoPulseAudioError,
    find_default_monitor_source,
    list_sinks,
    list_sources,
)


def _make_sink(name: str, desc: str = "") -> MagicMock:
    sink = MagicMock()
    sink.name = name
    sink.description = desc or name
    return sink


def _make_source(name: str, desc: str = "") -> MagicMock:
    src = MagicMock()
    src.name = name
    src.description = desc or name
    return src


def test_audio_device_dataclass_str() -> None:
    """AudioDevice.__str__ should be human-readable."""
    d = AudioDevice(name="alsa_output.foo", description="Speakers")
    assert "alsa_output.foo" in str(d)
    assert "Speakers" in str(d)


def test_list_sinks_returns_audio_devices() -> None:
    """list_sinks should return AudioDevice list from pulse."""
    fake_sinks = [_make_sink("alsa_output.pci-0000", "Built-in Audio")]
    with patch("captions.audio.probe.pulsectl.Pulse") as mock_pulse_cls:
        mock_pulse = mock_pulse_cls.return_value.__enter__.return_value
        mock_pulse.sink_list.return_value = fake_sinks

        result = list_sinks()

    assert len(result) == 1
    assert result[0].name == "alsa_output.pci-0000"
    assert result[0].description == "Built-in Audio"


def test_list_sources_returns_audio_devices() -> None:
    """list_sources should return AudioDevice list from pulse."""
    fake_sources = [
        _make_source("alsa_output.pci-0000.monitor", "Monitor of Built-in"),
        _make_source("alsa_input.usb-mic", "USB Microphone"),
    ]
    with patch("captions.audio.probe.pulsectl.Pulse") as mock_pulse_cls:
        mock_pulse = mock_pulse_cls.return_value.__enter__.return_value
        mock_pulse.source_list.return_value = fake_sources

        result = list_sources()

    assert len(result) == 2
    assert result[1].name == "alsa_input.usb-mic"


def test_no_pulse_raises_clear_error() -> None:
    """When pulse daemon is unreachable, should raise NoPulseAudioError."""
    with patch("captions.audio.probe.pulsectl.Pulse") as mock_pulse_cls:
        mock_pulse_cls.side_effect = Exception("Connection refused")

        with pytest.raises(NoPulseAudioError):
            list_sinks()


def test_find_default_monitor_returns_monitor_source() -> None:
    """find_default_monitor_source should resolve default sink to its .monitor source."""
    fake_sinks = [_make_sink("alsa_output.pci-0000_00", "Speakers")]
    fake_sources = [_make_source("alsa_output.pci-0000_00.monitor", "Monitor of Speakers")]

    with patch("captions.audio.probe.pulsectl.Pulse") as mock_pulse_cls:
        mock_pulse = mock_pulse_cls.return_value.__enter__.return_value
        mock_pulse.sink_list.return_value = fake_sinks
        mock_pulse.source_list.return_value = fake_sources
        mock_pulse.server_info.return_value.default_sink_name = "alsa_output.pci-0000_00"

        monitor = find_default_monitor_source()

    assert monitor == "alsa_output.pci-0000_00.monitor"


def test_find_default_monitor_raises_when_monitor_missing() -> None:
    """If no .monitor source found for default sink, raise NoPulseAudioError."""
    fake_sinks = [_make_sink("alsa_output.pci-0000_00", "Speakers")]
    fake_sources = []  # no monitor

    with patch("captions.audio.probe.pulsectl.Pulse") as mock_pulse_cls:
        mock_pulse = mock_pulse_cls.return_value.__enter__.return_value
        mock_pulse.sink_list.return_value = fake_sinks
        mock_pulse.source_list.return_value = fake_sources
        mock_pulse.server_info.return_value.default_sink_name = "alsa_output.pci-0000_00"

        with pytest.raises(NoPulseAudioError, match="monitor"):
            find_default_monitor_source()


def test_find_default_monitor_prefers_running_sink() -> None:
    """When multiple sinks, prefer the running one over default."""
    fake_sinks = [
        _make_sink("alsa_output.hdmi", "HDMI"),
        _make_sink("alsa_output.speakers", "Speakers"),
    ]
    fake_sources = [
        _make_source("alsa_output.hdmi.monitor", "HDMI Monitor"),
        _make_source("alsa_output.speakers.monitor", "Speakers Monitor"),
    ]

    with patch("captions.audio.probe.pulsectl.Pulse") as mock_pulse_cls:
        mock_pulse = mock_pulse_cls.return_value.__enter__.return_value
        mock_pulse.sink_list.return_value = fake_sinks
        mock_pulse.source_list.return_value = fake_sources
        # default is HDMI but speakers is actually running
        mock_pulse.server_info.return_value.default_sink_name = "alsa_output.hdmi"
        mock_pulse.get_sink_by_name.side_effect = lambda n: {
            "alsa_output.hdmi": MagicMock(state="idle"),
            "alsa_output.speakers": MagicMock(state="running"),
        }[n]

        monitor = find_default_monitor_source()

    assert monitor == "alsa_output.speakers.monitor"
