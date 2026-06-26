"""PulseAudio environment probe.

Detects default sink and its corresponding `.monitor` loopback source so the
audio capture stage can subscribe to system audio without needing a microphone.

This module is intentionally pure-Python and uses `pulsectl` (libpulse binding).
It never opens an audio stream — it only inspects the PulseAudio server.
"""
from __future__ import annotations

from dataclasses import dataclass

import pulsectl


class NoPulseAudioError(RuntimeError):
    """Raised when the PulseAudio daemon is unreachable or has no monitor source."""


@dataclass(frozen=True, slots=True)
class AudioDevice:
    """Lightweight DTO representing a PulseAudio sink or source."""

    name: str
    description: str

    def __str__(self) -> str:
        return f"{self.name} — {self.description}"


def _pulse_connect():
    """Open a PulseAudio context manager, raising NoPulseAudioError on any failure.

    `pulsectl.Pulse` instances are themselves context managers, so this just
    performs the connection and lets the caller use `with`. Wrapping construction
    means connection failures surface as NoPulseAudioError instead of generic
    pulsectl exceptions.
    """
    try:
        return pulsectl.Pulse("captions-probe")
    except Exception as exc:
        raise NoPulseAudioError(
            f"Cannot connect to PulseAudio daemon: {exc}. "
            "Is `pulseaudio` running? Try `systemctl --user status pulseaudio`."
        ) from exc


def list_sinks() -> list[AudioDevice]:
    """Return all output sinks (speakers, headphones, HDMI, etc.)."""
    with _pulse_connect() as pulse:
        return [AudioDevice(name=s.name, description=s.description) for s in pulse.sink_list()]


def list_sources() -> list[AudioDevice]:
    """Return all input sources (microphones AND .monitor loopbacks)."""
    with _pulse_connect() as pulse:
        return [AudioDevice(name=s.name, description=s.description) for s in pulse.source_list()]


def find_default_monitor_source() -> str:
    """Resolve the best `.monitor` source to capture system audio from.

    Strategy:
      1. Prefer a sink whose state == "running" (something is actually playing).
      2. Fall back to PulseAudio's configured default sink.
      3. Derive its monitor source as `<sink_name>.monitor`.

    Returns the monitor source name (e.g. "alsa_output.pci-0000.monitor").

    Raises NoPulseAudioError if no monitor source can be found.
    """
    with _pulse_connect() as pulse:
        sinks = pulse.sink_list()
        sources = pulse.source_list()
        monitor_names = {s.name for s in sources}

        if not sinks:
            raise NoPulseAudioError("No output sinks found on PulseAudio.")

        # 1. Prefer running sink
        running = [s for s in sinks if _sink_state(pulse, s.name) == "running"]
        candidates = running or sinks

        # 2. Default sink first
        default_name = pulse.server_info().default_sink_name
        candidates.sort(key=lambda s: 0 if s.name == default_name else 1)

        # 3. Match to monitor
        for sink in candidates:
            monitor_name = f"{sink.name}.monitor"
            if monitor_name in monitor_names:
                return monitor_name

        raise NoPulseAudioError(
            f"Found {len(sinks)} sinks but none has a `.monitor` source. "
            "Try logging out / restarting PulseAudio: `pulseaudio -k && pulseaudio --start`."
        )


def _sink_state(pulse: pulsectl.Pulse, sink_name: str) -> str:
    """Read the current state of a sink ('running', 'idle', 'suspended', ...)."""
    try:
        info = pulse.get_sink_by_name(sink_name)
    except pulsectl.PulseError:
        return "unknown"
    return info.state
