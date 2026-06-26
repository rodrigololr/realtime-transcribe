"""CLI configuration parsing.

Pure argparse wrapper. Returns a frozen `Config` dataclass so downstream
modules can safely pass it around without worrying about mutation.
"""
from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence

from captions.stt.whisper_engine import VALID_MODELS


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """Immutable runtime configuration."""

    model: str
    language: str
    device: str  # "cuda" | "cpu" | "auto"
    source: str | None  # PulseAudio source name; None = auto-detect
    no_overlay: bool
    list_sources: bool
    vad_threshold: float
    chunk_seconds: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="captions",
        description="Real-time audio transcriber overlay for Linux.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(VALID_MODELS),
        default="small.en",
        help="Whisper model size (default: small.en). Tiny = fast/CPU, small = balanced, medium = best quality.",
    )
    parser.add_argument(
        "--lang",
        "--language",
        dest="language",
        default="en",
        choices=["en", "pt", "es", "fr", "de", "it", "auto"],
        help="Language hint for Whisper (default: en).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Inference device (default: auto — pick CUDA if available).",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="PulseAudio source name (e.g. alsa_output.pci-0000.monitor). "
             "Default: auto-detect running sink's monitor source.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.01,
        help="RMS energy threshold for VAD (0-1, default 0.01). Higher = stricter.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=2.0,
        help="Audio chunk duration in seconds (default 2.0).",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip the GUI overlay; print captions to stdout instead.",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Print available PulseAudio sources and exit.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> Config:
    """Parse command-line args into a frozen Config.

    `argv=None` reads from sys.argv (normal CLI). Pass an explicit list for
    testing or programmatic invocation.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return Config(
        model=args.model,
        language=args.language,
        device=args.device,
        source=args.source,
        no_overlay=args.no_overlay,
        list_sources=args.list_sources,
        vad_threshold=args.vad_threshold,
        chunk_seconds=args.chunk_seconds,
    )


__all__ = ["Config", "parse_args"]
