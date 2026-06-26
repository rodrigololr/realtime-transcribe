"""Tests for Config CLI (T008)."""
from __future__ import annotations

import dataclasses

import pytest

from captions.core.config import parse_args


def test_defaults() -> None:
    """No args → sensible defaults for English + GPU."""
    cfg = parse_args([])
    assert cfg.model == "small.en"
    assert cfg.language == "en"
    assert cfg.device == "auto"
    assert cfg.no_overlay is False
    assert cfg.list_sources is False
    assert cfg.vad_threshold == 0.01


def test_model_short_alias() -> None:
    cfg = parse_args(["--model", "tiny"])
    assert cfg.model == "tiny"


def test_invalid_model_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--model", "gpt-4"])


def test_language_override() -> None:
    cfg = parse_args(["--lang", "pt"])
    assert cfg.language == "pt"


def test_device_explicit() -> None:
    cfg = parse_args(["--device", "cpu"])
    assert cfg.device == "cpu"


def test_no_overlay_flag() -> None:
    cfg = parse_args(["--no-overlay"])
    assert cfg.no_overlay is True


def test_list_sources_flag() -> None:
    cfg = parse_args(["--list-sources"])
    assert cfg.list_sources is True


def test_source_override() -> None:
    cfg = parse_args(["--source", "alsa_output.foo.monitor"])
    assert cfg.source == "alsa_output.foo.monitor"


def test_vad_threshold() -> None:
    cfg = parse_args(["--vad-threshold", "0.05"])
    assert cfg.vad_threshold == 0.05


def test_config_is_frozen() -> None:
    """Config dataclass should be immutable."""
    cfg = parse_args([])

    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.model = "hacked"  # type: ignore[misc]


def test_help_exits_cleanly() -> None:
    """--help should exit with code 0."""
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--help"])
    assert excinfo.value.code == 0
