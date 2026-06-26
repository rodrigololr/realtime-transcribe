"""Tests for package skeleton (T001)."""
from __future__ import annotations


def test_package_importable() -> None:
    """Package should import without error."""
    import captions  # noqa: F401


def test_version_defined() -> None:
    """Package should expose __version__ string."""
    import captions

    assert isinstance(captions.__version__, str)
    assert captions.__version__  # non-empty


def test_subpackages_importable() -> None:
    """All subpackages should be importable."""
    from captions import audio, core, stt, ui  # noqa: F401


def test_main_entrypoint_exists() -> None:
    """main module should be importable (entrypoint wired in pyproject)."""
    from captions import main  # noqa: F401
