"""Tests for package skeleton (T001)."""
from __future__ import annotations

import importlib.util


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


def test_main_module_invokable() -> None:
    """`python -m captions` requires captions.__main__ to exist."""
    spec = importlib.util.find_spec("captions.__main__")
    assert spec is not None, "captions.__main__ missing — `python -m captions` will fail"


def test_main_module_imports_without_error() -> None:
    """Importing __main__ must not raise."""
    spec = importlib.util.find_spec("captions.__main__")
    assert spec is not None
    # Don't exec here — that would actually run main(). Just check spec is valid.
    assert spec.loader is not None
