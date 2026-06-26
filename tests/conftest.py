"""Shared pytest fixtures and patches."""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_parec_available():
    """Make tests think `parec` is installed (we mock subprocess.Popen anyway)."""
    with patch("shutil.which", return_value="/usr/bin/parec"):
        yield
