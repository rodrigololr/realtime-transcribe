"""Entry point for `python -m captions`.

Without this module, Python can't execute the package as a script. The
console_scripts entry in pyproject.toml registers `captions` as a command,
but `python -m captions` requires a `__main__.py` file at the package root.

This file delegates to `captions.main.main()` — keeps all wiring in one place.
"""
from __future__ import annotations

import sys

from captions.main import main

if __name__ == "__main__":
    sys.exit(main())
