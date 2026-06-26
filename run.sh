#!/usr/bin/env bash
# Captions launcher — creates venv if needed, installs deps, runs app
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

# Check Python version
PY_VERSION="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VERSION" < "3.10" ]]; then
    echo "Error: Python 3.10+ required (found $PY_VERSION)" >&2
    exit 1
fi

# Create venv if missing
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[captions] Creating venv..."
    $PYTHON -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Install/upgrade deps
echo "[captions] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -e ".[dev]"

# Pre-download whisper model on first run (small.en ~460MB)
MODEL_CACHE="$HOME/.cache/captions/models"
mkdir -p "$MODEL_CACHE"

echo "[captions] Launching app..."
exec python -m captions "$@"
