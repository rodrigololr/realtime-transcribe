#!/usr/bin/env bash
# Captions launcher — idempotente, instala só o que falta
# Uso: ./run.sh [args...]
#      ./run.sh --list-sources
#      ./run.sh --no-overlay
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

# Check Python version
PY_VERSION="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VERSION" < "3.10" ]]; then
    echo "[captions] ERRO: Python 3.10+ necessário (encontrado $PY_VERSION)" >&2
    echo "[captions] Instale: sudo apt install python3.11 python3-pip python3-venv" >&2
    exit 1
fi

# Create venv if missing
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[captions] Criando venv em $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Ensure pip is recent (cheap if already up to date)
pip install --quiet --upgrade pip > /dev/null 2>&1 || true

# Install project deps (core + dev + stt). pip é idempotente:
# só baixa de verdade se a versão instalada é diferente da disponível.
# Após a 1ª instalação, execuções seguintes terminam em ~2s.
echo "[captions] Verificando dependências..."
pip install --quiet -e ".[dev,stt]"

# Verify critical imports before launching
if ! python -c "import faster_whisper" 2>/dev/null; then
    echo "[captions] ERRO: faster-whisper não instalou corretamente." >&2
    echo "[captions] Tente: source .venv/bin/activate && pip install -e '.[stt]'" >&2
    exit 1
fi

# Smoke check: make sure __main__ resolves (catch packaging bugs early)
if ! python -c "import importlib.util; assert importlib.util.find_spec('captions.__main__')" 2>/dev/null; then
    echo "[captions] ERRO: pacote captions não instalou direito." >&2
    echo "[captions] Tente: pip install --force-reinstall -e ." >&2
    exit 1
fi

echo "[captions] Iniciando app..."
exec python -m captions "$@"
