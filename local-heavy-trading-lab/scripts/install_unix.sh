#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-0}"
cd "$ROOT"

PYTHON=""
for candidate in python3.12 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)' >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "Python 3.12+ is required." >&2
  exit 2
fi

if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi

VENV_PYTHON="$ROOT/.venv/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install -e '.[dev]'
"$VENV_PYTHON" scripts/install_torch.py
"$VENV_PYTHON" -m heavy_lab.cli doctor --json

if [[ "$DOWNLOAD_MODELS" == "1" ]]; then
  "$VENV_PYTHON" -m heavy_lab.cli download-models --all --root "$ROOT"
fi

echo "Local Heavy Trading Lab installed successfully."
echo "Activate with: source .venv/bin/activate"
echo "Then run: lab doctor --json"
