#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.13}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 || ! "$PYTHON_BIN" --version >/dev/null 2>&1; then
  echo "Python 3.13 was not found or is not runnable. Install it and rerun this script." >&2
  exit 1
fi

cd "$SCRIPT_DIR"
"$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

echo "Installation complete. Start GateKeeper AI with ./run.sh"
