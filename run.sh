#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.13}"

is_usable_python() {
  command -v "$1" >/dev/null 2>&1 && "$1" --version >/dev/null 2>&1
}

if [[ -x "$VENV_DIR/bin/python" ]]; then
  PYTHON="$VENV_DIR/bin/python"
elif is_usable_python "$PYTHON_BIN"; then
  PYTHON="$PYTHON_BIN"
elif is_usable_python python3; then
  PYTHON="python3"
else
  echo "Python 3 is required. Run ./install.sh first." >&2
  exit 1
fi

cd "$SCRIPT_DIR"
exec "$PYTHON" "$SCRIPT_DIR/main.py" "$@"
