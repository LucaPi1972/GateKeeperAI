#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ROUTES="$ROOT/src/gatekeeper/web/routes.py"
MOBILE="$ROOT/src/gatekeeper/web/templates/mobile.html"

if [ ! -f "$ROUTES" ] || [ ! -f "$MOBILE" ]; then
  echo "GateKeeperAI files not found. Run this script from the project root."
  exit 1
fi

cp -n "$ROUTES" "$ROUTES.m8-backup"
cp -n "$MOBILE" "$MOBILE.m8-backup"

python3 - "$ROUTES" "$MOBILE" <<'PY'
from pathlib import Path
import sys
routes = Path(sys.argv[1])
mobile = Path(sys.argv[2])

text = routes.read_text(encoding='utf-8')
old = '_OCR_STABLE_WINDOW = 1.0'
new = '_OCR_STABLE_WINDOW = 0.5'
if old not in text:
    if new not in text:
        raise SystemExit('Expected stabilization setting not found in routes.py')
else:
    text = text.replace(old, new, 1)
    routes.write_text(text, encoding='utf-8')

text = mobile.read_text(encoding='utf-8')
old = 'setInterval(refresh,800);'
new = 'setInterval(refresh,1200);'
if old in text:
    text = text.replace(old, new, 1)
elif new not in text:
    raise SystemExit('Expected mobile polling setting not found in mobile.html')
mobile.write_text(text, encoding='utf-8')
PY

echo "M8 applied:"
grep -n '_OCR_STABLE_WINDOW' "$ROUTES" | head -1
grep -n 'setInterval(refresh' "$MOBILE" | tail -1
echo "Backup files: routes.py.m8-backup and mobile.html.m8-backup"
