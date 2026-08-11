#!/usr/bin/env bash
set -Eeuo pipefail

if ! command -v nmcli >/dev/null 2>&1; then
  echo "ERROR: nmcli/NetworkManager is not available." >&2
  exit 1
fi

if [[ $EUID -ne 0 ]]; then
  exec sudo -E "$0" "$@"
fi

nmcli connection modify GateKeeper-Offline connection.autoconnect no || true
nmcli connection down GateKeeper-Offline || true

echo "GateKeeper offline hotspot disabled. Wired networking was not modified."
