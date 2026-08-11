#!/usr/bin/env bash
set -Eeuo pipefail

if ! command -v nmcli >/dev/null 2>&1; then
  echo "ERROR: nmcli/NetworkManager is not available." >&2
  exit 1
fi

if [[ $EUID -ne 0 ]]; then
  exec sudo -E "$0" "$@"
fi

nmcli connection modify GateKeeper-Offline connection.autoconnect yes
nmcli connection up GateKeeper-Offline

echo "GateKeeper offline hotspot enabled."
echo "Open http://192.168.50.1:8080/mobile from the connected phone."
