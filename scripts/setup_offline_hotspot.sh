#!/usr/bin/env bash
set -Eeuo pipefail

# GateKeeper AI offline field hotspot setup for Raspberry Pi OS with NetworkManager.
# Keeps wired Ethernet untouched and creates a local-only Wi-Fi network on wlan0.

CONNECTION_NAME="GateKeeper-Offline"
DEFAULT_SSID="GateKeeperAI"
DEFAULT_ADDRESS="192.168.50.1/24"

if ! command -v nmcli >/dev/null 2>&1; then
  echo "ERROR: nmcli/NetworkManager is not available." >&2
  echo "Install/enable NetworkManager while the Raspberry Pi still has Internet access." >&2
  exit 1
fi

if [[ $EUID -ne 0 ]]; then
  exec sudo -E "$0" "$@"
fi

read -r -p "Wi-Fi SSID [${DEFAULT_SSID}]: " SSID
SSID="${SSID:-$DEFAULT_SSID}"

while true; do
  read -r -s -p "Wi-Fi password (minimum 8 characters): " PASSWORD
  echo
  if [[ ${#PASSWORD} -ge 8 ]]; then
    break
  fi
  echo "Password too short. Use at least 8 characters."
done

# Remove an older GateKeeper profile without touching other connections.
if nmcli -t -f NAME connection show | grep -Fxq "$CONNECTION_NAME"; then
  nmcli connection delete "$CONNECTION_NAME"
fi

nmcli connection add type wifi ifname wlan0 con-name "$CONNECTION_NAME" ssid "$SSID"
nmcli connection modify "$CONNECTION_NAME" \
  802-11-wireless.mode ap \
  802-11-wireless.band bg \
  802-11-wireless.channel 6 \
  802-11-wireless-security.key-mgmt wpa-psk \
  802-11-wireless-security.psk "$PASSWORD" \
  ipv4.method shared \
  ipv4.addresses "$DEFAULT_ADDRESS" \
  ipv6.method disabled \
  connection.autoconnect yes \
  connection.autoconnect-priority 100

nmcli connection up "$CONNECTION_NAME"

cat <<EOF

========================================
 GateKeeper AI OFFLINE HOTSPOT READY
========================================
SSID:       $SSID
Raspberry:  192.168.50.1
Web UI:     http://192.168.50.1:8080/mobile
Network:    local only / no Internet required
Ethernet:   existing wired connection left untouched
========================================

Connect the Android phone to "$SSID" and open:
  http://192.168.50.1:8080/mobile

To verify:
  nmcli connection show --active
  ip -4 addr show wlan0
EOF
