# Offline field mode

GateKeeper AI can be operated without Internet access. The Raspberry Pi creates its own Wi-Fi access point and the Android phone connects directly to it.

## Network layout

```text
Android phone
     │ Wi-Fi
     ▼
Raspberry Pi
  wlan0: 192.168.50.1
     │
     ├── GateKeeper web UI :8080
     ├── Camera / OCR local processing
     └── GPIO17 LED

Internet: not required
Ethernet: optional and independent
```

The application HTTP server listens on `0.0.0.0:8080`, so it is reachable through the hotspot interface as well as the wired interface. The mobile dashboard is available at `/mobile`.

## Initial setup

Do this once while the Raspberry Pi is still connected to the Internet or otherwise has NetworkManager available.

```bash
cd ~/GateKeeperAI
chmod +x scripts/setup_offline_hotspot.sh
sudo ./scripts/setup_offline_hotspot.sh
```

The script asks for the Wi-Fi SSID and WPA2 password. It creates the `GateKeeper-Offline` NetworkManager profile, assigns `192.168.50.1/24` to `wlan0`, enables NetworkManager's local shared-network/DHCP mode, disables IPv6 on this profile, and enables autoconnect.

The existing Ethernet connection is not modified.

## Field use

1. Power the Raspberry Pi from the powerbank.
2. Wait for the Pi to boot and GateKeeper AI to start.
3. Connect the Android phone to the configured GateKeeper Wi-Fi SSID.
4. Open `http://192.168.50.1:8080/mobile`.
5. Use the mobile dashboard to monitor OCR/Plate Lock and GPIO status.
6. The LED on BCM GPIO17 continues to provide the local recognition feedback.

No Internet connection, cloud service, Bluetooth, or external server is required for this mode.

## Verification

On the Raspberry Pi:

```bash
nmcli connection show --active
ip -4 addr show wlan0
ping -c 1 192.168.50.1
```

From the phone, open:

```text
http://192.168.50.1:8080/mobile
```

## Returning to normal wired operation

The hotspot profile is independent. To temporarily stop it:

```bash
sudo nmcli connection down GateKeeper-Offline
```

To restore it:

```bash
sudo nmcli connection up GateKeeper-Offline
```

Because the profile is configured with autoconnect, it will normally come back automatically after reboot. If field mode should be disabled permanently, set `connection.autoconnect no` for this profile.
