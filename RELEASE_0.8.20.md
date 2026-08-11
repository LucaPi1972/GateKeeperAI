# GateKeeper AI 0.8.20

Offline field hotspot mode for Raspberry Pi.

- Creates a local Wi-Fi AP on `wlan0` using NetworkManager.
- Uses `192.168.50.1/24` for the Raspberry Pi hotspot interface.
- Provides local DHCP/shared-network service through NetworkManager.
- Keeps the existing wired Ethernet connection untouched.
- No Internet, cloud service, Bluetooth, or external server is required.
- Mobile dashboard remains available at `http://192.168.50.1:8080/mobile`.
- GPIO17 LED/OCR behavior is unchanged.

Setup: see `OFFLINE_HOTSPOT.md` and `scripts/setup_offline_hotspot.sh`.
