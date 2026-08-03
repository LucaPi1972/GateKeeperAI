# GateKeeper AI

GateKeeper AI is the core runtime for Raspberry Pi camera capture. Release 0.3.3 stabilizes startup, logging, camera capture, and process lifecycle management so the project is ready for a future Motion Detection release.

## Version

The single source of truth for the application version is the `VERSION` file. Startup reads this file and prints the current release in the banner.

## Startup banner

On startup the application prints runtime metadata:

```text
========================================
 GateKeeper AI v0.3.3
========================================
Build: <git short hash or "development">
Python: <python version>
Platform: <OS name>
Camera backend: Picamera2
OpenCV: <version>
```

After configuration, logging, database readiness, camera initialization, and capture complete, startup prints:

```text
[OK] Configuration
[OK] Logger
[OK] Database
[OK] Camera
[OK] Capture
Waiting...
```

## Runtime directory and PID file

The `runtime/` directory is created automatically while the application is running. It contains `gatekeeper.pid`, which stores the current process ID. The PID file is removed automatically when the application exits, including when Ctrl+C sends SIGINT.

## Configuration

Runtime settings live in `config/config.yaml`:

```yaml
camera:
  width: 1640
  height: 1232
  fps: 1

debug:
  save_latest: true

logging:
  level: INFO
```

## Capture output and logs

The application captures `images/latest.jpg` at startup. Startup metadata and capture details are written to `logs/gatekeeper.log`, including version, Git commit, Python version, platform, camera backend, camera resolution, capture time, image path, and image size.

## Run

```bash
python main.py
```

Validate configuration and print the startup banner without opening the camera:

```bash
python main.py --check
```
