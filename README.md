# GateKeeper AI

GateKeeper AI is the core runtime for Raspberry Pi camera capture and full-frame motion detection. Release 0.4.0 adds a stable OpenCV motion detection loop on top of the existing `CameraManager` camera pipeline.

## Version

The single source of truth for the application version is the `VERSION` file. Startup reads this file and prints the current release in the banner.

## Startup banner

On startup the application prints runtime metadata:

```text
========================================
 GateKeeper AI v0.4.0
========================================
Build: <git short hash or "development">
Python: <python version>
Platform: <OS name>
Camera backend: Picamera2
OpenCV: <version>
```

After configuration, logging, database readiness, camera initialization, and the first capture complete, startup prints:

```text
[OK] Configuration
[OK] Logger
[OK] Database
[OK] Camera
[OK] Capture
Waiting...
```

## Runtime directory, database, and PID file

The `runtime/` directory is created automatically while the application is running. It contains `gatekeeper.pid`, which stores the current process ID, and `gatekeeper.db`, the SQLite database used for events. The PID file is removed automatically when the application exits, including when Ctrl+C sends SIGINT.

Motion events are inserted into the `events` table with these fields:

- `timestamp`
- `event_type` (`MOTION` for motion events)
- `details`

## Configuration

Runtime settings live in `config/config.yaml`:

```yaml
camera:
  width: 1640
  height: 1232
  fps: 1

motion:
  enabled: true
  min_area: 1000
  threshold: 25

debug:
  save_latest: true

logging:
  level: INFO
```

### Camera parameters

- `camera.width`: capture width passed to the existing `CameraManager`.
- `camera.height`: capture height passed to the existing `CameraManager`.
- `camera.fps`: continuous frame acquisition rate. The default is `1` frame per second.

### Motion Detection

When `motion.enabled` is `true`, each continuously captured frame is processed with OpenCV:

1. Capture frame.
2. Convert to grayscale.
3. Apply Gaussian blur.
4. Compute absolute difference from the previous frame.
5. Threshold the difference image using `motion.threshold`.
6. Dilate the threshold image.
7. Find contours.
8. Report motion only when at least one contour area is greater than or equal to `motion.min_area`.

The first frame seeds the detector and does not emit a motion event because there is no previous frame to compare.

## Capture output and logs

The application captures `images/latest.jpg` once during startup. The continuous motion loop does not overwrite `latest.jpg`.

When motion is detected, the application logs `Motion detected` with the UTC timestamp, contour area, and image path. It also saves a motion image as:

```text
images/motion_<timestamp>.jpg
```

Startup metadata and capture details are written to `logs/gatekeeper.log`, including version, Git commit, Python version, platform, camera backend, camera resolution, capture time, image path, and image size.

## Shutdown

Press Ctrl+C to stop the application. Shutdown stops the camera, closes the SQLite database, removes the PID file, and exits cleanly.

## Run

```bash
python main.py
```

Validate configuration and print the startup banner without opening the camera:

```bash
python main.py --check
```
