# GateKeeper AI

GateKeeper AI is the core runtime for Raspberry Pi camera capture and full-frame motion detection. Release 0.5.0 adds the first lightweight OpenCV Plate Detector on top of the existing `CameraManager` camera and motion-event pipeline.

## Version

The single source of truth for the application version is the `VERSION` file. Startup reads this file and prints the current release in the banner.

## Startup banner

On startup the application prints runtime metadata:

```text
========================================
 GateKeeper AI v0.5.0
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

Motion events are inserted into the `events` table as event boundaries instead of per-frame detections. The Motion Event Manager stores:

- `event_id`
- `start_time`
- `end_time`
- `duration`
- `max_contour_area`
- `image_start`
- `image_end`

## Configuration

Runtime settings live in `config/config.yaml`:

```yaml
camera:
  width: 1640
  height: 1232
  fps: 1

motion:
  enabled: true
  threshold: 25
  min_area: 1000
  end_delay_seconds: 2

debug:
  save_latest: true

logging:
  level: INFO
```

### Camera parameters

- `camera.width`: capture width passed to the existing `CameraManager`.
- `camera.height`: capture height passed to the existing `CameraManager`.
- `camera.fps`: continuous frame acquisition rate. The default is `1` frame per second.

### Motion Event Manager

When `motion.enabled` is `true`, each continuously captured frame is processed by OpenCV and then converted into event-based motion state by `MotionEventManager`.

States:

- `IDLE`
- `MOTION_STARTED`
- `MOTION_ACTIVE`
- `MOTION_FINISHED`

Frame processing still uses grayscale conversion, Gaussian blur, absolute frame difference, thresholding with `motion.threshold`, dilation, contour discovery, and `motion.min_area` to decide whether movement exists. The first frame seeds the detector and does not emit a motion event because there is no previous frame to compare.

Event behavior:

1. Movement starts: save one `images/motion_START_<timestamp>.jpg` image, insert one `MOTION_START` SQLite event, and log `Motion started`.
2. Movement continues: do not save additional images or database rows; update only the event duration tracking and maximum contour area, and log `Motion active`.
3. Movement stops: after no motion has been detected for `motion.end_delay_seconds`, save one `images/motion_END_<timestamp>.jpg` image, insert one `MOTION_END` SQLite event, and log `Motion finished`, `Duration`, and `Max contour area`.

### Plate Detector

Release 0.5.0 introduces the first plate detector. It runs when `MotionEventManager` enters `MOTION_STARTED`, so plate analysis happens once at the beginning of a motion event and does not perform OCR.

The detector is implemented in `src/gatekeeper/plate_detector.py` and uses a lightweight OpenCV pipeline:

1. Convert the frame to grayscale.
2. Apply a bilateral filter.
3. Run Canny edge detection.
4. Find contours.
5. Approximate contour polygons.
6. Keep quadrilateral candidates.
7. Filter candidates by license-plate-like aspect ratio.
8. Return the best candidate with a confidence score and bounding box.

When a plate candidate is found, GateKeeper AI saves a non-overwriting crop at:

```text
images/plate_<timestamp>.jpg
```

Each crop creates one row in the SQLite `plates` table with `event_id`, `image_path`, `confidence`, and `created_at`. Logs include `Plate detected`, confidence, bounding box, and crop path.

OCR, whitelist matching, GPIO, and dashboard features are intentionally not part of this release.


## Capture output and logs

The application captures `images/latest.jpg` once during startup. The continuous motion loop does not overwrite `latest.jpg`.

Motion event images are saved only at event boundaries:

```text
images/motion_START_<timestamp>.jpg
images/motion_END_<timestamp>.jpg
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
