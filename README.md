# GateKeeper AI

GateKeeper AI is the core runtime for Raspberry Pi camera capture, full-frame motion detection, plate-candidate debugging, and local display calibration. Release 0.7.0 starts the license-plate calibration phase for observing the existing PlateDetector with a printed Italian plate while keeping OCR, whitelist, GPIO/access control, Live Preview color handling, Motion Detection, and the SQLite schema unchanged.

## Version

The single source of truth for the application version is the `VERSION` file. Startup reads this file and prints the current release in the banner.

## Startup banner

On startup the application prints runtime metadata:

```text
========================================
 GateKeeper AI v0.7.0
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


## Single Frame Pipeline architecture

Release 0.6.7 makes `CameraManager.capture_frame()` the single public capture API for runtime consumers. The frame flow is:

```text
Picamera2
↓
capture_array()
↓
CameraManager.capture_frame()
↓
apply_color_pipeline()
↓
apply_orientation()
↓
FRAME_MASTER
↓
Motion Detector / Plate Detector / HTTP Preview / Snapshot / Diagnostics / Future OCR
```

`FRAME_MASTER` is an RGB ndarray that has already passed through the configured color pipeline, rotation, and flip settings. Runtime consumers must treat it as read-only input for analysis, overlay rendering, JPEG output, or persistence. No consumer owns an independent camera conversion path.

### CameraManager responsibilities

`CameraManager` owns camera acquisition, color pipeline selection, rotation, flips, and JPEG preparation. `capture_frame()` returns the processed RGB `FRAME_MASTER`; `encode_jpeg()` is the central JPEG preparation API for HTTP preview, diagnostics, and snapshots. Camera reconfiguration pauses acquisition, resets the motion detector frame cache through the live state, safely restarts camera acquisition, and resumes without terminating GateKeeper.

### Pipeline verification

GateKeeper records runtime metadata for the shared frame: frame id, shape, dtype, pipeline, rotation, and horizontal/vertical flip flags. `/api/pipeline` exposes the real runtime state for camera, preview, motion, plate, snapshot, diagnostics, and JPEG consumers as read-only metadata. Startup validates the shared path and logs `FRAME PIPELINE VERIFIED` when the consumers agree, or `FRAME PIPELINE ERROR` / `FRAME PIPELINE MISMATCH` if one differs. Every 10 seconds the camera logs a `FRAME MASTER` summary with frame id, shape, pipeline, rotation, flips, and consumers.


## Plate Calibration

Release 0.7.0 adds a configurable plate calibration mode for observing the existing `PlateDetector`; it does not add OCR, whitelist matching, GPIO/access control, camera color-pipeline changes, Live Preview behavior changes, Motion Detection changes, or SQLite schema changes.

When `plate_calibration.enabled` is true, frames processed during motion expose candidate metadata through `/api/plate_calibration` and the web interface. Each candidate reports bounding box, area, aspect ratio, rectangularity, confidence, selected/rejected state, and explicit rejection reasons such as `area_too_small`, `area_too_large`, `aspect_ratio`, `rectangularity`, `invalid_geometry`, `confidence`, or `not_selected`.

Annotated calibration frames are saved as `images/calibration_*.jpg` and use green for the selected candidate, yellow for valid candidates that were not selected, and red for rejected candidates. Existing selected plate crops continue to be saved as `images/plate_*.jpg`, and selected crops continue to be inserted into the existing `plates` table. The Plate Calibration panel exposes the existing thresholds as read-only values: `min_area`, `max_area`, `min_aspect_ratio`, `max_aspect_ratio`, `min_rectangularity`, and `confidence_threshold`; no tuning sliders or automatic optimization are included in this release.

Manual calibration sequence:

1. Start GateKeeper.
2. Place the printed Italian plate in the camera view.
3. Trigger motion.
4. Observe candidate detection.
5. Save calibration frame.
6. Move the plate to different positions.
7. Repeat.

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

display:
  enabled: true
  fullscreen: false
  window_name: GateKeeper AI
  show_fps: true
  show_motion: true
  show_plate_box: true
  show_confidence: true
  show_timestamp: true
  save_snapshot_key: s

debug:
  enabled: true
  live_preview: true
  save_annotated_frames: true

plate_calibration:
  enabled: true
  save_frames: true
  show_candidates: true
  show_rejected: true
  show_metrics: true

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

### Display & Calibration mode

Release 0.6.4 replaces the previous web dashboard proposal with a local OpenCV display subsystem implemented by `DisplayManager`. Configure it in `config/config.yaml`:

```yaml
display:
  enabled: true
  fullscreen: false
  window_name: GateKeeper AI
  show_fps: true
  show_motion: true
  show_plate_box: true
  show_confidence: true
  show_timestamp: true
  save_snapshot_key: s
```

When a graphical desktop is available, GateKeeper AI opens one OpenCV window and updates it continuously with the latest camera frame. Display overlays can show current FPS, motion state, timestamp, application version, Git commit, and plate-candidate debugging information. When `PlateDetector` returns a candidate, the display draws a green rectangle plus bounding box coordinates and confidence.

### Headless mode

On Linux hosts without `DISPLAY` or `WAYLAND_DISPLAY`, `DisplayManager` disables itself automatically, logs `No graphical display detected. Running headless.`, and the application continues capturing frames and processing motion normally.

### Keyboard shortcuts

The display window supports these shortcuts:

- `q`: quit the application cleanly
- `s`: save the current displayed frame to `snapshots/snapshot_<timestamp>.jpg`
- `f`: toggle fullscreen mode
- `d`: enable or disable overlays


### Debug Vision mode

Release 0.6.4 adds Debug Vision mode for plate-detector development. Configure it in `config/config.yaml`:

```yaml
debug:
  enabled: true
  live_preview: true
  save_annotated_frames: true

plate_calibration:
  enabled: true
  save_frames: true
  show_candidates: true
  show_rejected: true
  show_metrics: true
```

When `debug.live_preview` is enabled and a display is available, GateKeeper AI opens an OpenCV preview window for the current frame. If the host is headless and neither `DISPLAY` nor `WAYLAND_DISPLAY` is available on Linux, preview is disabled automatically and the application continues running normally.

Debug overlays include:

- Motion state
- FPS
- Camera resolution
- Timestamp
- Plate candidate bounding box, confidence, and coordinates when the Plate Detector returns a candidate

Keyboard shortcuts in the preview window:

- `q`: quit the application cleanly
- `s`: save the current annotated frame
- `d`: enable or disable overlays

When `debug.save_annotated_frames` is enabled, annotated frames are written under:

```text
debug/frame_<timestamp>.jpg
```

Annotated frames are saved only when motion starts or when a plate candidate is detected.

### Plate Detector

Release 0.6.4 introduces the first plate detector. It runs when `MotionEventManager` enters `MOTION_STARTED`, so plate analysis happens once at the beginning of a motion event and does not perform OCR.

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

## Release 0.6.4 camera calibration and detector tuning

Release 0.6.4 is limited to camera calibration and plate detector tuning. It does not add OCR, whitelist logic, or database schema changes.

### Camera rotation and orientation

Configure orientation in `config/config.yaml` under `camera`:

```yaml
camera:
  rotation: 180
  flip_horizontal: false
  flip_vertical: false
```

Orientation is applied immediately after frame acquisition, so Camera consumers, Motion, Plate Detector, HTTP Preview, snapshots, and JPEG output all use the corrected frame.

### Color pipeline

Picamera2 is configured for `RGB888`. GateKeeper keeps the Release 0.6.9 Live Preview configuration for HTTP preview output and displays the active preview pipeline as read-only metadata; users cannot change the pipeline from the web UI. Startup logs the camera pixel format, internal frame format, and JPEG encoder format to help catch double-conversion mistakes.

### Detector thresholds and debug candidates

Detector tuning is configured under `plate_detector`:

```yaml
plate_detector:
  debug: true
  confidence_threshold: 0.70
  aspect_ratio_min: 3.5
  aspect_ratio_max: 6.5
  min_area: 2500
  max_area: 70000
  min_rectangularity: 0.80
  max_rotation: 15
  border_margin: 20
```

Candidates that violate any threshold are rejected. Plate events and crops are created only for a selected candidate whose confidence is at or above `confidence_threshold`. In debug mode, overlays draw every candidate: green for the selected candidate, yellow for valid discarded candidates, and red for rejected candidates. Labels show score, aspect ratio, area, and rectangularity.

### HTTP debug endpoints

Each endpoint returns MJPEG and leaves the normal HTTP Preview `/stream` behavior intact:

- `/debug/gray`
- `/debug/edges`
- `/debug/contours`
- `/debug/candidates`
- `/debug/final`

### Calibration workflow

1. Open the HTTP Preview and verify the image orientation first.
2. Adjust `camera.rotation`, `camera.flip_horizontal`, and `camera.flip_vertical` until motion, snapshots, and preview all match the real scene.
3. Enable `plate_detector.debug` and review candidate overlays.
4. Tune confidence, aspect ratio, area, rectangularity, rotation, and border margin until rejected candidates are red and the selected candidate is green.
5. Use `/api/snapshot` or the display snapshot shortcut to save `snapshot_<timestamp>.jpg` files with overlays for comparison.

## Release 0.6.7 single frame architecture

Release 0.6.7 is an architecture refactor only. It does not implement OCR, improve plate detection, change the database schema, add features, or alter the motion detection algorithm.

### RAW_FRAME

`RAW_FRAME` is the exact image object returned by Picamera2 `capture_array()`. GateKeeper AI never modifies, rotates, flips, color-converts, encodes, or annotates this object. It exists as the camera truth source and as the input to camera diagnostics.

### FRAME_MASTER

`FRAME_MASTER` is generated only by `CameraManager` from `RAW_FRAME`. `CameraManager` applies the selected color pipeline, then rotation, then flips. This is the only runtime image object consumed by motion detection, plate detection, HTTP live preview, snapshots, display/debug surfaces, and future OCR.

### Diagnostics

Diagnostics are a camera calibration tool, not the runtime preview. Each diagnostics refresh starts from `RAW_FRAME` and writes four calibration cards: `frame_raw.jpg`, `frame_rgb.jpg`, `frame_bgr.jpg`, and `frame_swap_rb.jpg`. Changing the runtime pipeline, rotation, or flips does not change the source used for diagnostics cards.

### Runtime consumers

The runtime path is deterministic:

```text
Picamera2 -> capture_array() -> RAW_FRAME -> CameraManager.apply_pipeline() -> CameraManager.apply_rotation() -> CameraManager.apply_flip() -> FRAME_MASTER -> Motion -> Plate Detector -> HTTP Preview -> Snapshot -> Future OCR
```

Live preview reproduces the Release 0.6.9 preview-only red/blue swap behavior, and snapshots use `CameraManager.encode_jpeg(FRAME_MASTER)` without preview-only changes. `/api/pipeline` exposes frame object IDs, pipeline settings, and CRC32 checksums for runtime verification. The diagnostics page includes a **Runtime Verification** section showing those IDs, checksums, pipeline settings, and PASS/FAIL status.
