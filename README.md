# GateKeeper AI

GateKeeper AI is the core runtime for Raspberry Pi camera capture, full-frame motion detection, plate-candidate debugging, and local display calibration. Release 0.7.2 adds a compact centered plate-reading guide for printed-plate calibration while keeping OCR, whitelist, GPIO/access control, Live Preview color handling, Motion Detection, and the SQLite schema unchanged.

## Version

The single source of truth for the application version is the `VERSION` file. Startup reads this file and prints the current release in the banner.

## Startup banner

On startup the application prints runtime metadata:

```text
========================================
 GateKeeper AI v0.7.2
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

## Plate reading guide

The live calibration frame uses a compact centered guide for positioning the printed plate. On the default 1640x1232 camera frame the guide is:

```text
x = 656
 y = 581
width = 328 px
height = 70 px
aspect ratio ≈ 4.7:1
```

The guide is intentionally much smaller than the full camera frame so the printed plate can be positioned deliberately inside it. The guide is visual only in this release; it does not replace the existing full-frame PlateDetector and does not introduce OCR.

## Calibration frame capture

Automatic continuous calibration-frame saving is disabled. Use the existing **Save Calibration Frame** button when a specific test frame should be retained.

The PlateDetector calibration threshold remains `max_area: 150000`; the other detector thresholds are unchanged from 0.7.1.

## Single Frame Pipeline architecture

Release 0.6.7 makes `CameraManager.capture_frame()` the single public capture API for runtime consumers. The frame flow is:

```text
Camera
  ↓
CameraManager.capture_frame()
  ↓
FRAME_MASTER
  ├── Motion Detection
  ├── Plate Detection
  ├── Snapshot
  ├── Live Preview
  └── Diagnostics (RAW_FRAME source)
```

The live preview keeps the fixed 0.6.9/0.6.11 preview color behavior and does not expose pipeline-selection controls.

## Current calibration scope

The current phase is visual PlateDetector calibration using a printed Italian plate. OCR, whitelist matching, GPIO/access control, and schema changes are intentionally out of scope until the plate candidate is stable.
