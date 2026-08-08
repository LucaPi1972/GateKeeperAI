# Changelog

## 0.6.10

- Fix HTTP Live Preview to use a fixed BGR pipeline by converting FRAME_MASTER RGB to BGR exactly once before MJPEG JPEG encoding.
- Keep FRAME_MASTER, motion detection, plate detection, snapshots, diagnostics, and database schema unchanged while runtime camera pipeline controls remain available.

## 0.6.9

- Fix live HTTP preview color by applying the configured preview-only red/blue swap before MJPEG JPEG encoding.
- Keep FRAME_MASTER, snapshots, diagnostics, motion detection, and plate detection unchanged.

## 0.6.8

- Stabilize live preview color handling with an isolated preview-only red/blue swap configuration.
- Keep diagnostics generated from RAW_FRAME on every refresh, independent of runtime pipeline changes.
- Rename diagnostics to Camera Calibration and separate sensor calibration images from runtime preview metadata.

## 0.6.7

- Refactor runtime capture around one CameraManager-owned FRAME_MASTER consumed by preview, diagnostics, snapshots, motion detection, and plate detection.
- Add runtime pipeline metadata, `/api/pipeline` frame id reporting, startup verification logs, mismatch logging, and periodic FRAME MASTER debug summaries.
- Centralize JPEG preparation behind `CameraManager.encode_jpeg()` while preserving the existing motion, plate, and SQLite behavior.

## 0.6.6

- Add a diagnostic camera pipeline inspector for startup, reconfiguration, first-frame runtime path logging, and the `/api/pipeline` runtime endpoint.
- Add motion detector frame-size safety reset before `cv2.absdiff()` to avoid crashes after orientation changes.
- Document all frame acquisition, conversion, orientation, and JPEG encoding call sites in `docs/CAMERA_PIPELINE_AUDIT.md`.

## 0.6.5

- Stabilize the shared camera processing path so diagnostics, live preview, streams, snapshots, motion, and plate detection consume the same processed frame.
- Safely reconfigure camera pipeline, rotation, flips, and controls at runtime without crashing the stream.

## 0.6.4

- Fix camera orientation order so rotation and flips are applied immediately after acquisition and before motion detection, plate detection, preview, snapshots, diagnostics, and JPEG encoding.

## 0.6.3

- Add camera orientation calibration, corrected RGB/BGR logging, configurable plate detector filters, debug MJPEG endpoints, calibration panel metadata, and overlay snapshots.


## 0.5.2

- Add Display & Calibration local OpenCV preview with overlays, fullscreen control, snapshots, keyboard shortcuts, and automatic headless fallback.


## Current Release

See `VERSION` for the current application version.

## 0.5.0

- Added the first lightweight OpenCV Plate Detector.
- Added plate crop generation when motion starts.
- Added SQLite `plates` records for detected plate crops.

## 0.4.1

- Converted motion detection from frame-based records to event-based `MOTION_START` and `MOTION_END` records.
- Added Motion Event Manager state handling, event images, duration tracking, and maximum contour area tracking.
- Added motion end delay configuration.

- Stabilized the core runtime startup path.
- Added automatic runtime PID file lifecycle management.
- Added startup capture metadata logging.
