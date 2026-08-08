# Changelog

## 0.7.2

- Shrink the centered plate-ideal/reading guide to a compact 328x70 px zone on the 1640x1232 live frame, preserving an approximately 4.7:1 plate proportion.
- Keep the guide centered at x=656, y=581 so the printed plate can be positioned deliberately inside a small reading area and false candidates outside the area remain visually obvious during calibration.
- Disable continuous automatic calibration-frame saving; calibration frames remain manual through the existing Save Calibration Frame button.
- Keep the PlateDetector thresholds from 0.7.1 unchanged, including `max_area: 150000`.
- Keep Live Preview color handling, FRAME_MASTER, Motion Detection, diagnostics, snapshots, and OCR scope unchanged.

## 0.7.1

- Increase only the PlateDetector `max_area` default and configured value from `70000` to `150000`; all other detector thresholds remain unchanged.
- Add live Plate Calibration overlays to the existing HTTP Live Preview using the same current FRAME_MASTER-derived frame already evaluated by PlateDetector, without reopening the camera or creating another stream.
- Draw selected, valid-not-selected, and rejected candidates with metrics, rejection reasons, selected-plate emphasis, and runtime threshold summaries.
- Update the Plate Calibration web panel and save action so calibration state refreshes continuously and saved calibration frames use the current annotated preview frame.

## 0.7.0

- Start the license-plate calibration phase for observing the existing PlateDetector with printed Italian plates.
- Add configurable `plate_calibration` mode, candidate metadata, explicit rejection reasons, selected-candidate logging, and annotated `images/calibration_*.jpg` saves while preserving existing `images/plate_*.jpg` crops.
- Add `/api/plate_calibration`, manual calibration-frame saving, and a Plate Calibration web panel with read-only existing detector thresholds.
- Keep OCR, whitelist, GPIO/access control, Live Preview color handling, motion detection, and SQLite schema unchanged.

## 0.6.11

- Restore the HTTP Live Preview color behavior from Release 0.6.9 using the fixed preview-only red/blue swap configuration.
- Remove web/API pipeline-selection operations while keeping `/api/pipeline` as read-only runtime metadata.
- Keep motion detection, plate detection, snapshots, diagnostics, CameraManager, and rotation behavior unchanged.

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

## 0.4.1

- Converted motion detection from frame-based records to event-based `MOTION_START` and `MOTION_END` records.
- Added Motion Event Manager state handling, event images, duration tracking, and maximum contour area tracking.
- Added motion end delay configuration.
- Stabilized the core runtime startup path.
- Added automatic runtime PID file lifecycle management.
- Added startup capture metadata logging.
