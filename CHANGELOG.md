# Changelog

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
