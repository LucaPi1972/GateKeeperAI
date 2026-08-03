# Changelog

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
