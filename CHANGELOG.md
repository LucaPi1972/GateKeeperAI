# Changelog

## Current Release

See `VERSION` for the current application version.

## 0.4.1

- Converted motion detection from frame-based records to event-based `MOTION_START` and `MOTION_END` records.
- Added Motion Event Manager state handling, event images, duration tracking, and maximum contour area tracking.
- Added motion end delay configuration.

- Stabilized the core runtime startup path.
- Added automatic runtime PID file lifecycle management.
- Added startup capture metadata logging.
