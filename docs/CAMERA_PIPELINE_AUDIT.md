# Camera Pipeline Conversion Audit

Release 0.6.7 makes `CameraManager` the only runtime pixel transformation boundary.
This audit is architecture-only: it does not implement OCR, improve plate detection,
change the database schema, or alter the motion detection algorithm.

## Runtime processing path

The runtime camera path contains exactly two image objects:

1. `RAW_FRAME`: the unmodified object returned by `Picamera2.capture_array()`.
2. `FRAME_MASTER`: the single processed object created by `CameraManager` from
   `RAW_FRAME` with the selected color pipeline, rotation, and flips applied once.

All runtime consumers receive `FRAME_MASTER`: motion detection, plate detection,
HTTP live preview, snapshots, display/debug surfaces, and future OCR.
Diagnostics are the only exception and intentionally use `RAW_FRAME` only.

## Allowed pixel operations

All occurrences of `cv2.cvtColor()`, `cv2.rotate()`, `cv2.flip()`, and
`cv2.imencode()` are centralized in `main.py` on `CameraManager` methods:

- `CameraManager.apply_pipeline()`
- `CameraManager.apply_rotation()`
- `CameraManager.apply_flip()`
- `CameraManager.to_gray()`
- `CameraManager.encode_jpeg()`
- `CameraManager.write_image()`

## Verification command

```bash
rg -n "cv2\.(cvtColor|rotate|flip|imencode)|capture_array" main.py src tests
```

Expected result: pixel transformations are reported only in `main.py` inside
`CameraManager`; `capture_array()` appears only where RAW_FRAME is acquired.
