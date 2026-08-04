# Camera Pipeline Conversion Audit

Release 0.6.6 documents every occurrence of frame conversion, orientation, JPEG encoding, and Picamera2 acquisition calls. This audit is intentionally diagnostic-only and does not change OCR, plate detection algorithms, motion detection algorithms, or the database schema.

## Runtime processing path

The runtime camera frame path is:

1. `CameraManager.get_processed_frame()` acquires one frame with `capture_array()`.
2. `apply_color_pipeline()` applies the selected color pipeline.
3. `apply_orientation()` applies rotation and flips once.
4. The same processed frame object is passed to motion detection, plate detection, HTTP preview, snapshots, display/debug surfaces, and JPEG encoding boundaries.

## Occurrence inventory

Command used:

```bash
rg -n "cv2\.(cvtColor|rotate|flip|imencode)|capture_array" main.py src tests
```

| File | Line | Occurrence | Purpose |
| --- | ---: | --- | --- |
| `main.py` | 186 | `cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)` | Selected `bgr` diagnostic camera pipeline conversion. |
| `main.py` | 198 | `cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)` | Diagnostic image generation for `frame_bgr.jpg`. |
| `main.py` | 462 | `cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)` | HTTP debug grayscale stream preparation. |
| `main.py` | 480 | `cv2.rotate(...ROTATE_90_CLOCKWISE)` | Single orientation path for 90° rotation. |
| `main.py` | 482 | `cv2.rotate(...ROTATE_180)` | Single orientation path for 180° rotation. |
| `main.py` | 484 | `cv2.rotate(...ROTATE_90_COUNTERCLOCKWISE)` | Single orientation path for 270° rotation. |
| `main.py` | 488 | `cv2.flip(..., -1)` | Single orientation path for combined horizontal/vertical flip. |
| `main.py` | 490 | `cv2.flip(..., 1)` | Single orientation path for horizontal flip. |
| `main.py` | 492 | `cv2.flip(..., 0)` | Single orientation path for vertical flip. |
| `main.py` | 504 | `cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)` | JPEG boundary conversion in `encode_jpeg()`. |
| `main.py` | 513 | `cv2.imencode(".jpg", image)` | Single JPEG byte encoder helper. |
| `main.py` | 632 | `capture_array()` | Runtime processed-frame acquisition. |
| `main.py` | 669 | `capture_array()` | Raw diagnostic acquisition for diagnostic images only. |
| `main.py` | 912 | `cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)` | Motion detector grayscale preparation. |
| `main.py` | 950 | `cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)` | Motion event image file boundary. |
| `main.py` | 1207 | `cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)` | Debug Vision annotated image save/display boundary. |
| `main.py` | 1223 | `cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)` | Debug Vision OpenCV window display boundary. |
| `main.py` | 1402 | `cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)` | Plate crop image file boundary. |
| `src/gatekeeper/plate_detector.py` | 78 | `cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)` | Plate detector grayscale preparation. |
| `src/gatekeeper/display_manager.py` | 126 | `cv2.cvtColor(displayed, cv2.COLOR_RGB2BGR)` | OpenCV display boundary after overlay. |
| `src/gatekeeper/display_manager.py` | 151 | `cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)` | Display snapshot image file boundary. |
| `tests/test_main.py` | 791 | `capture_array()` | Test fake camera acquisition method. |

