"""License-plate ROI extraction and OCR-oriented preview variants."""

from __future__ import annotations

from typing import Any


def build_roi_variants(frame: Any, bounding_box: tuple[int, int, int, int], margin_ratio: float = 0.10) -> dict[str, Any]:
    """Return diagnostic ROI variants derived from the selected plate box.

    The detector bounding box is expanded slightly so character edges are not
    clipped. No OCR is performed here; these images are intended to inspect
    exactly what will later be supplied to the OCR stage.
    """
    import cv2

    x, y, width, height = [int(value) for value in bounding_box]
    frame_h, frame_w = frame.shape[:2]
    margin_x = max(2, int(width * margin_ratio))
    margin_y = max(2, int(height * margin_ratio))
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame_w, x + width + margin_x)
    y2 = min(frame_h, y + height + margin_y)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("Selected plate ROI is empty.")

    if len(roi.shape) == 2:
        gray = roi.copy()
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Upscale before enhancement so the browser preview is useful even when
    # the physical plate occupies only a small portion of the camera frame.
    scale = 3
    interpolation = cv2.INTER_CUBIC
    original = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=interpolation)
    gray_up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)

    denoised = cv2.bilateralFilter(gray_up, 5, 35, 35)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    threshold = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    return {
        "original": original,
        "gray": gray_up,
        "enhanced": enhanced,
        "threshold": threshold,
        "box": (x1, y1, x2 - x1, y2 - y1),
    }


def encode_variant(image: Any) -> bytes:
    """Encode an ROI variant as JPEG for the HTTP preview."""
    import cv2

    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Unable to encode plate ROI preview.")
    return encoded.tobytes()
