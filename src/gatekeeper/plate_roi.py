"""License-plate ROI extraction and OCR-oriented preview variants."""

from __future__ import annotations

from typing import Any


def _order_quad(points: Any) -> Any:
    import numpy as np
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = points[:, 0] - points[:, 1]
    ordered[0] = points[sums.argmin()]
    ordered[2] = points[sums.argmax()]
    ordered[1] = points[diffs.argmax()]
    ordered[3] = points[diffs.argmin()]
    return ordered


def _rectify_plate(roi: Any) -> Any:
    import cv2
    import numpy as np

    if roi.size == 0:
        return roi
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = float(roi.shape[0] * roi.shape[1])
    best = None
    best_score = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < roi_area * 0.12 or area > roi_area * 0.98:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, w, h = cv2.boundingRect(approx)
        ratio = max(w, h) / max(min(w, h), 1)
        if not 2.5 <= ratio <= 8.0:
            continue
        rectangularity = area / max(float(w * h), 1.0)
        score = area * rectangularity
        if score > best_score:
            best = approx.reshape(4, 2)
            best_score = score
    if best is None:
        return roi

    src = _order_quad(best)
    width_top = np.linalg.norm(src[1] - src[0])
    width_bottom = np.linalg.norm(src[2] - src[3])
    height_left = np.linalg.norm(src[3] - src[0])
    height_right = np.linalg.norm(src[2] - src[1])
    out_w = max(160, int(max(width_top, width_bottom)))
    out_h = max(32, int(max(height_left, height_right)))
    out_h = max(32, min(out_h, int(out_w / 3.0)))
    destination = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, destination)
    return cv2.warpPerspective(roi, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def build_roi_variants(frame: Any, bounding_box: tuple[int, int, int, int], margin_ratio: float = 0.10) -> dict[str, Any]:
    """Return diagnostic ROI variants derived from the selected plate box.

    The detector bounding box is expanded slightly so character edges are not
    clipped. A lightweight four-corner rectification is attempted inside the
    selected ROI. No OCR is performed here.
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

    rectified = _rectify_plate(roi)
    if len(rectified.shape) == 2:
        gray = rectified.copy()
    else:
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

    scale = max(2, min(4, int(900 / max(rectified.shape[1], 1))))
    scale = max(2, scale)
    interpolation = cv2.INTER_CUBIC
    original = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=interpolation)
    rectified_up = cv2.resize(rectified, None, fx=scale, fy=scale, interpolation=interpolation)
    gray_up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)

    denoised = cv2.bilateralFilter(gray_up, 5, 35, 35)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    threshold = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )

    return {
        "original": original,
        "rectified": rectified_up,
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
