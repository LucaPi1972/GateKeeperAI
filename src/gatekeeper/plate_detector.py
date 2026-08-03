"""Lightweight OpenCV license plate candidate detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlateDetection:
    """A single plate candidate returned by :class:`PlateDetector`."""

    bounding_box: tuple[int, int, int, int]
    confidence: float
    contour: Any


class PlateDetector:
    """Detect rectangular license plate candidates without OCR."""

    def __init__(
        self,
        min_aspect_ratio: float = 2.0,
        max_aspect_ratio: float = 6.0,
        min_area: float = 500.0,
    ) -> None:
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_area = min_area
        self._initialized = False
        self._cv2: Any | None = None

    def initialize(self) -> None:
        """Load OpenCV so startup fails early when plate detection is required."""
        if self._initialized:
            return
        import cv2

        self._cv2 = cv2
        self._initialized = True

    def detect(self, frame: Any) -> PlateDetection | None:
        """Return the best quadrilateral plate candidate in ``frame``, if any."""
        self.initialize()
        cv2 = self._cv2
        if cv2 is None:
            raise RuntimeError("PlateDetector is not initialized.")

        gray = (
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if len(frame.shape) == 3
            else frame
        )
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(filtered, 30, 200)
        contours, _ = cv2.findContours(
            edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: list[PlateDetection] = []
        frame_area = float(frame.shape[0] * frame.shape[1])
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4:
                continue

            x, y, width, height = cv2.boundingRect(approx)
            if width <= 0 or height <= 0:
                continue
            area = float(width * height)
            if area < self.min_area:
                continue
            aspect_ratio = width / height
            if not self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                continue

            area_score = min(area / max(frame_area * 0.10, 1.0), 1.0)
            ratio_center = (self.min_aspect_ratio + self.max_aspect_ratio) / 2
            ratio_range = (self.max_aspect_ratio - self.min_aspect_ratio) / 2
            ratio_score = max(
                0.0, 1.0 - abs(aspect_ratio - ratio_center) / ratio_range
            )
            confidence = round(
                max(0.01, (area_score * 0.6) + (ratio_score * 0.4)), 3
            )
            candidates.append(
                PlateDetection((x, y, width, height), confidence, approx)
            )

        return max(
            candidates, key=lambda candidate: candidate.confidence, default=None
        )

    def crop(self, frame: Any, detection: PlateDetection) -> Any:
        """Return the detected plate region cropped from ``frame``."""
        x, y, width, height = detection.bounding_box
        return frame[y : y + height, x : x + width]
