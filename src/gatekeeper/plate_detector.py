"""Lightweight OpenCV license plate candidate detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlateCandidate:
    bounding_box: tuple[int, int, int, int]
    confidence: float
    contour: Any
    aspect_ratio: float
    area: float
    rectangularity: float
    rotation: float
    valid: bool
    rejected_reason: str = ""


@dataclass(frozen=True)
class PlateDetection:
    """A single plate candidate returned by :class:`PlateDetector`."""

    bounding_box: tuple[int, int, int, int]
    confidence: float
    contour: Any
    aspect_ratio: float = 0.0
    area: float = 0.0
    rectangularity: float = 0.0
    rotation: float = 0.0
    candidates: tuple[PlateCandidate, ...] = field(default_factory=tuple)


class PlateDetector:
    """Detect rectangular license plate candidates without OCR."""

    def __init__(
        self,
        min_aspect_ratio: float = 3.5,
        max_aspect_ratio: float = 6.5,
        min_area: float = 2500.0,
        max_area: float = 70000.0,
        confidence_threshold: float = 0.70,
        min_rectangularity: float = 0.80,
        max_rotation: float = 15.0,
        border_margin: int = 20,
        debug: bool = False,
    ) -> None:
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_area = min_area
        self.max_area = max_area
        self.confidence_threshold = confidence_threshold
        self.min_rectangularity = min_rectangularity
        self.max_rotation = max_rotation
        self.border_margin = border_margin
        self.debug = debug
        self.last_candidates: list[PlateCandidate] = []
        self.last_debug_frames: dict[str, Any] = {}
        self._initialized = False
        self._cv2: Any | None = None

    def initialize(self) -> None:
        if self._initialized:
            return
        import cv2
        self._cv2 = cv2
        self._initialized = True

    def detect(self, frame: Any) -> PlateDetection | None:
        self.initialize()
        cv2 = self._cv2
        if cv2 is None:
            raise RuntimeError("PlateDetector is not initialized.")

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(filtered, 30, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.last_debug_frames = {"gray": gray, "edges": edged}

        candidates: list[PlateCandidate] = []
        frame_h, frame_w = frame.shape[:2]
        frame_area = float(frame_h * frame_w)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) < 4:
                continue
            x, y, width, height = cv2.boundingRect(approx)
            if width <= 0 or height <= 0:
                continue
            rect = cv2.minAreaRect(approx)
            (_, _), (rw, rh), angle = rect
            long_side, short_side = max(rw, rh), max(min(rw, rh), 1.0)
            aspect_ratio = float(long_side / short_side)
            area = float(width * height)
            contour_area = float(cv2.contourArea(approx))
            rectangularity = contour_area / max(area, 1.0)
            rotation = abs(float(angle))
            if rotation > 45:
                rotation = abs(90 - rotation)
            area_score = min(area / max(frame_area * 0.10, 1.0), 1.0)
            ratio_center = (self.min_aspect_ratio + self.max_aspect_ratio) / 2
            ratio_range = max((self.max_aspect_ratio - self.min_aspect_ratio) / 2, 0.001)
            ratio_score = max(0.0, 1.0 - abs(aspect_ratio - ratio_center) / ratio_range)
            confidence = round(max(0.01, area_score * 0.45 + ratio_score * 0.35 + rectangularity * 0.20), 3)
            reason = ""
            if confidence < self.confidence_threshold: reason = "confidence"
            elif not self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio: reason = "aspect_ratio"
            elif not self.min_area <= area <= self.max_area: reason = "area"
            elif rectangularity < self.min_rectangularity: reason = "rectangularity"
            elif rotation > self.max_rotation: reason = "rotation"
            elif x < self.border_margin or y < self.border_margin or x + width > frame_w - self.border_margin or y + height > frame_h - self.border_margin: reason = "border"
            candidates.append(PlateCandidate((x, y, width, height), confidence, approx, aspect_ratio, area, rectangularity, rotation, reason == "", reason))

        self.last_candidates = candidates
        valid = [c for c in candidates if c.valid]
        selected = max(valid, key=lambda c: c.confidence, default=None)
        if selected is None:
            return None
        return PlateDetection(selected.bounding_box, selected.confidence, selected.contour, selected.aspect_ratio, selected.area, selected.rectangularity, selected.rotation, tuple(candidates))

    def crop(self, frame: Any, detection: PlateDetection) -> Any:
        x, y, width, height = detection.bounding_box
        return frame[y : y + height, x : x + width]
