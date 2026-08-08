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
    selected: bool = False

    @property
    def rejected(self) -> bool:
        return not self.valid or bool(self.rejected_reason)

    def metadata(self) -> dict[str, Any]:
        reason = self.rejected_reason
        if self.valid and not self.selected:
            reason = "not_selected"
        return {
            "bounding_box": self.bounding_box,
            "area": self.area,
            "aspect_ratio": self.aspect_ratio,
            "rectangularity": self.rectangularity,
            "confidence": self.confidence,
            "selected": self.selected,
            "rejected": not self.selected,
            "rejection_reason": "" if self.selected else reason,
        }


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
        max_area: float = 150000.0,
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

        import sys
        camera_manager = getattr(sys.modules.get("main") or sys.modules.get("__main__"), "CameraManager")
        gray = camera_manager.to_gray(frame)
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
                candidates.append(PlateCandidate((x, y, max(width, 0), max(height, 0)), 0.0, approx, 0.0, 0.0, 0.0, 0.0, False, "invalid_geometry"))
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
            if area < self.min_area:
                reason = "area_too_small"
            elif area > self.max_area:
                reason = "area_too_large"
            elif not self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                reason = "aspect_ratio"
            elif rectangularity < self.min_rectangularity:
                reason = "rectangularity"
            elif confidence < self.confidence_threshold:
                reason = "confidence"
            candidates.append(PlateCandidate((x, y, width, height), confidence, approx, aspect_ratio, area, rectangularity, rotation, reason == "", reason))

        valid = [c for c in candidates if c.valid]
        selected = max(valid, key=lambda c: c.confidence, default=None)
        if selected is not None:
            candidates = [PlateCandidate(c.bounding_box, c.confidence, c.contour, c.aspect_ratio, c.area, c.rectangularity, c.rotation, c.valid, c.rejected_reason, c.bounding_box == selected.bounding_box) for c in candidates]
            selected = next(c for c in candidates if c.selected)
        self.last_candidates = candidates
        if selected is None:
            return None
        return PlateDetection(selected.bounding_box, selected.confidence, selected.contour, selected.aspect_ratio, selected.area, selected.rectangularity, selected.rotation, tuple(candidates))

    def crop(self, frame: Any, detection: PlateDetection) -> Any:
        x, y, width, height = detection.bounding_box
        return frame[y : y + height, x : x + width]
