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
    zone_score: float = 0.0
    selection_score: float = 0.0

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
            "zone_score": self.zone_score,
            "selection_score": self.selection_score,
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
    """Detect rectangular license plate candidates without OCR.

    Release 0.7.5 keeps the 0.7.4 full-frame detector, but uses the configured
    reading zone only as a soft ranking signal. Candidates are never discarded
    merely because they are outside the zone. A near-valid fallback is allowed
    when the strict detector has no valid candidate, which prevents the reading
    zone from creating false negatives while still improving candidate ranking.
    """

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
        reading_zone: dict[str, Any] | None = None,
        reading_zone_weight: float = 0.15,
        soft_selection_threshold: float = 0.42,
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
        self.reading_zone = dict(reading_zone or {})
        self.reading_zone_weight = max(0.0, min(float(reading_zone_weight), 1.0))
        self.soft_selection_threshold = max(0.0, min(float(soft_selection_threshold), 1.0))
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

    def _zone_score(self, bbox: tuple[int, int, int, int], frame_w: int, frame_h: int) -> float:
        """Return 0..1 proximity to the configured reading zone center.

        The zone is a soft preference only. A disabled or invalid zone returns
        the neutral score 0.5 so candidate ranking remains geometry-driven.
        """
        if not self.reading_zone:
            return 0.5
        x = float(self.reading_zone.get("x", 0))
        y = float(self.reading_zone.get("y", 0))
        w = max(float(self.reading_zone.get("width", 0)), 1.0)
        h = max(float(self.reading_zone.get("height", 0)), 1.0)
        if w <= 1 or h <= 1:
            return 0.5
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        zone_cx = x + w / 2.0
        zone_cy = y + h / 2.0
        # One zone width/height away reaches zero; inside the zone scores 1.
        dx = abs(cx - zone_cx) / (w / 2.0)
        dy = abs(cy - zone_cy) / (h / 2.0)
        distance = max(dx, dy)
        return round(max(0.0, 1.0 - distance), 3)

    def _selection_score(
        self,
        confidence: float,
        aspect_ratio: float,
        rectangularity: float,
        area: float,
        zone_score: float,
        frame_area: float,
    ) -> float:
        """Combine geometry quality with a deliberately small zone preference."""
        ratio_center = (self.min_aspect_ratio + self.max_aspect_ratio) / 2
        ratio_half_range = max((self.max_aspect_ratio - self.min_aspect_ratio) / 2, 0.001)
        ratio_score = max(0.0, 1.0 - abs(aspect_ratio - ratio_center) / ratio_half_range)
        area_score = min(area / max(frame_area * 0.10, 1.0), 1.0)
        geometry = (
            confidence * 0.35
            + ratio_score * 0.25
            + max(0.0, min(rectangularity, 1.0)) * 0.20
            + area_score * 0.20
        )
        weight = self.reading_zone_weight
        return round(max(0.0, min(1.0, geometry * (1.0 - weight) + zone_score * weight)), 3)

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
            zone_score = self._zone_score((x, y, width, height), frame_w, frame_h)
            selection_score = self._selection_score(confidence, aspect_ratio, rectangularity, area, zone_score, frame_area)
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
            candidates.append(PlateCandidate((x, y, width, height), confidence, approx, aspect_ratio, area, rectangularity, rotation, reason == "", reason, False, zone_score, selection_score))

        strict_valid = [c for c in candidates if c.valid]
        selected = max(strict_valid, key=lambda c: c.selection_score, default=None)

        # If strict filtering finds nothing, allow one near-valid candidate.
        # This is intentionally conservative: grossly small/large objects and
        # wildly wrong aspect ratios are still excluded from fallback selection.
        if selected is None:
            near_valid = [
                c for c in candidates
                if c.area >= self.min_area * 0.75
                and c.area <= self.max_area * 1.05
                and self.min_aspect_ratio - 0.35 <= c.aspect_ratio <= self.max_aspect_ratio + 0.35
                and c.rectangularity >= max(0.45, self.min_rectangularity - 0.35)
                and c.rotation <= self.max_rotation + 10.0
                and c.selection_score >= self.soft_selection_threshold
            ]
            selected = max(near_valid, key=lambda c: c.selection_score, default=None)
            if selected is not None:
                selected = PlateCandidate(
                    selected.bounding_box, selected.confidence, selected.contour,
                    selected.aspect_ratio, selected.area, selected.rectangularity,
                    selected.rotation, True, "soft_selected", False,
                    selected.zone_score, selected.selection_score,
                )

        if selected is not None:
            candidates = [
                PlateCandidate(
                    c.bounding_box, c.confidence, c.contour, c.aspect_ratio, c.area,
                    c.rectangularity, c.rotation, c.valid, c.rejected_reason,
                    c.bounding_box == selected.bounding_box,
                    c.zone_score, c.selection_score,
                )
                for c in candidates
            ]
            selected = next(c for c in candidates if c.selected)

        self.last_candidates = candidates
        if selected is None:
            return None
        return PlateDetection(
            selected.bounding_box, selected.confidence, selected.contour,
            selected.aspect_ratio, selected.area, selected.rectangularity,
            selected.rotation, tuple(candidates)
        )

    def crop(self, frame: Any, detection: PlateDetection) -> Any:
        x, y, width, height = detection.bounding_box
        return frame[y : y + height, x : x + width]
