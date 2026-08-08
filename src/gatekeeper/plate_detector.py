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
    """Detect rectangular license-plate candidates inside a small reading ROI.

    The detector keeps the public API used by GateKeeper unchanged.  The new
    reading-zone stage is deliberately isolated here: the full camera frame is
    never modified, while contour detection operates only on the central ROI.
    Candidate coordinates are mapped back to full-frame coordinates before they
    are returned, so existing overlays, snapshots and APIs remain compatible.
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
        reading_zone_enabled: bool = True,
        reading_zone_width_ratio: float = 0.20,
        reading_zone_aspect_ratio: float = 4.7,
        reading_zone_scale: float = 3.0,
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
        self.reading_zone_enabled = reading_zone_enabled
        self.reading_zone_width_ratio = reading_zone_width_ratio
        self.reading_zone_aspect_ratio = reading_zone_aspect_ratio
        self.reading_zone_scale = reading_zone_scale
        self.last_candidates: list[PlateCandidate] = []
        self.last_debug_frames: dict[str, Any] = {}
        self.last_reading_zone: tuple[int, int, int, int] | None = None
        self._initialized = False
        self._cv2: Any | None = None

    def initialize(self) -> None:
        if self._initialized:
            return
        import cv2
        self._cv2 = cv2
        self._initialized = True

    def _reading_zone(self, frame: Any) -> tuple[int, int, int, int]:
        """Return the small central plate-reading ROI in full-frame coordinates."""
        height, width = frame.shape[:2]
        zone_width = max(64, int(width * self.reading_zone_width_ratio))
        zone_height = max(20, int(zone_width / self.reading_zone_aspect_ratio))
        x = max(0, (width - zone_width) // 2)
        y = max(0, (height - zone_height) // 2)
        return x, y, zone_width, zone_height

    def _map_candidate(self, candidate: PlateCandidate, offset_x: int, offset_y: int) -> PlateCandidate:
        x, y, w, h = candidate.bounding_box
        return PlateCandidate(
            (x + offset_x, y + offset_y, w, h),
            candidate.confidence,
            candidate.contour,
            candidate.aspect_ratio,
            candidate.area,
            candidate.rectangularity,
            candidate.rotation,
            candidate.valid,
            candidate.rejected_reason,
            candidate.selected,
        )

    def detect(self, frame: Any) -> PlateDetection | None:
        self.initialize()
        cv2 = self._cv2
        if cv2 is None:
            raise RuntimeError("PlateDetector is not initialized.")

        import sys
        camera_manager = getattr(sys.modules.get("main") or sys.modules.get("__main__"), "CameraManager")

        frame_h, frame_w = frame.shape[:2]
        if self.reading_zone_enabled:
            rx, ry, rw, rh = self._reading_zone(frame)
            self.last_reading_zone = (rx, ry, rw, rh)
            working = frame[ry:ry + rh, rx:rx + rw]
        else:
            rx, ry = 0, 0
            working = frame
            self.last_reading_zone = (0, 0, frame_w, frame_h)

        # The crop is enlarged before edge detection.  This improves separation
        # of the plate border without changing the original FRAME_MASTER.
        scale = max(1.0, float(self.reading_zone_scale))
        if scale != 1.0:
            working = cv2.resize(working, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        gray = camera_manager.to_gray(working)
        filtered = cv2.bilateralFilter(gray, 7, 45, 45)
        edged = cv2.Canny(filtered, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.last_debug_frames = {"gray": gray, "edges": edged, "contours": closed}

        candidates: list[PlateCandidate] = []
        working_h, working_w = working.shape[:2]
        working_area = float(working_h * working_w)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
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

            area_score = min(area / max(working_area * 0.45, 1.0), 1.0)
            ratio_center = (self.min_aspect_ratio + self.max_aspect_ratio) / 2
            ratio_range = max((self.max_aspect_ratio - self.min_aspect_ratio) / 2, 0.001)
            ratio_score = max(0.0, 1.0 - abs(aspect_ratio - ratio_center) / ratio_range)
            confidence = round(max(0.01, area_score * 0.35 + ratio_score * 0.45 + rectangularity * 0.20), 3)

            reason = ""
            if area < self.min_area:
                reason = "area_too_small"
            elif area > self.max_area:
                reason = "area_too_large"
            elif not self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                reason = "aspect_ratio"
            elif rectangularity < self.min_rectangularity:
                reason = "rectangularity"
            elif rotation > self.max_rotation:
                reason = "rotation"
            elif confidence < self.confidence_threshold:
                reason = "confidence"

            # Map geometry from the enlarged ROI back to the original frame.
            scale_x = float(rw / working_w) if False else scale
            full_x = int(round(x / scale_x)) + rx
            full_y = int(round(y / scale_x)) + ry
            full_w = max(1, int(round(width / scale_x)))
            full_h = max(1, int(round(height / scale_x)))
            candidates.append(
                PlateCandidate(
                    (full_x, full_y, full_w, full_h),
                    confidence,
                    approx,
                    aspect_ratio,
                    area / max(scale * scale, 1.0),
                    rectangularity,
                    rotation,
                    reason == "",
                    reason,
                )
            )

        # Prefer the candidate closest to the expected plate aspect ratio, then
        # confidence.  Only valid candidates can become selected.
        valid = [c for c in candidates if c.valid]
        selected = max(
            valid,
            key=lambda c: (c.confidence, -abs(c.aspect_ratio - self.reading_zone_aspect_ratio)),
            default=None,
        )
        if selected is not None:
            candidates = [
                PlateCandidate(
                    c.bounding_box,
                    c.confidence,
                    c.contour,
                    c.aspect_ratio,
                    c.area,
                    c.rectangularity,
                    c.rotation,
                    c.valid,
                    c.rejected_reason,
                    c.bounding_box == selected.bounding_box,
                )
                for c in candidates
            ]
            selected = next(c for c in candidates if c.selected)

        self.last_candidates = candidates
        if selected is None:
            return None
        return PlateDetection(
            selected.bounding_box,
            selected.confidence,
            selected.contour,
            selected.aspect_ratio,
            selected.area,
            selected.rectangularity,
            selected.rotation,
            tuple(candidates),
        )

    def crop(self, frame: Any, detection: PlateDetection) -> Any:
        x, y, width, height = detection.bounding_box
        return frame[y : y + height, x : x + width]
