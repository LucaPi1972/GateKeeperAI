"""Lightweight OpenCV license plate candidate detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import logging
import time


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
    stability_score: float = 0.0
    selection_margin: float = 0.0
    selection_mode: str = "none"
    stable_selection: bool = False

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
            "stability_score": self.stability_score,
            "selection_margin": self.selection_margin,
            "selection_mode": self.selection_mode,
            "stable_selection": self.stable_selection,
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
    selection_score: float = 0.0
    zone_score: float = 0.0
    stability_score: float = 0.0
    selection_margin: float = 0.0
    selection_mode: str = "strict"
    stable_selection: bool = False


class PlateDetector:
    """Detect rectangular license plate candidates without OCR.

    Release 0.7.7 keeps the 0.7.6 detector behavior unchanged and adds
    measurement-only diagnostics. The diagnostics count frames, candidates,
    selections, soft selections, stable selections and score margins so field
    tests can quantify whether subsequent detector changes really improve the
    result. No threshold or selection rule is changed by the diagnostics.
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
        stability_window: float = 0.08,
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
        self.stability_window = max(0.0, min(float(stability_window), 0.5))
        self.last_candidates: list[PlateCandidate] = []
        self.last_debug_frames: dict[str, Any] = {}
        self.last_selection_margin = 0.0
        self.last_selection_mode = "none"
        self.last_stable_selection = False
        self._previous_selected_bbox: tuple[int, int, int, int] | None = None
        self._initialized = False
        self._cv2: Any | None = None
        self._logger = logging.getLogger("gatekeeper.plate_detector")
        self._diagnostics_started_at = time.monotonic()
        self._frames_seen = 0
        self._frames_with_candidates = 0
        self._frames_selected = 0
        self._strict_selected = 0
        self._soft_selected = 0
        self._stable_selected = 0
        self._frames_without_selection = 0
        self._candidate_total = 0
        self._selection_score_total = 0.0
        self._selection_margin_total = 0.0
        self._last_diagnostics_log = 0.0

    def initialize(self) -> None:
        if self._initialized:
            return
        import cv2
        self._cv2 = cv2
        self._initialized = True

    @staticmethod
    def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        ax1, ay1, aw, ah = first
        bx1, by1, bw, bh = second
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        intersection = float(iw * ih)
        union = float(aw * ah + bw * bh) - intersection
        return intersection / union if union > 0 else 0.0

    def _stability_score(self, bbox: tuple[int, int, int, int]) -> float:
        """Return 0..1 overlap with the previously selected candidate."""
        if self._previous_selected_bbox is None:
            return 0.5
        return round(self._iou(bbox, self._previous_selected_bbox), 3)

    def _zone_score(self, bbox: tuple[int, int, int, int], frame_w: int, frame_h: int) -> float:
        """Return 0..1 proximity to the small central reading zone."""
        if self.reading_zone:
            zone = self.reading_zone
            x = float(zone.get("x", 0))
            y = float(zone.get("y", 0))
            w = max(float(zone.get("width", 0)), 1.0)
            h = max(float(zone.get("height", 0)), 1.0)
        else:
            w = frame_w * 0.20
            h = frame_h * (70.0 / 1232.0)
            x = (frame_w - w) / 2.0
            y = (frame_h - h) / 2.0
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        zone_cx = x + w / 2.0
        zone_cy = y + h / 2.0
        dx = abs(cx - zone_cx) / max(w / 2.0, 1.0)
        dy = abs(cy - zone_cy) / max(h / 2.0, 1.0)
        return round(max(0.0, 1.0 - max(dx, dy)), 3)

    def _selection_score(
        self,
        confidence: float,
        aspect_ratio: float,
        rectangularity: float,
        area: float,
        zone_score: float,
        stability_score: float,
        frame_area: float,
    ) -> float:
        """Combine geometry quality with small position/stability preferences."""
        ratio_center = (self.min_aspect_ratio + self.max_aspect_ratio) / 2
        ratio_half_range = max((self.max_aspect_ratio - self.min_aspect_ratio) / 2, 0.001)
        ratio_score = max(0.0, 1.0 - abs(aspect_ratio - ratio_center) / ratio_half_range)
        area_score = min(area / max(frame_area * 0.10, 1.0), 1.0)
        geometry = (
            confidence * 0.30
            + ratio_score * 0.25
            + max(0.0, min(rectangularity, 1.0)) * 0.20
            + area_score * 0.25
        )
        position_and_stability = zone_score * self.reading_zone_weight
        stability_weight = 0.10
        return round(max(0.0, min(1.0, geometry * (1.0 - self.reading_zone_weight - stability_weight) + position_and_stability + stability_score * stability_weight)), 3)

    def _prefer_previous_candidate(self, ranked: list[PlateCandidate]) -> PlateCandidate | None:
        """Keep the previous plate when a nearby candidate remains competitive."""
        if self._previous_selected_bbox is None or not ranked:
            return None
        previous = max(
            ranked,
            key=lambda candidate: self._iou(candidate.bounding_box, self._previous_selected_bbox),
            default=None,
        )
        if previous is None:
            return None
        overlap = self._iou(previous.bounding_box, self._previous_selected_bbox)
        if overlap < 0.25:
            return None
        leader = ranked[0]
        if previous.selection_score >= leader.selection_score - self.stability_window:
            return previous
        return None

    def diagnostics_snapshot(self) -> dict[str, Any]:
        """Return measurement-only detector statistics for calibration tests."""
        elapsed = max(time.monotonic() - self._diagnostics_started_at, 0.001)
        frames = self._frames_seen
        return {
            "frames_seen": frames,
            "frames_with_candidates": self._frames_with_candidates,
            "frames_selected": self._frames_selected,
            "strict_selected": self._strict_selected,
            "soft_selected": self._soft_selected,
            "stable_selected": self._stable_selected,
            "frames_without_selection": self._frames_without_selection,
            "candidate_total": self._candidate_total,
            "candidate_average": round(self._candidate_total / max(frames, 1), 2),
            "selection_rate": round(self._frames_selected / max(frames, 1), 3),
            "stable_selection_rate": round(self._stable_selected / max(self._frames_selected, 1), 3),
            "average_selection_score": round(self._selection_score_total / max(self._frames_selected, 1), 3),
            "average_selection_margin": round(self._selection_margin_total / max(self._frames_selected, 1), 3),
            "elapsed_seconds": round(elapsed, 1),
            "last_selection_mode": self.last_selection_mode,
            "last_stable_selection": self.last_stable_selection,
            "last_selection_margin": self.last_selection_margin,
        }

    def _log_diagnostics(self) -> None:
        now = time.monotonic()
        if now - self._last_diagnostics_log < 10.0:
            return
        self._last_diagnostics_log = now
        self._logger.info("PLATE DIAGNOSTICS %s", self.diagnostics_snapshot())

    def detect(self, frame: Any) -> PlateDetection | None:
        self.initialize()
        cv2 = self._cv2
        if cv2 is None:
            raise RuntimeError("PlateDetector is not initialized.")

        self._frames_seen += 1
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
            stability_score = self._stability_score((x, y, width, height))
            selection_score = self._selection_score(confidence, aspect_ratio, rectangularity, area, zone_score, stability_score, frame_area)
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
            candidates.append(PlateCandidate((x, y, width, height), confidence, approx, aspect_ratio, area, rectangularity, rotation, reason == "", reason, False, zone_score, selection_score, stability_score))

        self._candidate_total += len(candidates)
        if candidates:
            self._frames_with_candidates += 1

        strict_valid = sorted((c for c in candidates if c.valid), key=lambda c: c.selection_score, reverse=True)
        previous_selected = self._prefer_previous_candidate(strict_valid)
        selected = previous_selected or (strict_valid[0] if strict_valid else None)
        selection_mode = "strict" if selected is not None else "none"
        stable_selection = previous_selected is not None

        if selected is None:
            near_valid = sorted(
                [
                    c for c in candidates
                    if c.area >= self.min_area * 0.75
                    and c.area <= self.max_area * 1.05
                    and self.min_aspect_ratio - 0.35 <= c.aspect_ratio <= self.max_aspect_ratio + 0.35
                    and c.rectangularity >= max(0.45, self.min_rectangularity - 0.35)
                    and c.rotation <= self.max_rotation + 10.0
                    and c.selection_score >= self.soft_selection_threshold
                ],
                key=lambda c: c.selection_score,
                reverse=True,
            )
            previous_selected = self._prefer_previous_candidate(near_valid)
            selected = previous_selected or (near_valid[0] if near_valid else None)
            selection_mode = "soft" if selected is not None else "none"
            stable_selection = previous_selected is not None

        ranked = strict_valid if strict_valid else sorted(candidates, key=lambda c: c.selection_score, reverse=True)
        top_score = ranked[0].selection_score if ranked else 0.0
        second_score = ranked[1].selection_score if len(ranked) > 1 else 0.0
        selection_margin = round(max(0.0, top_score - second_score), 3)
        self.last_selection_margin = selection_margin
        self.last_selection_mode = selection_mode
        self.last_stable_selection = stable_selection

        if selected is not None:
            self._frames_selected += 1
            self._strict_selected += 1 if selection_mode == "strict" else 0
            self._soft_selected += 1 if selection_mode == "soft" else 0
            self._stable_selected += 1 if stable_selection else 0
            self._selection_score_total += selected.selection_score
            self._selection_margin_total += selection_margin
            candidates = [
                PlateCandidate(
                    c.bounding_box, c.confidence, c.contour, c.aspect_ratio, c.area,
                    c.rectangularity, c.rotation, c.valid, c.rejected_reason,
                    c.bounding_box == selected.bounding_box,
                    c.zone_score, c.selection_score, c.stability_score,
                    selection_margin, selection_mode, stable_selection,
                )
                for c in candidates
            ]
            selected = next(c for c in candidates if c.selected)
            self._previous_selected_bbox = selected.bounding_box
        else:
            self._frames_without_selection += 1
            self._previous_selected_bbox = None
            candidates = [
                PlateCandidate(
                    c.bounding_box, c.confidence, c.contour, c.aspect_ratio, c.area,
                    c.rectangularity, c.rotation, c.valid, c.rejected_reason,
                    False, c.zone_score, c.selection_score, c.stability_score,
                    selection_margin, selection_mode, False,
                )
                for c in candidates
            ]

        self.last_candidates = candidates
        self._log_diagnostics()
        if selected is None:
            return None
        return PlateDetection(
            selected.bounding_box, selected.confidence, selected.contour,
            selected.aspect_ratio, selected.area, selected.rectangularity,
            selected.rotation, tuple(candidates), selected.selection_score,
            selected.zone_score, selected.stability_score, selection_margin,
            selection_mode, stable_selection,
        )

    def crop(self, frame: Any, detection: PlateDetection) -> Any:
        x, y, width, height = detection.bounding_box
        return frame[y : y + height, x : x + width]
