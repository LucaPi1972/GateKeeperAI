"""Optional OCR preparation and comparison for the selected plate ROI.

OCR itself is intentionally not coupled to the live detector yet.  This module
provides a small, dependency-light interface that compares the prepared ROI
variants and can be used by the live preview without changing detection.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np


def _normalise_text(text: str) -> str:
    """Normalise OCR output for Italian-style plate comparison."""
    text = re.sub(r"[^A-Za-z0-9]", "", text or "").upper()
    return text


def build_ocr_variants(roi: np.ndarray) -> Dict[str, np.ndarray]:
    """Build deterministic variants from an already isolated plate ROI."""
    if roi is None or roi.size == 0:
        return {}

    if len(roi.shape) == 2:
        gray = roi.copy()
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    threshold = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 7,
    )

    return {
        "rectified": roi,
        "grayscale": gray,
        "enhanced": enhanced,
        "adaptive_threshold": threshold,
    }


def clean_ocr_text(text: str) -> str:
    """Return OCR text in a stable comparison form."""
    return _normalise_text(text)


def compare_ocr_outputs(outputs: Iterable[Optional[str]]) -> Dict[str, Any]:
    """Summarise agreement between OCR outputs without choosing an OCR engine."""
    values = [clean_ocr_text(value or "") for value in outputs]
    values = [value for value in values if value]
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return {"consensus": "", "agreement": 0.0, "outputs": []}
    consensus = max(counts, key=counts.get)
    return {
        "consensus": consensus,
        "agreement": counts[consensus] / len(values),
        "outputs": values,
    }
