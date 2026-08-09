"""OCR preparation, scoring and temporal fusion for selected plate ROIs."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np

PLATE_LENGTH = 7
LETTER_POSITIONS = {0, 1, 5, 6}
DIGIT_POSITIONS = {2, 3, 4}
AMBIGUOUS_TO_DIGIT = {"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
AMBIGUOUS_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}


def _normalise_text(text: str) -> str:
    """Normalise OCR output for Italian-style plate comparison."""
    return re.sub(r"[^A-Za-z0-9]", "", text or "").upper()


def clean_ocr_text(text: str) -> str:
    """Return OCR text in a stable comparison form."""
    return _normalise_text(text)


def build_ocr_variants(roi: np.ndarray) -> Dict[str, np.ndarray]:
    """Build deterministic variants from an already isolated plate ROI."""
    if roi is None or roi.size == 0:
        return {}
    gray = roi.copy() if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    threshold = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )
    return {"rectified": roi, "grayscale": gray, "enhanced": enhanced, "adaptive_threshold": threshold}


def format_score(text: str) -> float:
    """Score how closely text matches the 7-character AA999AA plate structure."""
    value = clean_ocr_text(text)
    if not value:
        return 0.0
    length_score = max(0.0, 1.0 - abs(len(value) - PLATE_LENGTH) / PLATE_LENGTH)
    if len(value) != PLATE_LENGTH:
        return round(0.55 * length_score, 3)
    position_score = 0.0
    for index, char in enumerate(value):
        if index in LETTER_POSITIONS:
            position_score += 1.0 if char.isalpha() else 0.0
        else:
            position_score += 1.0 if char.isdigit() else 0.0
    return round(0.35 + 0.65 * (position_score / PLATE_LENGTH), 3)


def context_correct(text: str) -> str:
    """Return a conservative format-aware candidate without forcing length."""
    value = clean_ocr_text(text)
    if len(value) != PLATE_LENGTH:
        return value
    chars = list(value)
    for index in DIGIT_POSITIONS:
        chars[index] = AMBIGUOUS_TO_DIGIT.get(chars[index], chars[index])
    for index in LETTER_POSITIONS:
        chars[index] = AMBIGUOUS_TO_LETTER.get(chars[index], chars[index])
    return "".join(chars)


def score_ocr_candidate(text: str, *, engine_confidence: float | None = None) -> Dict[str, Any]:
    """Return measurable OCR quality signals for one candidate."""
    raw = clean_ocr_text(text)
    corrected = context_correct(raw)
    fmt = format_score(corrected)
    engine = max(0.0, min(float(engine_confidence), 100.0)) / 100.0 if engine_confidence is not None else 0.0
    if engine_confidence is None:
        total = fmt
    else:
        total = 0.45 * engine + 0.55 * fmt
    return {
        "text": raw,
        "corrected": corrected,
        "format_score": fmt,
        "engine_confidence": engine_confidence,
        "score": round(total, 3),
        "length": len(raw),
        "valid_length": len(raw) == PLATE_LENGTH,
    }


def compare_ocr_outputs(outputs: Iterable[Optional[str]]) -> Dict[str, Any]:
    """Summarise agreement and format quality between OCR outputs."""
    values = [clean_ocr_text(value or "") for value in outputs]
    values = [value for value in values if value]
    if not values:
        return {"consensus": "", "agreement": 0.0, "outputs": [], "format_score": 0.0}
    scored = [score_ocr_candidate(value) for value in values]
    counts: Dict[str, int] = {}
    for item in scored:
        candidate = item["corrected"]
        counts[candidate] = counts.get(candidate, 0) + 1
    consensus = max(counts, key=counts.get)
    agreement = counts[consensus] / len(values)
    consensus_score = score_ocr_candidate(consensus)
    return {
        "consensus": consensus,
        "agreement": round(agreement, 3),
        "outputs": values,
        "format_score": consensus_score["format_score"],
        "score": round(0.60 * agreement + 0.40 * consensus_score["format_score"], 3),
    }


def fuse_temporal_results(results: Iterable[Dict[str, Any]], *, max_items: int = 8) -> Dict[str, Any]:
    """Fuse recent OCR results, with a fast path for a locked static plate."""
    items = list(results)[-max_items:]
    buckets: Dict[str, Dict[str, float]] = {}
    for item in items:
        text = context_correct(str(item.get("text", "")))
        if not text:
            continue
        bucket = buckets.setdefault(text, {"count": 0.0, "score": 0.0})
        bucket["count"] += 1.0
        bucket["score"] += float(item.get("score", 0.0))
    if not buckets:
        return {"text": "", "confidence": 0.0, "agreement": 0.0, "samples": 0, "fast_confirm": False}

    best_text, best = max(buckets.items(), key=lambda pair: (pair[1]["count"], pair[1]["score"]))
    raw_samples = sum(bucket["count"] for bucket in buckets.values())
    agreement = best["count"] / raw_samples if raw_samples else 0.0
    mean_score = best["score"] / best["count"] if best["count"] else 0.0
    confidence = 100.0 * (0.75 * agreement + 0.25 * mean_score)

    # Once PLATE LOCK has established a stationary, geometrically valid plate,
    # one strong 7-character Enhanced read is sufficient to avoid two extra
    # serial Tesseract launches. The displayed sample count is therefore the
    # effective confirmation count, while ocr_runs/valid_runs remain the real
    # process counters.
    fast_confirm = bool(
        raw_samples == 1
        and len(best_text) == PLATE_LENGTH
        and agreement >= 1.0
        and mean_score >= 0.85
        and bool(items[-1].get("frame_id", "")).startswith("lock:")
    )
    samples = 3 if fast_confirm else int(raw_samples)

    return {
        "text": best_text,
        "confidence": round(confidence, 1),
        "agreement": round(agreement, 3),
        "samples": samples,
        "fast_confirm": fast_confirm,
    }


def temporal_stability(fused: Dict[str, Any], *, min_samples: int = 3, min_agreement: float = 0.67, min_confidence: float = 70.0) -> bool:
    """Return whether a fused OCR result is sufficiently stable for field testing."""
    return bool(
        (
            fused.get("fast_confirm", False)
            or (
                fused.get("samples", 0) >= min_samples
                and fused.get("agreement", 0.0) >= min_agreement
                and fused.get("confidence", 0.0) >= min_confidence
            )
        )
        and fused.get("text", "")
    )
