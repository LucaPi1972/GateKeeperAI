from __future__ import annotations

import numpy as np

from src.gatekeeper.plate_detector import PlateCandidate, PlateDetector


def test_candidate_metadata_exposes_stability_score():
    candidate = PlateCandidate(
        (10, 20, 300, 70), 0.5, object(), 4.3, 21000.0, 0.9, 2.0, True,
        zone_score=0.8, selection_score=0.6, stability_score=0.9,
    )

    metadata = candidate.metadata()

    assert metadata["zone_score"] == 0.8
    assert metadata["selection_score"] == 0.6
    assert metadata["stability_score"] == 0.9


def test_detector_exposes_selection_margin_and_mode():
    detector = PlateDetector(min_area=100, confidence_threshold=0.99)
    frame = np.zeros((160, 320, 3), dtype=np.uint8)

    detection = detector.detect(frame)

    assert detection is None
    assert detector.last_selection_mode == "none"
    assert detector.last_selection_margin == 0.0


def test_detector_keeps_previous_candidate_when_scores_are_close(monkeypatch):
    detector = PlateDetector(min_area=100)
    first = (80, 60, 160, 40)
    second = (82, 61, 160, 40)
    detector._previous_selected_bbox = first

    previous = PlateCandidate(first, 0.5, object(), 4.0, 6400.0, 0.85, 0.0, True, selection_score=0.50)
    leader = PlateCandidate(second, 0.5, object(), 4.0, 6400.0, 0.85, 0.0, True, selection_score=0.54)

    selected = detector._prefer_previous_candidate([leader, previous])

    assert selected is previous


def test_detector_does_not_prefer_previous_when_score_gap_is_large():
    detector = PlateDetector(min_area=100)
    first = (80, 60, 160, 40)
    previous = PlateCandidate(first, 0.5, object(), 4.0, 6400.0, 0.85, 0.0, True, selection_score=0.40)
    leader = PlateCandidate((300, 60, 160, 40), 0.5, object(), 4.0, 6400.0, 0.85, 0.0, True, selection_score=0.60)
    detector._previous_selected_bbox = first

    selected = detector._prefer_previous_candidate([leader, previous])

    assert selected is None
