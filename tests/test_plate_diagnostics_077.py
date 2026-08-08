from __future__ import annotations

from src.gatekeeper.plate_detector import PlateCandidate, PlateDetector


def test_candidate_metadata_exposes_077_diagnostic_fields():
    candidate = PlateCandidate(
        (10, 20, 300, 70),
        0.6,
        object(),
        4.3,
        21000.0,
        0.9,
        2.0,
        True,
        selected=True,
        zone_score=0.8,
        selection_score=0.72,
        stability_score=0.95,
        selection_margin=0.18,
        selection_mode="strict",
        stable_selection=True,
    )

    metadata = candidate.metadata()

    assert metadata["selection_margin"] == 0.18
    assert metadata["selection_mode"] == "strict"
    assert metadata["stable_selection"] is True


def test_detector_diagnostics_start_empty_and_report_rates():
    detector = PlateDetector()

    stats = detector.diagnostics_snapshot()

    assert stats["frames_seen"] == 0
    assert stats["frames_selected"] == 0
    assert stats["selection_rate"] == 0.0
    assert stats["stable_selection_rate"] == 0.0
    assert stats["average_selection_margin"] == 0.0


def test_detector_diagnostics_count_selected_modes(monkeypatch):
    detector = PlateDetector()
    detector._frames_seen = 10
    detector._frames_with_candidates = 8
    detector._frames_selected = 7
    detector._strict_selected = 5
    detector._soft_selected = 2
    detector._stable_selected = 4
    detector._frames_without_selection = 3
    detector._candidate_total = 40
    detector._selection_score_total = 4.2
    detector._selection_margin_total = 0.7

    stats = detector.diagnostics_snapshot()

    assert stats["candidate_average"] == 4.0
    assert stats["selection_rate"] == 0.7
    assert stats["stable_selection_rate"] == round(4 / 7, 3)
    assert stats["average_selection_score"] == 0.6
    assert stats["average_selection_margin"] == 0.1
