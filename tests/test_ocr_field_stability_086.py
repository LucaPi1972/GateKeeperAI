from src.plate_ocr import temporal_stability


def test_temporal_stability_requires_multiple_consistent_samples():
    assert temporal_stability({"text": "FT427ST", "samples": 1, "agreement": 1.0, "confidence": 95.0}) is False
    assert temporal_stability({"text": "FT427ST", "samples": 3, "agreement": 0.67, "confidence": 70.0}) is True


def test_temporal_stability_rejects_low_agreement():
    assert temporal_stability({"text": "FT427ST", "samples": 6, "agreement": 0.50, "confidence": 90.0}) is False


def test_temporal_stability_rejects_low_confidence():
    assert temporal_stability({"text": "FT427ST", "samples": 6, "agreement": 0.83, "confidence": 69.9}) is False
