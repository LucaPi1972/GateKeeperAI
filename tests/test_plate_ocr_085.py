from src.plate_ocr import context_correct, format_score, score_ocr_candidate, fuse_temporal_results


def test_context_corrects_numeric_positions_only():
    assert context_correct("FT4Z7ST") == "FT427ST"


def test_context_corrects_letter_positions_only():
    assert context_correct("0T42751") == "OT427SI"


def test_valid_plate_format_scores_higher_than_wrong_shape():
    assert format_score("FT427ST") > format_score("FT42ST")
    assert format_score("FT427ST") == 1.0


def test_candidate_score_uses_engine_confidence_and_format():
    result = score_ocr_candidate("FT427ST", engine_confidence=92.0)
    assert result["corrected"] == "FT427ST"
    assert result["valid_length"] is True
    assert result["format_score"] == 1.0
    assert result["score"] > 0.9


def test_temporal_fusion_prefers_repeated_good_reading():
    results = [
        {"text": "FT4275T", "score": 0.62},
        {"text": "FT427ST", "score": 0.92},
        {"text": "FT427ST", "score": 0.90},
        {"text": "FT427ST", "score": 0.91},
    ]
    fused = fuse_temporal_results(results)
    assert fused["text"] == "FT427ST"
    assert fused["agreement"] == 0.75
    assert fused["confidence"] > 70.0
