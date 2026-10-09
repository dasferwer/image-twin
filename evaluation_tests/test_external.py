import runpy
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/evaluate_external.py"


def test_false_positives_and_misses_have_separate_denominators():
    metrics = runpy.run_path(str(MODULE))["summarize"](
        [
            {
                "positive": True,
                "candidate_hit": True,
                "matches": 2,
                "true_match": True,
                "candidates": 5,
                "latency_ms": 10,
            },
            {
                "positive": True,
                "candidate_hit": False,
                "matches": 0,
                "true_match": False,
                "candidates": 5,
                "latency_ms": 20,
            },
            {
                "positive": False,
                "candidate_hit": False,
                "matches": 1,
                "true_match": False,
                "candidates": 5,
                "latency_ms": 30,
            },
        ]
    )
    assert metrics["candidate_recall"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["false_positives"] == 2
    assert metrics["negative_query_false_positive_rate"] == 1
    assert metrics["verified_candidate_false_positive_rate"] == pytest.approx(2 / 14)


def test_no_predictions_does_not_claim_perfect_precision():
    metrics = runpy.run_path(str(MODULE))["summarize"]([])
    assert metrics["precision"] is None
    assert metrics["candidate_recall"] is None


def test_corrupted_external_cache_is_rejected_before_decode(tmp_path):
    module = runpy.run_path(str(MODULE))
    (tmp_path / "t10k-images-idx3-ubyte.gz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Dataset checksum mismatch"):
        module["corpus"](tmp_path)
