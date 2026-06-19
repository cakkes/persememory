import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ochat


def test_cosine_similarity_identical_vectors_is_one():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(ochat.cosine_similarity(a, a) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(ochat.cosine_similarity(a, b)) < 1e-6


def _fact(id_, text, vec):
    return {
        "id": id_,
        "text": text,
        "embedding": np.array(vec, dtype=np.float32),
        "source_thread": "default",
        "created_at": "2026-06-19T00:00:00+00:00",
    }


def test_top_k_facts_orders_by_similarity_and_respects_cutoff():
    query = np.array([1.0, 0.0], dtype=np.float32)
    facts = [
        _fact(1, "close match", [0.9, 0.1]),
        _fact(2, "exact match", [1.0, 0.0]),
        _fact(3, "unrelated", [0.0, 1.0]),  # similarity 0.0, below cutoff
    ]
    result = ochat.top_k_facts(query, facts, k=2, min_similarity=0.45)
    assert [f["id"] for f in result] == [2, 1]


def test_top_k_facts_returns_fewer_than_k_when_few_qualify():
    query = np.array([1.0, 0.0], dtype=np.float32)
    facts = [_fact(1, "only match", [1.0, 0.0])]
    result = ochat.top_k_facts(query, facts, k=8, min_similarity=0.45)
    assert len(result) == 1


def test_is_duplicate_fact_true_when_above_threshold():
    candidate = np.array([1.0, 0.0], dtype=np.float32)
    existing = [np.array([0.99, 0.01], dtype=np.float32)]
    assert ochat.is_duplicate_fact(candidate, existing, threshold=0.92) is True


def test_is_duplicate_fact_false_when_below_threshold():
    candidate = np.array([1.0, 0.0], dtype=np.float32)
    existing = [np.array([0.5, 0.5], dtype=np.float32)]
    assert ochat.is_duplicate_fact(candidate, existing, threshold=0.92) is False


def test_is_duplicate_fact_false_when_no_existing_facts():
    candidate = np.array([1.0, 0.0], dtype=np.float32)
    assert ochat.is_duplicate_fact(candidate, [], threshold=0.92) is False
