
import pytest

from evaluation.run_evaluation import (
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    ndcg_at_k,
    context_precision,
)


def test_recall_at_k():
    retrieved = ["a", "c"]
    relevant = {"a", "b"}

    assert recall_at_k(retrieved, relevant) == 0.5


def test_hit_rate_at_k():
    assert hit_rate_at_k(["a", "c"], {"a", "b"}) == 1.0
    assert hit_rate_at_k(["x", "y"], {"a", "b"}) == 0.0


def test_reciprocal_rank():
    retrieved = ["x", "b", "a"]
    relevant = {"a", "b"}

    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_reciprocal_rank_no_match():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_at_k():
    score = ndcg_at_k(["x", "a", "b"], {"a", "b"}, 3)

    assert score == pytest.approx(0.6934, abs=0.001)


def test_perfect_ndcg():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0


def test_context_precision():
    score = context_precision(["a", "x", "b"], {"a", "b"})

    assert score == pytest.approx(2 / 3)


def test_context_precision_empty_results():
    assert context_precision([], {"a"}) == 0.0


def test_recall_without_relevant_chunks():
    assert recall_at_k(["a", "b"], set()) == 0.0

