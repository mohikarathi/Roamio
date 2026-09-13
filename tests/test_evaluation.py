"""Unit tests for information retrieval and recommendation metrics."""

import pytest
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    catalog_coverage
)


def test_precision_at_k():
    recommended = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "F"}
    # Top 3: A (hit), B (miss), C (hit) -> 2 / 3 = 0.6667
    assert round(precision_at_k(recommended, relevant, k=3), 4) == 0.6667
    # Top 5: A (hit), B, C (hit), D, E -> 2 / 5 = 0.4
    assert precision_at_k(recommended, relevant, k=5) == 0.4
    # Empty
    assert precision_at_k([], relevant, k=5) == 0.0


def test_recall_at_k():
    recommended = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "F", "G"} # 4 total relevant
    # Top 3 retrieves A and C (2 of 4) -> 0.5
    assert recall_at_k(recommended, relevant, k=3) == 0.5
    # Empty
    assert recall_at_k([], relevant, k=5) == 0.0


def test_ndcg_at_k():
    recommended = ["A", "B", "C"]
    relevance_scores = {"A": 1.0, "B": 0.5, "C": 0.0}
    # Ideal ranking is A, B, C -> NDCG should be 1.0
    ndcg = ndcg_at_k(recommended, relevance_scores, k=3)
    assert round(ndcg, 4) == 1.0

    # Sub-optimal ranking: B, A, C
    sub_optimal = ["B", "A", "C"]
    ndcg_sub = ndcg_at_k(sub_optimal, relevance_scores, k=3)
    assert ndcg_sub < 1.0 and ndcg_sub > 0.0


def test_mrr():
    # First relevant item at rank 1
    assert mean_reciprocal_rank(["A", "B", "C"], {"A"}) == 1.0
    # First relevant item at rank 2
    assert mean_reciprocal_rank(["X", "A", "C"], {"A"}) == 0.5
    # No relevant item
    assert mean_reciprocal_rank(["X", "Y", "Z"], {"A"}) == 0.0


def test_catalog_coverage():
    all_rec = {"A", "B", "C", "D"}
    assert catalog_coverage(all_rec, 100) == 0.04
