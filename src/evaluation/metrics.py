"""Evaluation metrics for information retrieval and recommender systems."""

import math
from typing import List, Set, Dict


def precision_at_k(recommended: List[str], relevant: Set[str], k: int = 5) -> float:
    """Calculate Precision@K: proportion of top-k recommended items that are relevant."""
    if k <= 0 or not recommended:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: List[str], relevant: Set[str], k: int = 10) -> float:
    """Calculate Recall@K: proportion of all relevant items retrieved in top-k."""
    if not relevant or k <= 0 or not recommended:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def dcg_at_k(recommended: List[str], relevance_scores: Dict[str, float], k: int = 10) -> float:
    """Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    for idx, item in enumerate(recommended[:k]):
        rel = relevance_scores.get(item, 0.0)
        # DCG formula: rel_i / log2(i + 2)
        dcg += rel / math.log2(idx + 2)
    return dcg


def ndcg_at_k(recommended: List[str], relevance_scores: Dict[str, float], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    actual_dcg = dcg_at_k(recommended, relevance_scores, k)
    # Ideal DCG: sort relevance scores descending
    ideal_scores = sorted(relevance_scores.values(), reverse=True)
    idcg = sum(score / math.log2(idx + 2) for idx, score in enumerate(ideal_scores[:k]))
    if idcg <= 0.0:
        return 0.0
    return actual_dcg / idcg


def mean_reciprocal_rank(recommended: List[str], relevant: Set[str]) -> float:
    """Calculate Reciprocal Rank (RR): reciprocal of the rank of the first relevant item."""
    for idx, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1.0 / idx
    return 0.0


def catalog_coverage(all_recommended_ids: Set[str], total_catalog_size: int) -> float:
    """Percentage of unique catalog items recommended across all benchmark queries."""
    if total_catalog_size <= 0:
        return 0.0
    return len(all_recommended_ids) / total_catalog_size
