"""Diversity re-ranking using Maximal Marginal Relevance (MMR) and intra-list diversity metrics."""

import math
from typing import List, Tuple, Any
from src.data.models import Destination
from src.data.entity_resolution import haversine_distance_km


def inter_destination_similarity(dest1: Destination, dest2: Destination) -> float:
    """
    Compute pairwise similarity between two destinations based on geography, country, and category.
    Returns value in [0.0, 1.0] where 1.0 means practically identical location.
    """
    if dest1.destination_id == dest2.destination_id:
        return 1.0

    # 1. Geographic distance similarity
    dist_km = haversine_distance_km(dest1.latitude, dest1.longitude, dest2.latitude, dest2.longitude)
    if dist_km < 30.0:
        geo_sim = 1.0
    elif dist_km > 1000.0:
        geo_sim = 0.0
    else:
        # Linear decay between 30km and 1000km
        geo_sim = max(0.0, 1.0 - (dist_km - 30.0) / 970.0)

    # 2. Country match
    country_sim = 1.0 if dest1.country.lower() == dest2.country.lower() else 0.0

    # 3. Category match
    cat_sim = 1.0 if dest1.category.lower() == dest2.category.lower() else 0.0

    # Composite pairwise similarity
    return (0.50 * geo_sim) + (0.30 * country_sim) + (0.20 * cat_sim)


def calculate_intra_list_diversity(destinations: List[Destination]) -> float:
    """
    Calculate Intra-List Diversity (ILD) score [0.0 - 1.0].
    Higher means more diverse recommendations across geography and categories.
    """
    k = len(destinations)
    if k <= 1:
        return 1.0

    total_dist = 0.0
    pairs_count = 0

    for i in range(k):
        for j in range(i + 1, k):
            sim = inter_destination_similarity(destinations[i], destinations[j])
            total_dist += (1.0 - sim)
            pairs_count += 1

    return total_dist / pairs_count if pairs_count > 0 else 1.0


def apply_mmr_diversity(
    candidates_with_scores: List[Tuple[Destination, float, Any]],
    top_k: int = 5,
    lambda_param: float = 0.75
) -> Tuple[List[Tuple[Destination, float, Any]], float]:
    """
    Apply Maximal Marginal Relevance (MMR) re-ranking.
    lambda_param = 1.0 -> purely relevance ranking
    lambda_param = 0.0 -> maximal diversity ranking
    Default lambda_param = 0.75 provides optimal balance between high relevance and varied locations.
    """
    if not candidates_with_scores:
        return [], 0.0

    if len(candidates_with_scores) <= top_k:
        ild = calculate_intra_list_diversity([c[0] for c in candidates_with_scores])
        return candidates_with_scores, ild

    selected: List[Tuple[Destination, float, Any]] = []
    remaining = list(candidates_with_scores)

    # Pick the highest scoring item first
    first_item = remaining.pop(0)
    selected.append(first_item)

    while len(selected) < top_k and remaining:
        best_idx = -1
        best_mmr_score = -float("inf")

        for idx, (cand_dest, cand_score, cand_meta) in enumerate(remaining):
            # Calculate maximum similarity to already selected items
            max_sim = max(
                inter_destination_similarity(cand_dest, sel_dest)
                for sel_dest, _, _ in selected
            )

            # MMR formula
            mmr = (lambda_param * cand_score) - ((1.0 - lambda_param) * max_sim)

            if mmr > best_mmr_score:
                best_mmr_score = mmr
                best_idx = idx

        selected.append(remaining.pop(best_idx))

    ild = calculate_intra_list_diversity([c[0] for c in selected])
    return selected, ild
