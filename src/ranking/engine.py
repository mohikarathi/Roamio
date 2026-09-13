"""Master recommendation service combining two-stage retrieval, hybrid ranking, diversity, and explainability."""

import time
import logging
from typing import Optional, List

from src.config import DEFAULT_WEIGHTS, DEFAULT_DIVERSITY_LAMBDA
from src.data.models import (
    UserPreferences,
    RecommendationResponse,
    RecommendationItem
)
from src.retrieval.two_stage import TwoStageRetriever
from src.ranking.scorer import HybridScorer
from src.ranking.diversity import apply_mmr_diversity
from src.ranking.explain import ExplainabilityEngine

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Unified recommendation engine executing:
    1. Structured candidate filtering
    2. Dense semantic & TF-IDF candidate retrieval
    3. Multi-feature hybrid scoring
    4. Maximal Marginal Relevance (MMR) diversity re-ranking
    5. Evidence-based explainability generation
    """

    def __init__(
        self,
        retriever: Optional[TwoStageRetriever] = None,
        scorer: Optional[HybridScorer] = None,
        explainer: Optional[ExplainabilityEngine] = None,
        diversity_lambda: float = DEFAULT_DIVERSITY_LAMBDA
    ):
        self.retriever = retriever or TwoStageRetriever()
        self.scorer = scorer or HybridScorer(DEFAULT_WEIGHTS)
        self.explainer = explainer or ExplainabilityEngine(self.scorer)
        self.diversity_lambda = diversity_lambda

    def recommend(
        self,
        preferences: UserPreferences,
        top_k: int = 5,
        apply_diversity: bool = True
    ) -> RecommendationResponse:
        """
        Execute end-to-end recommendation workflow.
        Returns complete RecommendationResponse payload with latency and diversity metrics.
        """
        t0 = time.time()

        # Step 1: Two-Stage Candidate Retrieval
        candidates = self.retriever.retrieve_candidates(preferences, top_k_stage2=50)
        t_retrieval = (time.time() - t0) * 1000.0

        t1 = time.time()
        # Step 2: Hybrid Scoring
        scored_candidates = []
        for cand in candidates:
            breakdown = self.scorer.compute_scores(cand, preferences)
            scored_candidates.append((cand.destination, breakdown.final_score, (cand, breakdown)))

        # Sort candidates descending by hybrid score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Diversity Re-ranking (MMR)
        if apply_diversity and len(scored_candidates) > top_k:
            ranked_items, diversity_score = apply_mmr_diversity(
                scored_candidates,
                top_k=top_k,
                lambda_param=self.diversity_lambda
            )
        else:
            ranked_items = scored_candidates[:top_k]
            from src.ranking.diversity import calculate_intra_list_diversity
            diversity_score = calculate_intra_list_diversity([x[0] for x in ranked_items])

        # Step 4: Explainability Generation
        final_results: List[RecommendationItem] = []
        for rank_idx, (dest, score, (cand_rec, breakdown)) in enumerate(ranked_items, start=1):
            explanation = self.explainer.generate_explanation(dest, breakdown, preferences)
            final_results.append(RecommendationItem(
                rank=rank_idx,
                destination=dest,
                final_score=round(score, 4),
                dense_similarity=round(cand_rec.dense_similarity, 4),
                lexical_similarity=round(cand_rec.lexical_similarity, 4),
                explanation=explanation
            ))

        t_ranking = (time.time() - t1) * 1000.0

        return RecommendationResponse(
            items=final_results,
            total_candidates_evaluated=len(candidates),
            retrieval_latency_ms=round(t_retrieval, 2),
            ranking_latency_ms=round(t_ranking, 2),
            diversity_metric=round(diversity_score, 3),
            applied_preferences=preferences
        )
