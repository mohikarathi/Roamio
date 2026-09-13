"""Unit tests for hybrid scoring, MMR diversity re-ranking, and master RecommendationEngine."""

import pytest
from src.data.models import Destination, UserPreferences
from src.ranking.scorer import HybridScorer
from src.ranking.diversity import apply_mmr_diversity, calculate_intra_list_diversity
from src.ranking.engine import RecommendationEngine
from src.retrieval.two_stage import CandidateRecord


@pytest.fixture
def sample_destination():
    return Destination(
        destination_id="TEST-001",
        name="Kyoto",
        country="Japan",
        continent="Asia",
        latitude=35.0116,
        longitude=135.7681,
        category="Cultural",
        tags=["temples", "culture", "gardens"],
        description="Historical imperial capital with shrines.",
        cost_level="Medium",
        est_daily_cost_inr=5000.0,
        best_months=[3, 4, 5, 10, 11],
        best_seasons=["Spring", "Fall"],
        safety_rating="High"
    )


def test_budget_scoring(sample_destination):
    scorer = HybridScorer()

    # User with ₹70k budget for 7 days -> ₹10k/day (destination is ₹5k -> within budget)
    prefs_good = UserPreferences(budget_max_inr=70000, duration_days=7)
    score_good, reason_good = scorer.score_budget_compatibility(sample_destination, prefs_good)
    assert score_good >= 0.85
    assert "within budget" in reason_good.lower()

    # User with ₹15k budget for 7 days -> ~₹2k/day (destination is ₹5k -> over budget penalty)
    prefs_tight = UserPreferences(budget_max_inr=15000, duration_days=7)
    score_tight, reason_tight = scorer.score_budget_compatibility(sample_destination, prefs_tight)
    assert score_tight < 0.60
    assert "exceeds" in reason_tight.lower()


def test_seasonal_scoring(sample_destination):
    scorer = HybridScorer()

    # April (month 4) is in best_months
    prefs_april = UserPreferences(travel_month=4)
    score_apr, reason_apr = scorer.score_seasonal_compatibility(sample_destination, prefs_april)
    assert score_apr == 1.0
    assert "optimal" in reason_apr.lower()

    # January (month 1) is off-peak
    prefs_jan = UserPreferences(travel_month=1)
    score_jan, _ = scorer.score_seasonal_compatibility(sample_destination, prefs_jan)
    assert score_jan < 0.8


def test_mmr_diversity():
    # Create 3 near-identical destinations in Japan and 1 in Europe
    d1 = Destination(destination_id="D1", name="Kyoto", country="Japan", continent="Asia", latitude=35.0, longitude=135.7, category="Cultural", description="desc", est_daily_cost_inr=5000)
    d2 = Destination(destination_id="D2", name="Nara", country="Japan", continent="Asia", latitude=34.6, longitude=135.8, category="Cultural", description="desc", est_daily_cost_inr=5000)
    d3 = Destination(destination_id="D3", name="Osaka", country="Japan", continent="Asia", latitude=34.7, longitude=135.5, category="City", description="desc", est_daily_cost_inr=5000)
    d4 = Destination(destination_id="D4", name="Rome", country="Italy", continent="Europe", latitude=41.9, longitude=12.5, category="Cultural", description="desc", est_daily_cost_inr=7000)

    # Scored candidates where D1, D2, D3 have slightly higher relevance than D4
    candidates = [
        (d1, 0.95, None),
        (d2, 0.93, None),
        (d3, 0.91, None),
        (d4, 0.88, None)
    ]

    selected, ild = apply_mmr_diversity(candidates, top_k=3, lambda_param=0.5)
    selected_ids = [s[0].destination_id for s in selected]

    # D1 should be picked first (highest score)
    assert selected_ids[0] == "D1"
    # MMR should boost D4 because it provides geographical diversity from Rome (Italy vs Japan)
    assert "D4" in selected_ids
    assert ild > 0.4


def test_recommendation_engine_end_to_end():
    engine = RecommendationEngine()
    prefs = UserPreferences(
        query_text="peaceful cultural temples",
        budget_max_inr=60000,
        duration_days=7
    )
    resp = engine.recommend(prefs, top_k=5)
    assert len(resp.items) == 5
    assert resp.total_candidates_evaluated > 0
    assert resp.retrieval_latency_ms >= 0.0
    assert resp.ranking_latency_ms >= 0.0
    # Verify top result has grounded reasons
    top_item = resp.items[0]
    assert len(top_item.explanation.reasons) > 0
    assert top_item.rank == 1
