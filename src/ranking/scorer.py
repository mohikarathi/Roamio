"""Multi-signal hybrid ranking engine combining semantic, lexical, preference, budget, seasonal, and quality features."""

import math
from typing import Dict, Any, List, Tuple
from src.config import DEFAULT_WEIGHTS
from src.data.models import Destination, UserPreferences
from src.retrieval.two_stage import CandidateRecord


class FeatureBreakdown:
    """Stores individual normalized feature scores and weighted contributions."""
    def __init__(self):
        self.raw_features: Dict[str, float] = {}
        self.weighted_contributions: Dict[str, float] = {}
        self.final_score: float = 0.0


class HybridScorer:
    """
    Computes principled multi-feature scores for candidate destinations.
    All feature signals are normalized to [0.0, 1.0] before weighting.
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        # Normalize weights so sum = 1.0
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

    def score_budget_compatibility(self, dest: Destination, prefs: UserPreferences) -> Tuple[float, str]:
        """
        Evaluate budget match between user's daily budget and destination's estimated cost.
        Returns (score in [0, 1], explanation_string).
        """
        if not prefs.budget_max_inr or not prefs.duration_days or prefs.duration_days <= 0:
            return 0.85, "Budget not specified; assumed compatible"

        total_budget = prefs.budget_max_inr
        duration = prefs.duration_days
        # Allocate 80% of total budget to on-ground expenses
        daily_budget = (total_budget * 0.80) / duration
        dest_daily = dest.est_daily_cost_inr

        if dest_daily <= daily_budget:
            # Within budget
            savings_ratio = (daily_budget - dest_daily) / max(1.0, daily_budget)
            score = 1.0 - (0.15 * min(1.0, savings_ratio))
            return max(0.0, min(1.0, score)), f"Comfortably within budget (Est. ₹{dest_daily:,.0f}/day vs ₹{daily_budget:,.0f}/day)"
        else:
            # Over budget: exponential decay penalty
            overage_ratio = (dest_daily - daily_budget) / max(1.0, daily_budget)
            score = math.exp(-2.5 * overage_ratio)
            return max(0.0, min(1.0, score)), f"Exceeds target budget (Est. ₹{dest_daily:,.0f}/day vs ₹{daily_budget:,.0f}/day target)"

    def score_seasonal_compatibility(self, dest: Destination, prefs: UserPreferences) -> Tuple[float, str]:
        """
        Evaluate seasonal fit based on desired travel month (1-12).
        Returns (score in [0, 1], explanation_string).
        """
        if not prefs.travel_month or not (1 <= prefs.travel_month <= 12):
            return 0.8, "Travel month flexible"

        month = prefs.travel_month
        best_months = dest.best_months or list(range(1, 13))

        if month in best_months:
            return 1.0, "Optimal travel season with ideal weather"
        
        # Check shoulder months (adjacent months)
        prev_m = 12 if month == 1 else month - 1
        next_m = 1 if month == 12 else month + 1
        if prev_m in best_months or next_m in best_months:
            return 0.70, "Shoulder season with good conditions"

        return 0.35, "Off-peak travel season (potential weather trade-offs)"

    def score_preference_match(self, dest: Destination, prefs: UserPreferences) -> Tuple[float, str]:
        """
        Evaluate topical alignment between user interests / categories and destination attributes.
        """
        desired_tokens = set()
        if prefs.interests:
            desired_tokens.update([t.lower().strip() for t in prefs.interests])
        if prefs.categories:
            desired_tokens.update([c.lower().strip() for c in prefs.categories])

        if not desired_tokens:
            return 0.7, "General interest alignment"

        dest_attributes = set([dest.category.lower()])
        dest_attributes.update([t.lower() for t in dest.tags])
        dest_attributes.update([a.lower() for a in dest.activities])
        for food in dest.famous_foods:
            dest_attributes.update(food.lower().split())

        # Measure token overlap
        matched = desired_tokens.intersection(dest_attributes)
        score = len(matched) / len(desired_tokens)
        score = min(1.0, score * 1.5)  # Soft boost if partial match

        # Check disliked features penalty
        if prefs.disliked_features:
            disliked_set = set([d.lower().strip() for d in prefs.disliked_features])
            disliked_matches = disliked_set.intersection(dest_attributes)
            if disliked_matches:
                score *= 0.5  # Heavy penalty for disliked features

        reasons = f"Matched interests: {', '.join(matched)}" if matched else "Moderate topical match"
        return max(0.0, min(1.0, score)), reasons

    def score_quality_and_safety(self, dest: Destination) -> Tuple[float, str]:
        """Evaluate destination quality and safety prior."""
        safety_mult = 1.0 if dest.safety_rating == "High" else (0.75 if dest.safety_rating == "Medium" else 0.40)
        pop_score = dest.popularity_score or 0.5
        score = (0.6 * safety_mult) + (0.4 * pop_score)
        return max(0.0, min(1.0, score)), f"{dest.safety_rating} safety rating with high traveler satisfaction"

    def compute_scores(self, record: CandidateRecord, prefs: UserPreferences) -> FeatureBreakdown:
        """Calculate complete feature breakdown and composite weighted score."""
        dest = record.destination
        breakdown = FeatureBreakdown()

        # 1. Dense similarity
        dense_sim = max(0.0, min(1.0, record.dense_similarity))
        # 2. Lexical similarity
        lexical_sim = max(0.0, min(1.0, record.lexical_similarity))
        # 3. Preference match
        pref_score, _ = self.score_preference_match(dest, prefs)
        # 4. Budget match
        budget_score, _ = self.score_budget_compatibility(dest, prefs)
        # 5. Seasonal match
        seasonal_score, _ = self.score_seasonal_compatibility(dest, prefs)
        # 6. Quality prior
        quality_score, _ = self.score_quality_and_safety(dest)

        breakdown.raw_features = {
            "dense_similarity": dense_sim,
            "lexical_similarity": lexical_sim,
            "preference_match": pref_score,
            "budget_match": budget_score,
            "seasonal_match": seasonal_score,
            "quality_rating": quality_score
        }

        # Weighted composite score
        total_score = 0.0
        for feature, weight in self.weights.items():
            val = breakdown.raw_features.get(feature, 0.5)
            contrib = val * weight
            breakdown.weighted_contributions[feature] = contrib
            total_score += contrib

        # Hard geographic failsafe: if continent preference is specified, exclude non-matching continents
        if prefs.continents and dest.continent not in prefs.continents:
            breakdown.final_score = 0.0
            return breakdown

        # Country preference boost if specific country requested
        if prefs.countries:
            if dest.country in prefs.countries:
                total_score = min(1.0, total_score * 1.25 + 0.1)
            else:
                total_score *= 0.5

        breakdown.final_score = max(0.0, min(1.0, total_score))
        return breakdown
