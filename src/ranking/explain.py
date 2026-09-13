"""Explainability engine producing evidence-based rationales grounded in recommendation features."""

import calendar
from typing import List, Dict
from src.data.models import Destination, UserPreferences, RecommendationExplanation
from src.ranking.scorer import FeatureBreakdown, HybridScorer


class ExplainabilityEngine:
    """
    Generates human-readable, transparent explanation evidence for each recommendation.
    Every explanation is grounded in deterministic feature contributions; never hallucinated.
    """

    def __init__(self, scorer: HybridScorer = None):
        self.scorer = scorer or HybridScorer()

    def generate_explanation(
        self,
        dest: Destination,
        breakdown: FeatureBreakdown,
        prefs: UserPreferences
    ) -> RecommendationExplanation:
        """Construct verified explanation evidence based on calculated feature values."""
        reasons: List[str] = []
        raw = breakdown.raw_features

        # 1. Semantic Match Evidence
        dense_sim = raw.get("dense_similarity", 0.0)
        if dense_sim >= 0.65:
            reasons.append("Exceptional semantic match for your travel desires and trip ambiance")
        elif dense_sim >= 0.50:
            reasons.append("Strong semantic alignment with your preferred travel experience")

        # 2. Lexical & Keyword Evidence
        lexical_sim = raw.get("lexical_similarity", 0.0)
        if lexical_sim >= 0.08:
            reasons.append("Direct keyword and phrasing overlap with destination activities")

        # 3. Interests & Category Evidence
        pref_score = raw.get("preference_match", 0.0)
        matching_interests = []
        if prefs.interests:
            dest_text = f"{dest.category} {' '.join(dest.tags)} {' '.join(dest.activities)}".lower()
            for interest in prefs.interests:
                if interest.lower() in dest_text:
                    matching_interests.append(interest)

        if matching_interests:
            reasons.append(f"Direct match for your interests: {', '.join(matching_interests)}")
        elif pref_score >= 0.60:
            reasons.append(f"Fits your category preference for {dest.category} travel")

        # 4. Budget Evidence
        _, budget_fit_text = self.scorer.score_budget_compatibility(dest, prefs)
        reasons.append(budget_fit_text)

        # 5. Seasonal Evidence
        seasonal_score = raw.get("seasonal_match", 0.0)
        if prefs.travel_month and 1 <= prefs.travel_month <= 12:
            m_name = calendar.month_name[prefs.travel_month]
            if seasonal_score >= 0.90:
                seasonal_fit_text = f"Prime travel month: {m_name} offers peak weather"
            elif seasonal_score >= 0.60:
                seasonal_fit_text = f"Shoulder season: {m_name} offers good conditions with fewer crowds"
            else:
                seasonal_fit_text = f"Off-season in {m_name}; expect unique seasonal conditions"
            reasons.append(seasonal_fit_text)
        else:
            seasonal_fit_text = f"Recommended seasons: {', '.join(dest.best_seasons or ['Year-round'])}"

        # 6. Safety & Reputation Prior
        if dest.safety_rating == "High":
            reasons.append("High safety rating and excellent traveler infrastructure")

        # Feature contribution mapping for radar/bar visualizers
        contributions: Dict[str, float] = {
            "Semantic Match": round(dense_sim, 3),
            "Keyword Match": round(lexical_sim, 3),
            "Preferences": round(pref_score, 3),
            "Budget Fit": round(raw.get("budget_match", 0.0), 3),
            "Season Fit": round(seasonal_score, 3),
            "Quality/Safety": round(raw.get("quality_rating", 0.0), 3)
        }

        return RecommendationExplanation(
            reasons=reasons,
            feature_contributions=contributions,
            budget_fit=budget_fit_text,
            seasonal_fit=seasonal_fit_text
        )
