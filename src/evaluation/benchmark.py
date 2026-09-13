"""Curated ground-truth benchmark suite for offline recommendation evaluation."""

from typing import List, Dict, Set
from pydantic import BaseModel, Field
from src.data.models import UserPreferences


class BenchmarkQuery(BaseModel):
    """A test case representing a realistic traveler persona, query, and known relevant destinations."""
    query_id: str
    intent_description: str
    preferences: UserPreferences
    relevant_ids: Set[str] = Field(..., description="Set of canonical destination IDs considered relevant")
    relevance_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Graded relevance scores (1.0 = primary match, 0.5 = secondary match)"
    )


BENCHMARK_SUITE: List[BenchmarkQuery] = [
    BenchmarkQuery(
        query_id="Q01_CHEAP_BEACH_ASIA",
        intent_description="Affordable beach vacation in Asia under ₹60k",
        preferences=UserPreferences(
            query_text="cheap tropical beach vacation in asia",
            continents=["Asia"],
            budget_max_inr=60000,
            duration_days=7,
            categories=["Beach"],
            interests=["beaches", "snorkeling", "seafood"]
        ),
        relevant_ids={"CURATED-0004", "CURATED-0003", "SEED-0011", "CURATED-0012"},
        relevance_scores={"CURATED-0004": 1.0, "CURATED-0003": 0.8, "CURATED-0012": 0.9}
    ),
    BenchmarkQuery(
        query_id="Q02_CULTURAL_TEMPLES",
        intent_description="Historic and spiritual cultural destinations with ancient temples",
        preferences=UserPreferences(
            query_text="ancient temples cultural heritage and spiritual sites",
            interests=["temples", "history", "buddhist", "culture"]
        ),
        relevant_ids={"CURATED-0001", "CURATED-0005", "CURATED-0010", "CURATED-0004", "CURATED-0011"},
        relevance_scores={"CURATED-0001": 1.0, "CURATED-0005": 1.0, "CURATED-0010": 0.9, "CURATED-0004": 0.8, "CURATED-0011": 0.7}
    ),
    BenchmarkQuery(
        query_id="Q03_WARM_DECEMBER",
        intent_description="Escaping winter: warm, sunny destinations in December",
        preferences=UserPreferences(
            query_text="warm sunny destinations to travel in december",
            travel_month=12,
            interests=["warmth", "sunshine", "outdoor", "beaches"]
        ),
        relevant_ids={"CURATED-0003", "CURATED-0004", "CURATED-0021", "CURATED-0019", "CURATED-0025"},
        relevance_scores={"CURATED-0003": 1.0, "CURATED-0004": 1.0, "CURATED-0021": 0.9, "CURATED-0019": 0.9, "CURATED-0025": 0.9}
    ),
    BenchmarkQuery(
        query_id="Q04_ALPINE_HIKING",
        intent_description="Mountain landscapes, alpine hiking, and dramatic peaks",
        preferences=UserPreferences(
            query_text="dramatic mountains alpine hiking scenic trails and fresh air",
            categories=["Mountain", "National Park"],
            interests=["hiking", "mountains", "alps", "nature"]
        ),
        relevant_ids={"CURATED-0017", "CURATED-0020", "CURATED-0009", "SEED-0030", "CURATED-0026"},
        relevance_scores={"CURATED-0017": 1.0, "CURATED-0020": 1.0, "CURATED-0009": 0.9, "SEED-0030": 0.9, "CURATED-0026": 0.8}
    ),
    BenchmarkQuery(
        query_id="Q05_PEACEFUL_FOOD",
        intent_description="Serene, relaxing destinations celebrated for regional culinary traditions",
        preferences=UserPreferences(
            query_text="peaceful relaxing destinations with incredible local food and culinary culture",
            interests=["peaceful", "foodie", "local cuisine", "relaxation"]
        ),
        relevant_ids={"CURATED-0001", "CURATED-0012", "CURATED-0006", "SEED-0002", "CURATED-0004"},
        relevance_scores={"CURATED-0001": 1.0, "CURATED-0012": 1.0, "CURATED-0006": 0.9, "SEED-0002": 0.9, "CURATED-0004": 0.8}
    ),
    BenchmarkQuery(
        query_id="Q06_EUROPEAN_HERITAGE",
        intent_description="Classic European cities with world-famous museums and architecture",
        preferences=UserPreferences(
            query_text="historic european cities with world class museums art and architecture",
            continents=["Europe"],
            categories=["City", "Cultural"],
            interests=["museums", "art", "architecture", "history"]
        ),
        relevant_ids={"SEED-0001", "SEED-0002", "SEED-0021", "CURATED-0013", "CURATED-0015"},
        relevance_scores={"SEED-0001": 1.0, "SEED-0002": 1.0, "SEED-0021": 1.0, "CURATED-0013": 0.9, "CURATED-0015": 0.9}
    ),
    BenchmarkQuery(
        query_id="Q07_ANCIENT_WONDERS",
        intent_description="Archaeological wonders and ancient civilization ruins",
        preferences=UserPreferences(
            query_text="ancient archaeological wonders of the world and historic ruins",
            categories=["Archaeological Site"],
            interests=["ancient history", "ruins", "monuments", "unesco"]
        ),
        relevant_ids={"CURATED-0023", "CURATED-0024", "CURATED-0018", "CURATED-0005", "CURATED-0010"},
        relevance_scores={"CURATED-0023": 1.0, "CURATED-0024": 1.0, "CURATED-0018": 1.0, "CURATED-0005": 1.0, "CURATED-0010": 0.9}
    ),
    BenchmarkQuery(
        query_id="Q08_ADVENTURE_SPORTS",
        intent_description="High-adrenaline outdoor adventure activities",
        preferences=UserPreferences(
            query_text="extreme outdoor adventure bungee jumping skiing and thrilling sports",
            interests=["adventure", "skiing", "hiking", "sports"]
        ),
        relevant_ids={"CURATED-0026", "CURATED-0017", "CURATED-0020", "SEED-0030"},
        relevance_scores={"CURATED-0026": 1.0, "CURATED-0017": 0.9, "CURATED-0020": 0.9, "SEED-0030": 0.8}
    )
]
