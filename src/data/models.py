"""Pydantic data models for canonical destinations, user preferences, and recommendations."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class Destination(BaseModel):
    """Canonical representation of a travel destination."""
    destination_id: str = Field(..., description="Unique canonical identifier, e.g. DEST-0001")
    name: str = Field(..., description="Destination display name, e.g. Kyoto")
    city: Optional[str] = Field(None, description="City name if applicable")
    region: Optional[str] = Field(None, description="Region, state, or province")
    country: str = Field(..., description="Country name")
    continent: str = Field(default="Unknown", description="Continent: Asia, Europe, Americas, Africa, Oceania")
    latitude: float = Field(..., description="Geographic latitude in degrees")
    longitude: float = Field(..., description="Geographic longitude in degrees")
    category: str = Field(..., description="Primary category: City, Beach, Mountain, Cultural, etc.")
    tags: List[str] = Field(default_factory=list, description="Descriptive tags and secondary categories")
    description: str = Field(..., description="Main overview narrative")
    cultural_significance: Optional[str] = Field(None, description="Historical, cultural, or religious significance")
    activities: List[str] = Field(default_factory=list, description="Popular activities e.g. hiking, temples, diving")
    famous_foods: List[str] = Field(default_factory=list, description="Signature local culinary items")
    cost_level: str = Field(default="Medium", description="Budget tier: Low, Medium, High, Luxury")
    est_daily_cost_inr: float = Field(default=5000.0, description="Estimated daily cost per person in INR")
    currency: str = Field(default="INR", description="Local currency name or code")
    safety_rating: str = Field(default="High", description="Safety assessment: High, Medium, Low")
    best_months: List[int] = Field(default_factory=list, description="Best months to visit (1-12)")
    best_seasons: List[str] = Field(default_factory=list, description="Seasons: Spring, Summer, Fall, Winter, Year-round")
    annual_tourists: Optional[int] = Field(None, description="Approximate annual tourist count")
    popularity_score: float = Field(default=0.5, description="Normalized popularity score [0.0 - 1.0]")
    source: str = Field(default="Seed Dataset", description="Origin data source for provenance tracking")
    external_id: Optional[str] = Field(None, description="External authority ID, e.g. Wikidata QID or GeoNames ID")
    image_url: Optional[str] = Field(None, description="High-resolution landscape photo URL")
    thumbnail_url: Optional[str] = Field(None, description="Optimized thumbnail photo URL")
    photo_author: Optional[str] = Field(None, description="Photographer or author attribution name")
    photo_author_url: Optional[str] = Field(None, description="URL to photographer profile or source photo")
    image_provider: Optional[str] = Field(default="Unsplash", description="Image source provider name")
    image_alt: Optional[str] = Field(None, description="Descriptive alt text for accessibility")
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="ISO timestamp")

    def to_semantic_document(self) -> str:
        """
        Build a rich textual representation of the destination for TF-IDF and dense embeddings.
        Combines geography, categories, narrative descriptions, activities, and foods.
        """
        parts = [
            f"{self.name}, {self.country}.",
            f"Region: {self.region or 'Unknown'} ({self.continent}).",
            f"Category: {self.category}.",
            f"Tags: {', '.join(self.tags)}." if self.tags else "",
            f"Overview: {self.description}.",
            f"Cultural Heritage: {self.cultural_significance}." if self.cultural_significance else "",
            f"Activities and Attractions: {', '.join(self.activities)}." if self.activities else "",
            f"Culinary & Food Culture: {', '.join(self.famous_foods)}." if self.famous_foods else "",
            f"Cost Profile: {self.cost_level} budget.",
            f"Best Seasons: {', '.join(self.best_seasons)}." if self.best_seasons else "",
        ]
        return " ".join([p for p in parts if p.strip()])


class UserPreferences(BaseModel):
    """Structured user preferences extracted from conversational chat or UI filters."""
    query_text: str = Field(default="", description="Freeform user query or keyword desires")
    budget_max_inr: Optional[float] = Field(None, description="Maximum total trip budget in INR")
    duration_days: Optional[int] = Field(default=7, description="Estimated trip duration in days")
    travel_month: Optional[int] = Field(None, description="Planned travel month (1-12)")
    continents: List[str] = Field(default_factory=list, description="Allowed continents e.g. ['Asia']")
    countries: List[str] = Field(default_factory=list, description="Allowed countries e.g. ['Japan', 'Thailand']")
    categories: List[str] = Field(default_factory=list, description="Desired categories e.g. ['Beach', 'Cultural']")
    interests: List[str] = Field(default_factory=list, description="Extracted interest tokens e.g. ['temples', 'food', 'hiking']")
    disliked_features: List[str] = Field(default_factory=list, description="Features to avoid e.g. ['crowds', 'nightlife']")
    min_safety: Optional[str] = Field(default="Medium", description="Minimum safety requirement")
    user_location: Optional[str] = Field(None, description="User's starting city for distance calculations")


class RecommendationExplanation(BaseModel):
    """Evidence-based explanation breakdown grounded in actual feature values."""
    reasons: List[str] = Field(default_factory=list, description="Bullet points explaining why this destination matched")
    feature_contributions: Dict[str, float] = Field(
        default_factory=dict, 
        description="Normalized contribution score for each ranking component [0.0 - 1.0]"
    )
    budget_fit: str = Field(default="Within budget", description="Budget assessment summary")
    seasonal_fit: str = Field(default="Good time to visit", description="Seasonal suitability summary")


class RecommendationItem(BaseModel):
    """A single ranked destination result returned to the user or chatbot."""
    rank: int
    destination: Destination
    final_score: float = Field(..., description="Composite ranking score [0.0 - 1.0]")
    dense_similarity: float = Field(default=0.0, description="Semantic embedding similarity")
    lexical_similarity: float = Field(default=0.0, description="TF-IDF keyword similarity")
    explanation: RecommendationExplanation


class RecommendationResponse(BaseModel):
    """Complete response payload containing ranked results and metadata."""
    items: List[RecommendationItem]
    total_candidates_evaluated: int
    retrieval_latency_ms: float
    ranking_latency_ms: float
    diversity_metric: float = Field(default=0.0, description="Intra-list diversity score")
    applied_preferences: UserPreferences
