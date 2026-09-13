"""Two-stage retrieval orchestrator combining structured filtering with dual lexical & semantic search."""

import logging
from typing import List, Dict, Any, Optional
from src.data.models import Destination, UserPreferences
from src.data.db import filter_destinations_query, get_all_destinations
from src.retrieval.tfidf import TFIDFRetriever
from src.retrieval.dense import DenseRetriever

logger = logging.getLogger(__name__)


class CandidateRecord:
    """Stores candidate destination and its stage-2 retrieval scores."""
    def __init__(self, destination: Destination, dense_similarity: float = 0.0, lexical_similarity: float = 0.0):
        self.destination = destination
        self.dense_similarity = dense_similarity
        self.lexical_similarity = lexical_similarity


class TwoStageRetriever:
    """
    Two-stage retrieval pipeline:
    Stage 1: Structured SQL filtering (geography, budget, season, safety) to prune candidate space.
    Stage 2: Dense semantic & TF-IDF lexical retrieval over candidate subset.
    """

    def __init__(self, tfidf_retriever: Optional[TFIDFRetriever] = None, dense_retriever: Optional[DenseRetriever] = None):
        self.tfidf_retriever = tfidf_retriever or TFIDFRetriever()
        self.dense_retriever = dense_retriever or DenseRetriever()

    def filter_candidates(self, prefs: UserPreferences, min_candidates: int = 15) -> List[Destination]:
        """
        Stage 1: Fast structured candidate filtering.
        Includes soft relaxation fallback so that strict filters never return 0 results.
        """
        continents = prefs.continents if prefs.continents else None
        countries = prefs.countries if prefs.countries else None
        category = prefs.categories[0] if prefs.categories else None

        # Calculate max daily budget if total budget and duration are specified
        max_daily_cost = None
        if prefs.budget_max_inr and prefs.duration_days and prefs.duration_days > 0:
            # 85% of total budget allocated to destination daily expenses (allowing flight buffer)
            max_daily_cost = (prefs.budget_max_inr * 0.85) / prefs.duration_days

        candidates = filter_destinations_query(
            continent=continents,
            country=countries,
            category=category,
            max_daily_cost_inr=max_daily_cost,
            travel_month=prefs.travel_month,
            min_safety=prefs.min_safety
        )

        # Step 1: Relax budget if too restrictive
        if len(candidates) < min_candidates and max_daily_cost is not None:
            logger.info(f"Few candidates found with strict budget ({len(candidates)}). Relaxing budget threshold.")
            candidates = filter_destinations_query(
                continent=continents,
                country=countries,
                category=category,
                max_daily_cost_inr=None,  # Relax budget
                travel_month=prefs.travel_month,
                min_safety=prefs.min_safety
            )

        # Step 2: Relax travel month if few candidates
        if len(candidates) < min_candidates and prefs.travel_month is not None:
            candidates = filter_destinations_query(
                continent=continents,
                country=countries,
                category=category,
                max_daily_cost_inr=None,
                travel_month=None,
                min_safety=prefs.min_safety
            )

        # Step 3: Relax safety threshold if few candidates
        if len(candidates) < min_candidates and prefs.min_safety and prefs.min_safety != "All":
            candidates = filter_destinations_query(
                continent=continents,
                country=countries,
                category=category,
                max_daily_cost_inr=None,
                travel_month=None,
                min_safety=None
            )

        # Step 4: Relax category if few candidates
        if len(candidates) < min_candidates and category is not None:
            candidates = filter_destinations_query(
                continent=continents,
                country=countries,
                category=None,
                max_daily_cost_inr=None,
                travel_month=None,
                min_safety=None
            )

        # Step 5: If country was specified and still 0 candidates, relax country to continent
        if not candidates and countries is not None:
            candidates = filter_destinations_query(
                continent=continents,
                country=None,
                category=None,
                max_daily_cost_inr=None,
                travel_month=None,
                min_safety=None
            )

        # Step 6: If still no candidates, only relax continent if user did NOT specify continent
        if not candidates:
            if not continents:
                candidates = get_all_destinations()
            else:
                candidates = filter_destinations_query(continent=continents)
                if not candidates:
                    candidates = get_all_destinations()

        # Hard geographic invariant: if continents were requested, exclude any destination from other continents
        if continents:
            req_set = set(continents if isinstance(continents, list) else [continents])
            candidates = [c for c in candidates if c.continent in req_set]

        return candidates

    def retrieve_candidates(self, prefs: UserPreferences, top_k_stage2: int = 40) -> List[CandidateRecord]:
        """
        Execute two-stage retrieval:
        1. Prune candidate pool via structured filtering.
        2. Score remaining candidates via dense embeddings and TF-IDF.
        """
        # Stage 1: Filter candidate destinations
        candidates = self.filter_candidates(prefs)
        if prefs.continents:
            req_set = set(prefs.continents)
            candidates = [d for d in candidates if d.continent in req_set]
        candidate_ids = [d.destination_id for d in candidates]
        dest_map = {d.destination_id: d for d in candidates}

        query = prefs.query_text.strip()
        # If no query text provided, synthesize one from interests / categories
        if not query:
            tokens = []
            if prefs.interests:
                tokens.extend(prefs.interests)
            if prefs.categories:
                tokens.extend(prefs.categories)
            query = " ".join(tokens) if tokens else "scenic travel destination"

        # Stage 2: Dual scoring over candidates
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k_stage2, candidate_ids=candidate_ids)
        tfidf_results = self.tfidf_retriever.retrieve(query, top_k=top_k_stage2, candidate_ids=candidate_ids)

        dense_score_map = {r.destination.destination_id: r.score for r in dense_results}
        tfidf_score_map = {r.destination.destination_id: r.score for r in tfidf_results}

        records: List[CandidateRecord] = []
        for cid in candidate_ids:
            dest = dest_map[cid]
            d_score = dense_score_map.get(cid, 0.0)
            t_score = tfidf_score_map.get(cid, 0.0)
            records.append(CandidateRecord(
                destination=dest,
                dense_similarity=d_score,
                lexical_similarity=t_score
            ))

        return records
