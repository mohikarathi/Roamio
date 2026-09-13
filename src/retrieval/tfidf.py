"""TF-IDF lexical retrieval baseline component."""

import pickle
import logging
from typing import List, Optional, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH, DATABASE_PATH
from src.data.models import Destination
from src.data.db import get_all_destinations
from src.retrieval.base import BaseRetriever, SearchResult

logger = logging.getLogger(__name__)


class TFIDFRetriever(BaseRetriever):
    """
    Lexical candidate retriever using TF-IDF token representation and cosine similarity.
    Serves as the deterministic lexical baseline.
    """

    def __init__(self, vectorizer_path=TFIDF_VECTORIZER_PATH, matrix_path=TFIDF_MATRIX_PATH):
        self.vectorizer_path = vectorizer_path
        self.matrix_path = matrix_path
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.destinations: List[Destination] = []
        self.id_to_index: Dict[str, int] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load precomputed TF-IDF vectorizer, sparse matrix, and destination lookup."""
        if not self.vectorizer_path.exists() or not self.matrix_path.exists():
            logger.warning("TF-IDF cache files not found. Retriever will return empty until pipeline is run.")
            return

        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(self.matrix_path, "rb") as f:
            self.tfidf_matrix = pickle.load(f)

        self.destinations = get_all_destinations(DATABASE_PATH)
        self.id_to_index = {d.destination_id: idx for idx, d in enumerate(self.destinations)}
        logger.info(f"Loaded TFIDFRetriever with {len(self.destinations)} indexed destinations.")

    def retrieve(self, query: str, top_k: int = 20, candidate_ids: Optional[List[str]] = None) -> List[SearchResult]:
        """
        Retrieve destinations matching query using TF-IDF cosine similarity.
        Supports candidate filtering for two-stage retrieval.
        """
        if not query.strip() or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        # Vectorize query
        query_vec = self.vectorizer.transform([query])

        # If candidate_ids specified, mask indices
        if candidate_ids is not None:
            valid_indices = [self.id_to_index[cid] for cid in candidate_ids if cid in self.id_to_index]
            if not valid_indices:
                return []
            sub_matrix = self.tfidf_matrix[valid_indices]
            sims = cosine_similarity(query_vec, sub_matrix).flatten()
            target_indices = valid_indices
        else:
            sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            target_indices = list(range(len(self.destinations)))

        # Rank candidates by descending similarity
        top_idx_order = np.argsort(sims)[::-1][:top_k]

        results = []
        for rank_pos in top_idx_order:
            score = float(sims[rank_pos])
            dest_idx = target_indices[rank_pos]
            results.append(SearchResult(
                destination=self.destinations[dest_idx],
                score=max(0.0, min(1.0, score)),
                retriever_name="TF-IDF"
            ))

        return results
