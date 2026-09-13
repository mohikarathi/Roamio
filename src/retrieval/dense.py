"""Dense semantic retrieval component using precomputed sentence embeddings."""

import json
import logging
from typing import List, Optional, Dict
import numpy as np
from fastembed import TextEmbedding

from src.config import (
    EMBEDDINGS_CACHE_PATH,
    EMBEDDINGS_METADATA_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DATABASE_PATH
)
from src.data.models import Destination
from src.data.db import get_all_destinations
from src.retrieval.base import BaseRetriever, SearchResult

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Dense semantic retriever computing cosine similarity over precomputed sentence embeddings.
    Embeddings are computed once and reused across all queries.
    """

    def __init__(
        self,
        embeddings_path=EMBEDDINGS_CACHE_PATH,
        metadata_path=EMBEDDINGS_METADATA_PATH,
        model_name=DEFAULT_EMBEDDING_MODEL
    ):
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.embeddings: Optional[np.ndarray] = None
        self.destinations: List[Destination] = []
        self.id_to_index: Dict[str, int] = {}
        self.model: Optional[TextEmbedding] = None
        self._load_cache()

    def _load_cache(self) -> None:
        """Load precomputed embedding matrix and initialize query embedder."""
        if not self.embeddings_path.exists() or not self.metadata_path.exists():
            logger.warning("Dense embedding cache not found. Please run the pipeline first.")
            return

        self.embeddings = np.load(self.embeddings_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.destinations = get_all_destinations(DATABASE_PATH)
        self.id_to_index = {d.destination_id: idx for idx, d in enumerate(self.destinations)}
        # Lazy load model for queries
        self.model = TextEmbedding(model_name=self.model_name)
        logger.info(f"Loaded DenseRetriever with {self.embeddings.shape[0]} precomputed vectors (dim={self.embeddings.shape[1]}).")

    def retrieve(self, query: str, top_k: int = 20, candidate_ids: Optional[List[str]] = None) -> List[SearchResult]:
        """
        Retrieve destinations matching query using embedding dot product.
        Precomputed destination embeddings are reused.
        """
        if not query.strip() or self.embeddings is None or self.model is None:
            return []

        # Embed query text
        query_generator = self.model.embed([query])
        query_vec = np.array(list(query_generator)[0], dtype=np.float32)

        # Normalize query vector
        norm = np.linalg.norm(query_vec)
        if norm > 0.0:
            query_vec = query_vec / norm

        # Subset candidate embeddings if candidate_ids specified
        if candidate_ids is not None:
            valid_indices = [self.id_to_index[cid] for cid in candidate_ids if cid in self.id_to_index]
            if not valid_indices:
                return []
            sub_embeddings = self.embeddings[valid_indices]
            sims = np.dot(sub_embeddings, query_vec)
            target_indices = valid_indices
        else:
            sims = np.dot(self.embeddings, query_vec)
            target_indices = list(range(len(self.destinations)))

        # Rank candidates by descending cosine similarity
        top_idx_order = np.argsort(sims)[::-1][:top_k]

        results = []
        for rank_pos in top_idx_order:
            score = float(sims[rank_pos])
            dest_idx = target_indices[rank_pos]
            results.append(SearchResult(
                destination=self.destinations[dest_idx],
                score=max(0.0, min(1.0, score)),
                retriever_name="Dense-Embeddings"
            ))

        return results
