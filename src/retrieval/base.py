"""Base classes and interfaces for retrieval components."""

from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field
from src.data.models import Destination


class SearchResult(BaseModel):
    """Container for a retrieved candidate and its score."""
    destination: Destination
    score: float = Field(..., description="Retrieval similarity score in range [0.0, 1.0]")
    retriever_name: str = Field(..., description="Name of retriever that produced this result")


class BaseRetriever(ABC):
    """Abstract interface for all Roamio candidate retrievers."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 20, candidate_ids: Optional[List[str]] = None) -> List[SearchResult]:
        """
        Retrieve top_k destinations matching natural language query.
        Optionally restricted to a subset of candidate_ids (for two-stage filtering).
        """
        pass
