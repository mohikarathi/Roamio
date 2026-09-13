"""Unit tests for TF-IDF, dense embedding retrieval, and two-stage candidate search."""

import pytest
from src.retrieval.tfidf import TFIDFRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.two_stage import TwoStageRetriever
from src.data.models import UserPreferences


@pytest.fixture(scope="module")
def tfidf():
    return TFIDFRetriever()


@pytest.fixture(scope="module")
def dense():
    return DenseRetriever()


@pytest.fixture(scope="module")
def two_stage(tfidf, dense):
    return TwoStageRetriever(tfidf, dense)


def test_tfidf_retriever(tfidf):
    results = tfidf.retrieve("temples and cultural heritage", top_k=5)
    assert len(results) > 0
    assert results[0].score >= 0.0
    assert results[0].retriever_name == "TF-IDF"


def test_dense_retriever(dense):
    results = dense.retrieve("scenic snowy mountains for winter skiing", top_k=5)
    assert len(results) > 0
    assert results[0].score >= 0.0
    assert results[0].retriever_name == "Dense-Embeddings"
    # Verify semantic match is strong for mountain destinations
    top_names = [r.destination.name.lower() for r in results]
    assert any("alps" in name or "zermatt" in name or "mountain" in name or "banff" in name for name in top_names)


def test_two_stage_candidate_filtering(two_stage):
    # Test strict Asia continent filter
    prefs = UserPreferences(
        query_text="tropical beach with seafood",
        continents=["Asia"],
        budget_max_inr=70000,
        duration_days=7
    )
    candidates = two_stage.retrieve_candidates(prefs, top_k_stage2=10)
    assert len(candidates) > 0
    # All returned candidates should be in Asia
    for c in candidates:
        assert c.destination.continent == "Asia"


def test_two_stage_europe_continent_filtering(two_stage):
    # Test strict Europe continent filter
    prefs = UserPreferences(
        query_text="European cultural city trip",
        continents=["Europe"],
        budget_max_inr=80000,
        duration_days=7
    )
    candidates = two_stage.retrieve_candidates(prefs, top_k_stage2=15)
    assert len(candidates) > 0
    for c in candidates:
        assert c.destination.continent == "Europe"
        assert c.destination.name.lower() != "bali"

