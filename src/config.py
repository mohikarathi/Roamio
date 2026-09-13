"""Configuration and global constants for Roamio."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
DATABASE_PATH = DATA_DIR / "roamio.db"

# Seed data path (original excel file preserved as baseline seed)
SEED_EXCEL_PATH = PROJECT_ROOT / "attached_assets" / "destinations.xlsx"
SEED_CSV_PATH = PROJECT_ROOT / "archive" / "destinations.csv"

# Precomputed artifact paths
TFIDF_VECTORIZER_PATH = PROCESSED_DATA_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = PROCESSED_DATA_DIR / "tfidf_matrix.pkl"
EMBEDDINGS_CACHE_PATH = PROCESSED_DATA_DIR / "destination_embeddings.npy"
EMBEDDINGS_METADATA_PATH = PROCESSED_DATA_DIR / "embeddings_metadata.json"

# Models and NLP
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Fast, highly accurate, lightweight ONNX model
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default Hybrid Ranking Weights (Configurable)
DEFAULT_WEIGHTS = {
    "dense_similarity": 0.35,      # Semantic embedding similarity
    "lexical_similarity": 0.15,    # TF-IDF lexical keyword similarity
    "preference_match": 0.20,      # Category & interest tags match
    "budget_match": 0.15,          # Budget compatibility score
    "seasonal_match": 0.10,        # Travel month & season alignment
    "quality_rating": 0.05,        # Safety & rating quality prior
}

# Diversity settings
DEFAULT_DIVERSITY_LAMBDA = 0.75  # 0 = max diversity, 1 = max relevance (MMR)

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
