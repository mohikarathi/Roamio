"""Reproducible data engineering and model precomputation pipeline."""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    DATABASE_PATH,
    TFIDF_VECTORIZER_PATH,
    TFIDF_MATRIX_PATH,
    EMBEDDINGS_CACHE_PATH,
    EMBEDDINGS_METADATA_PATH,
    DEFAULT_EMBEDDING_MODEL,
    PROCESSED_DATA_DIR,
    METADATA_DIR
)
from src.data.models import Destination
from src.data.ingestion import ingest_seed_excel, ingest_curated_dataset
from src.data.entity_resolution import EntityResolver
from src.data.validator import validate_destination_batch
from src.data.db import init_db, upsert_destinations, get_all_destinations, get_database_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RoamioPipeline")


def run_pipeline() -> Dict[str, Any]:
    """
    Run the end-to-end data pipeline:
    1. Multi-source Ingestion
    2. Entity Resolution & Deduplication
    3. Schema & Data Validation
    4. SQLite Canonical Database Update
    5. TF-IDF Corpus & Matrix Precomputation
    6. Dense Sentence Embedding Generation & Caching
    """
    start_time = time.time()
    logger.info("=== Starting Roamio Data & Model Pipeline ===")

    # 1. Multi-source Ingestion
    logger.info("Step 1: Ingesting seed and curated datasets...")
    seed_destinations = ingest_seed_excel()
    curated_destinations = ingest_curated_dataset()
    raw_total = len(seed_destinations) + len(curated_destinations)
    logger.info(f"Ingested {len(seed_destinations)} seed + {len(curated_destinations)} curated records = {raw_total} total raw records.")

    # 2. Entity Resolution
    logger.info("Step 2: Resolving entities and deduplicating...")
    resolver = EntityResolver(geo_threshold_km=35.0, string_sim_threshold=0.65)
    merged_count = 0

    for d in seed_destinations:
        resolver.resolve_and_add(d)

    for d in curated_destinations:
        _, was_merged = resolver.resolve_and_add(d)
        if was_merged:
            merged_count += 1

    canonical_list = list(resolver.canonical_entities.values())
    logger.info(f"Entity Resolution complete. Merged {merged_count} duplicates. Total unique entities: {len(canonical_list)}.")

    # 3. Data Validation
    logger.info("Step 3: Validating canonical records...")
    valid_destinations, val_report = validate_destination_batch(canonical_list)
    logger.info(f"Validation complete: {val_report['valid_count']} valid, {val_report['rejected_count']} rejected.")

    # Save validation report
    val_report_path = METADATA_DIR / "validation_report.json"
    with open(val_report_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)

    # 4. Database Persistence
    logger.info(f"Step 4: Writing {len(valid_destinations)} canonical records to SQLite at {DATABASE_PATH}...")
    init_db(DATABASE_PATH)
    upsert_destinations(valid_destinations, DATABASE_PATH)
    db_stats = get_database_stats(DATABASE_PATH)
    logger.info(f"Database update complete: {db_stats['total_destinations']} destinations across {db_stats['total_countries']} countries.")

    # 5. Semantic Corpus Generation
    logger.info("Step 5: Generating semantic documents for retrieval...")
    all_destinations = get_all_destinations(DATABASE_PATH)
    corpus = [d.to_semantic_document() for d in all_destinations]
    dest_ids = [d.destination_id for d in all_destinations]

    # 6. TF-IDF Precomputation
    logger.info("Step 6: Fitting and caching TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    with open(TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(TFIDF_MATRIX_PATH, "wb") as f:
        pickle.dump(tfidf_matrix, f)

    logger.info(f"TF-IDF matrix cached at {TFIDF_MATRIX_PATH} (Shape: {tfidf_matrix.shape}).")

    # 7. Dense Semantic Embeddings Generation
    logger.info(f"Step 7: Precomputing dense sentence embeddings using {DEFAULT_EMBEDDING_MODEL}...")
    from fastembed import TextEmbedding
    embedding_model = TextEmbedding(model_name=DEFAULT_EMBEDDING_MODEL)
    embedding_generator = embedding_model.embed(corpus)
    embeddings_list = list(embedding_generator)
    embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

    # Normalize vectors for cosine similarity via dot product
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized_embeddings = embeddings_matrix / norms

    # Cache embeddings and metadata
    np.save(EMBEDDINGS_CACHE_PATH, normalized_embeddings)

    metadata = {
        "model_name": DEFAULT_EMBEDDING_MODEL,
        "embedding_dim": int(normalized_embeddings.shape[1]),
        "total_destinations": int(normalized_embeddings.shape[0]),
        "destination_ids": dest_ids,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(EMBEDDINGS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dense embeddings cached at {EMBEDDINGS_CACHE_PATH} (Shape: {normalized_embeddings.shape}).")

    elapsed_time = time.time() - start_time
    logger.info(f"=== Pipeline Completed Successfully in {elapsed_time:.2f}s ===")

    return {
        "status": "success",
        "elapsed_seconds": elapsed_time,
        "total_destinations": len(valid_destinations),
        "total_countries": db_stats["total_countries"],
        "tfidf_shape": list(tfidf_matrix.shape),
        "embedding_shape": list(normalized_embeddings.shape)
    }


if __name__ == "__main__":
    run_pipeline()
