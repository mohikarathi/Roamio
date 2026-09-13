"""Evaluation benchmark runner and ablation study harness."""

import time
from typing import List, Dict, Any
import numpy as np

from src.data.models import UserPreferences, Destination
from src.retrieval.tfidf import TFIDFRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.two_stage import TwoStageRetriever
from src.ranking.engine import RecommendationEngine
from src.ranking.diversity import calculate_intra_list_diversity
from src.evaluation.benchmark import BENCHMARK_SUITE, BenchmarkQuery
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mean_reciprocal_rank
)


def evaluate_model(
    model_name: str,
    recommend_fn,
    benchmark_suite: List[BenchmarkQuery] = BENCHMARK_SUITE
) -> Dict[str, float]:
    """
    Evaluate a candidate model or retrieval configuration against the benchmark suite.
    Computes mean Precision@5, Recall@10, NDCG@10, MRR, Intra-List Diversity, and Latency.
    """
    p_at_5_list = []
    r_at_10_list = []
    ndcg_at_10_list = []
    mrr_list = []
    ild_list = []
    latencies = []

    for b_query in benchmark_suite:
        t0 = time.time()
        # Get recommended destination objects
        recommended_destinations: List[Destination] = recommend_fn(b_query.preferences, top_k=10)
        latency_ms = (time.time() - t0) * 1000.0
        latencies.append(latency_ms)

        rec_ids = [d.destination_id for d in recommended_destinations]
        relevant_set = b_query.relevant_ids
        rel_scores = b_query.relevance_scores

        p5 = precision_at_k(rec_ids, relevant_set, k=5)
        r10 = recall_at_k(rec_ids, relevant_set, k=10)
        ndcg10 = ndcg_at_k(rec_ids, rel_scores, k=10)
        rr = mean_reciprocal_rank(rec_ids, relevant_set)
        ild = calculate_intra_list_diversity(recommended_destinations[:5])

        p_at_5_list.append(p5)
        r_at_10_list.append(r10)
        ndcg_at_10_list.append(ndcg10)
        mrr_list.append(rr)
        ild_list.append(ild)

    return {
        "model": model_name,
        "precision@5": round(float(np.mean(p_at_5_list)), 4),
        "recall@10": round(float(np.mean(r_at_10_list)), 4),
        "ndcg@10": round(float(np.mean(ndcg_at_10_list)), 4),
        "mrr": round(float(np.mean(mrr_list)), 4),
        "diversity_ild": round(float(np.mean(ild_list)), 4),
        "avg_latency_ms": round(float(np.mean(latencies)), 2)
    }


def run_ablation_study() -> List[Dict[str, Any]]:
    """
    Run the full ablation study comparing:
    1. Pure TF-IDF Lexical Baseline
    2. Pure Dense Semantic Embeddings
    3. Two-Stage Candidate Retrieval
    4. Full Hybrid Recommender (No Diversity)
    5. Full Hybrid Recommender (With MMR Diversity)
    """
    tfidf_retriever = TFIDFRetriever()
    dense_retriever = DenseRetriever()
    two_stage_retriever = TwoStageRetriever(tfidf_retriever, dense_retriever)
    hybrid_engine = RecommendationEngine(retriever=two_stage_retriever)

    # 1. Pure TF-IDF Baseline
    def run_tfidf(prefs: UserPreferences, top_k: int = 10) -> List[Destination]:
        results = tfidf_retriever.retrieve(prefs.query_text, top_k=top_k)
        return [r.destination for r in results]

    # 2. Pure Dense Semantic Embeddings
    def run_dense(prefs: UserPreferences, top_k: int = 10) -> List[Destination]:
        results = dense_retriever.retrieve(prefs.query_text, top_k=top_k)
        return [r.destination for r in results]

    # 3. Two-Stage Filtered Retrieval
    def run_two_stage(prefs: UserPreferences, top_k: int = 10) -> List[Destination]:
        cands = two_stage_retriever.retrieve_candidates(prefs, top_k_stage2=top_k)
        # Sort by dense similarity
        cands.sort(key=lambda x: x.dense_similarity, reverse=True)
        return [c.destination for c in cands[:top_k]]

    # 4. Full Hybrid Recommender without MMR Diversity
    def run_hybrid_no_diversity(prefs: UserPreferences, top_k: int = 10) -> List[Destination]:
        res = hybrid_engine.recommend(prefs, top_k=top_k, apply_diversity=False)
        return [item.destination for item in res.items]

    # 5. Full Hybrid Recommender with MMR Diversity
    def run_hybrid_full(prefs: UserPreferences, top_k: int = 10) -> List[Destination]:
        res = hybrid_engine.recommend(prefs, top_k=top_k, apply_diversity=True)
        return [item.destination for item in res.items]

    models = [
        ("TF-IDF Baseline", run_tfidf),
        ("Dense Embeddings", run_dense),
        ("Two-Stage Retrieval", run_two_stage),
        ("Hybrid (No Diversity)", run_hybrid_no_diversity),
        ("Roamio Hybrid + MMR", run_hybrid_full),
    ]

    benchmark_results = []
    print("\n" + "=" * 80)
    print("ROAMIO RECOMMENDATION SYSTEM - BENCHMARK & ABLATION STUDY")
    print("=" * 80)

    for name, fn in models:
        metrics = evaluate_model(name, fn)
        benchmark_results.append(metrics)

    # Print Formatted Markdown Table
    header = f"| {'Model / Configuration':<24} | {'P@5':<7} | {'R@10':<7} | {'NDCG@10':<8} | {'MRR':<7} | {'Diversity (ILD)':<15} | {'Latency (ms)':<12} |"
    sep = f"|{'-' * 26}|{'-' * 9}|{'-' * 9}|{'-' * 10}|{'-' * 9}|{'-' * 17}|{'-' * 14}|"
    print(header)
    print(sep)
    for m in benchmark_results:
        row = f"| {m['model']:<24} | {m['precision@5']:<7.4f} | {m['recall@10']:<7.4f} | {m['ndcg@10']:<8.4f} | {m['mrr']:<7.4f} | {m['diversity_ild']:<15.4f} | {m['avg_latency_ms']:<12.2f} |"
        print(row)
    print("=" * 80 + "\n")

    return benchmark_results


if __name__ == "__main__":
    run_ablation_study()
