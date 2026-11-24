"""Retrieval Evaluation Harness for Local RAG.

Measures and compares:
- Recall@3 and Recall@5
- Mean Reciprocal Rank (MRR)
- Groundedness / Faithfulness ratio
- Retrieval latency (ms)

Evaluates:
1. Dense-only retrieval (SentenceTransformers cosine similarity)
2. Lexical-only retrieval (SQLite FTS5 BM25)
3. Hybrid retrieval (Reciprocal Rank Fusion)
"""

import time
from typing import Any
from sellbird.embeddings import LocalEmbeddingEngine
from sellbird.research_engine import verify_evidence_quote


# Curated benchmark dataset of sales queries and ground truth relevant entity IDs
DEFAULT_BENCHMARK_DATASET = [
    {
        "query": "PostgreSQL automated failover high availability clusters",
        "relevant_ids": ["prod_pg_features", "prod_pg_overview"],
        "product_id": "prod_pg",
    },
    {
        "query": "Kubernetes native CRD operator deployment",
        "relevant_ids": ["prod_pg_features"],
        "product_id": "prod_pg",
    },
    {
        "query": "too expensive out of our current budget",
        "relevant_ids": ["prod_pg_obj_price"],
        "product_id": "prod_pg",
    },
    {
        "query": "enterprise SOC2 compliance and zero-trust encryption",
        "relevant_ids": ["prod_sec_overview", "prod_sec_features"],
        "product_id": "prod_sec",
    },
    {
        "query": "Hiring backend engineers with Python and Go experience",
        "relevant_ids": ["ats_eng_01", "ats_eng_02"],
        "product_id": None,
    },
]


def evaluate_retrieval(
    engine: LocalEmbeddingEngine,
    dataset: list[dict[str, Any]] | None = None,
    mode: str = "hybrid",
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate retrieval performance across benchmark queries for a specific mode ('dense', 'lexical', 'hybrid')."""
    data = dataset or DEFAULT_BENCHMARK_DATASET
    if not data:
        return {"recall_at_3": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "avg_latency_ms": 0.0}

    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    latencies = []

    for item in data:
        q = item["query"]
        relevant = set(item.get("relevant_ids", []))
        pid = item.get("product_id")

        t0 = time.perf_counter()
        if mode == "dense":
            results = engine.search_dense(q, product_id=pid, top_k=top_k)
        elif mode == "lexical":
            results = engine.search_lexical(q, product_id=pid, top_k=top_k)
        else:
            results = engine.search_hybrid(q, product_id=pid, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        retrieved_ids = [r.get("entity_id", "") for r in results]

        # Calculate hits@3
        if any(rid in relevant for rid in retrieved_ids[:3]):
            hits_at_3 += 1

        # Calculate hits@5
        if any(rid in relevant for rid in retrieved_ids[:5]):
            hits_at_5 += 1

        # Calculate Reciprocal Rank
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    n = len(data)
    return {
        "mode": mode,
        "query_count": n,
        "recall_at_3": round(hits_at_3 / n, 4),
        "recall_at_5": round(hits_at_5 / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "avg_latency_ms": round(sum(latencies) / n, 2),
    }


def evaluate_groundedness(
    source_chunks: list[str],
    generated_claims: list[str],
) -> dict[str, Any]:
    """Measure groundedness ratio of generated claims against source chunks using verbatim verification."""
    if not generated_claims:
        return {"groundedness_ratio": 1.0, "verified_count": 0, "total_count": 0}

    combined_source = " ".join(source_chunks)
    verified = 0

    for claim in generated_claims:
        if verify_evidence_quote(combined_source, claim):
            verified += 1

    ratio = verified / len(generated_claims)
    return {
        "groundedness_ratio": round(ratio, 4),
        "verified_count": verified,
        "total_count": len(generated_claims),
    }


def compare_retrieval_modes(
    engine: LocalEmbeddingEngine,
    dataset: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run full comparative evaluation across dense, lexical, and hybrid retrieval."""
    dense_metrics = evaluate_retrieval(engine, dataset=dataset, mode="dense", top_k=5)
    lexical_metrics = evaluate_retrieval(engine, dataset=dataset, mode="lexical", top_k=5)
    hybrid_metrics = evaluate_retrieval(engine, dataset=dataset, mode="hybrid", top_k=5)

    return {
        "dense": dense_metrics,
        "lexical": lexical_metrics,
        "hybrid": hybrid_metrics,
        "hybrid_advantage": {
            "recall_at_5_improvement": round(hybrid_metrics["recall_at_5"] - max(dense_metrics["recall_at_5"], lexical_metrics["recall_at_5"]), 4),
            "mrr_improvement": round(hybrid_metrics["mrr"] - max(dense_metrics["mrr"], lexical_metrics["mrr"]), 4),
        },
    }
