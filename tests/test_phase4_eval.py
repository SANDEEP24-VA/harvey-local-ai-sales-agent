"""Tests and benchmark execution for Phase 4 Retrieval Evaluation Harness.

Measures:
- Recall@3 and Recall@5
- Mean Reciprocal Rank (MRR)
- Groundedness / Faithfulness ratio
- Latency (ms)
- Comparative analysis: Dense vs Lexical vs Hybrid (RRF)
"""

import os
import tempfile
import pytest

from sellbird.embeddings import LocalEmbeddingEngine
from sellbird.retrieval_eval import (
    DEFAULT_BENCHMARK_DATASET,
    compare_retrieval_modes,
    evaluate_groundedness,
    evaluate_retrieval,
)


def test_retrieval_evaluation_benchmark_comparison():
    """Verify evaluation harness measures Recall@3, Recall@5, MRR, latency, and compares modes."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        engine = LocalEmbeddingEngine(db_path=temp_db)

        # Pre-seed engine with benchmark documents
        engine.index_document(
            entity_type="product_feature",
            entity_id="prod_pg_features",
            text="PostgreSQL automated failover, high availability clusters, and Kubernetes native CRD operator deployment.",
            product_id="prod_pg",
        )
        engine.index_document(
            entity_type="product_doc",
            entity_id="prod_pg_overview",
            text="High performance distributed Postgres architecture with connection pooling and automated backups.",
            product_id="prod_pg",
        )
        engine.index_document(
            entity_type="objection_playbook",
            entity_id="prod_pg_obj_price",
            text="Objection: too expensive or out of our current budget. Response: Delivers 3x ROI through infrastructure consolidation.",
            product_id="prod_pg",
        )
        engine.index_document(
            entity_type="product_doc",
            entity_id="prod_sec_overview",
            text="Enterprise SOC2 compliance certification and zero-trust encryption for cloud workloads.",
            product_id="prod_sec",
        )
        engine.index_document(
            entity_type="product_feature",
            entity_id="prod_sec_features",
            text="Continuous automated security compliance monitoring with instant vulnerability alerts.",
            product_id="prod_sec",
        )
        engine.index_document(
            entity_type="ats_posting",
            entity_id="ats_eng_01",
            text="We are hiring Senior Backend Engineers with extensive Python and Go distributed systems experience.",
            product_id=None,
        )

        comparison = compare_retrieval_modes(engine, DEFAULT_BENCHMARK_DATASET)

        # 1. Dense metrics
        dense = comparison["dense"]
        assert "recall_at_3" in dense
        assert "recall_at_5" in dense
        assert "mrr" in dense
        assert "avg_latency_ms" in dense

        # 2. Lexical metrics
        lexical = comparison["lexical"]
        assert "recall_at_3" in lexical
        assert "recall_at_5" in lexical
        assert "mrr" in lexical
        assert "avg_latency_ms" in lexical

        # 3. Hybrid metrics
        hybrid = comparison["hybrid"]
        assert hybrid["recall_at_3"] > 0.0
        assert hybrid["recall_at_5"] > 0.0
        assert hybrid["mrr"] > 0.0
        assert hybrid["avg_latency_ms"] < 200.0  # CPU target <200ms

        print(f"\n--- Phase 4 Retrieval Evaluation Results ---")
        print(f"Dense  : Recall@3={dense['recall_at_3']}, Recall@5={dense['recall_at_5']}, MRR={dense['mrr']}, Latency={dense['avg_latency_ms']}ms")
        print(f"Lexical: Recall@3={lexical['recall_at_3']}, Recall@5={lexical['recall_at_5']}, MRR={lexical['mrr']}, Latency={lexical['avg_latency_ms']}ms")
        print(f"Hybrid : Recall@3={hybrid['recall_at_3']}, Recall@5={hybrid['recall_at_5']}, MRR={hybrid['mrr']}, Latency={hybrid['avg_latency_ms']}ms")
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)


def test_groundedness_ratio_evaluation():
    """Verify that evaluate_groundedness calculates accurate groundedness ratios."""
    source_chunks = [
        "CloudScale operates on PostgreSQL 16 with native pgvector support for AI embeddings.",
        "Pricing begins at $499 per node per month with no minimum contract length.",
    ]

    claims = [
        "CloudScale operates on PostgreSQL 16",  # Grounded
        "native pgvector support for AI embeddings",  # Grounded
        "pricing begins at $499 per node per month",  # Grounded
        "includes free dedicated 24/7 DBA support in Antarctica",  # Unverified
    ]

    res = evaluate_groundedness(source_chunks, claims)
    assert res["total_count"] == 4
    assert res["verified_count"] == 3
    assert res["groundedness_ratio"] == 0.75
