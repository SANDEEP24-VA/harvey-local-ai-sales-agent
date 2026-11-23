"""Tests for Phase 3 Opportunity Scoring and Local Embeddings Engine.

Verifies:
- Versioned, explainable opportunity score formula (70% ICP fit + 30% verified buying signals)
- Provisional flag when signals are missing (no negative intent penalty)
- LocalEmbeddingEngine: SentenceTransformers local model, SQLite storage, cosine similarity, and semantic vector search
"""

import os
import tempfile
import pytest

from sellbird.embeddings import LocalEmbeddingEngine
from sellbird.models.product import Product
from sellbird.product_intelligence import (
    OPPORTUNITY_SCORE_FORMULA_VERSION,
    calculate_opportunity_score,
)


def test_opportunity_score_provisional_when_no_signals():
    """Verify that when no verified signals exist, the score is marked provisional and does not penalize missing data."""
    product = Product(
        id="prod_01",
        name="Enterprise Security Suite",
        description="Zero-trust security for hybrid cloud.",
        value_proposition="Continuous compliance and threat detection.",
    )

    # 80% ICP fit with zero signals
    res = calculate_opportunity_score(
        icp_fit=80.0,
        verified_signals=[],
        product=product,
    )

    assert res["formula_version"] == OPPORTUNITY_SCORE_FORMULA_VERSION
    assert res["is_provisional"] is True
    # The score should reflect ICP fit without being suppressed to zero
    assert res["opportunity_score"] == 80.0
    assert "Provisional opportunity score" in res["explanation"]
    assert "Missing research is not penalized" in res["explanation"]


def test_opportunity_score_with_verified_signals():
    """Verify opportunity score combines 70% ICP fit and 30% signal contribution."""
    product = Product(
        id="prod_02",
        name="API Automation Tool",
        description="Low-code API orchestration for developers.",
        value_proposition="Cut integration time by 80%.",
    )

    signals = [
        {
            "signal_code": "HIRING_ROLE",
            "confidence": 0.95,
            "verification_status": "verified_observation",
            "result": "Hiring backend engineers",
        },
        {
            "signal_code": "TECH_STACK",
            "confidence": 0.90,
            "verification_status": "verified_observation",
            "result": "Detected React and Next.js",
        },
        {
            "signal_code": "DECISION_MAKER_TITLE",
            "confidence": 0.95,
            "verification_status": "verified_observation",
            "result": "VP Engineering",
        },
    ]

    res = calculate_opportunity_score(
        icp_fit=80.0,
        verified_signals=signals,
        product=product,
    )

    assert res["is_provisional"] is False
    assert res["icp_component"] == 56.0  # 70% of 80.0
    assert res["signal_component"] > 0.0
    assert res["opportunity_score"] == round(res["icp_component"] + res["signal_component"], 2)
    assert len(res["signal_breakdown"]) == 3
    assert res["status"] in ("high_priority", "moderate_priority")


def test_local_embeddings_engine_lifecycle():
    """Verify LocalEmbeddingEngine generates vectors, indexes documents, and performs semantic search."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        engine = LocalEmbeddingEngine(db_path=temp_db)

        # 1. Vector generation
        vec = engine.embed_text("Cloud database management and PostgreSQL scaling")
        assert len(vec) == 384
        assert isinstance(vec[0], float)

        # 2. Batch vector generation
        batch_vecs = engine.embed_batch(["First query", "Second query"])
        assert len(batch_vecs) == 2
        assert len(batch_vecs[0]) == 384

        # 3. Cosine similarity
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        assert LocalEmbeddingEngine.cosine_similarity(v1, v2) == pytest.approx(1.0)
        assert LocalEmbeddingEngine.cosine_similarity(v1, v3) == pytest.approx(0.0)

        # 4. Document indexing
        doc_text = """
        PostgreSQL High Availability.
        We provide automated failover, replication, and backup management for enterprise Postgres clusters.

        Kubernetes Native Operators.
        Deploy and manage PostgreSQL natively on Kubernetes with custom resource definitions.
        """
        chunks = engine.index_document(
            entity_type="product_doc",
            entity_id="prod_pg_01",
            text=doc_text,
            metadata={"product": "Postgres Operator"},
        )
        assert chunks >= 2

        # 5. Semantic search
        results = engine.search("PostgreSQL backup and replication failover", top_k=2)
        assert len(results) >= 1
        assert results[0]["entity_type"] == "product_doc"
        assert results[0]["entity_id"] == "prod_pg_01"
        assert results[0]["similarity"] > 0.20
    finally:
        if os.path.exists(temp_db):
            os.unlink(temp_db)
