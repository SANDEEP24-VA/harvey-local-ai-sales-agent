"""Tests for Strict Product and Prospect Multi-Tenant Isolation in Local RAG.

Verifies:
- Product A cannot retrieve Product B pricing.
- Product A cannot retrieve Product B internal features.
- Product-specific objection playbooks do not leak across products.
- Prospect-specific web research is isolated to that prospect.
- Explicitly shared global sales playbooks remain accessible.
"""

import os
import tempfile
import pytest

from sellbird.embeddings import LocalEmbeddingEngine


def test_strict_product_isolation_pricing_and_features():
    """Verify that Product A cannot retrieve Product B pricing or features."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        engine = LocalEmbeddingEngine(db_path=temp_db)

        # Index Product Alpha (Database tool)
        engine.index_document(
            entity_type="product_doc",
            entity_id="alpha_pricing",
            text="Product Alpha pricing is $99 per month with unlimited read replicas.",
            product_id="prod_alpha",
        )
        engine.index_document(
            entity_type="product_feature",
            entity_id="alpha_feature",
            text="Product Alpha features distributed multi-master consensus protocol.",
            product_id="prod_alpha",
        )

        # Index Product Beta (Cybersecurity tool)
        engine.index_document(
            entity_type="product_doc",
            entity_id="beta_pricing",
            text="Product Beta pricing is $4,999 annual contract for endpoint protection.",
            product_id="prod_beta",
        )
        engine.index_document(
            entity_type="product_feature",
            entity_id="beta_feature",
            text="Product Beta features automated SOC2 audit report generation.",
            product_id="prod_beta",
        )

        # Index Global Sales Playbook (Accessible to both)
        engine.index_document(
            entity_type="sales_playbook",
            entity_id="global_rules",
            text="General cold email rule: keep initial touch under 60 words.",
            product_id=None,
            is_global=True,
        )

        # 1. Query for pricing while scoped to Product Alpha
        res_alpha_pricing = engine.search_hybrid("pricing monthly contract cost", product_id="prod_alpha")
        assert len(res_alpha_pricing) > 0
        # Must ONLY find Product Alpha pricing, never Product Beta pricing
        for r in res_alpha_pricing:
            assert r["product_id"] in ("prod_alpha", "", None)
            assert "$4,999" not in r["chunk_text"]
            assert "Product Beta" not in r["chunk_text"]

        # 2. Query for SOC2 while scoped to Product Alpha
        res_alpha_soc2 = engine.search_hybrid("SOC2 audit compliance", product_id="prod_alpha")
        # Product Alpha has no SOC2 feature, Product Beta does; Product Alpha query MUST NOT leak Product Beta
        for r in res_alpha_soc2:
            assert r["product_id"] != "prod_beta"
            assert "Product Beta" not in r["chunk_text"]

        # 3. Query for pricing while scoped to Product Beta
        res_beta_pricing = engine.search_hybrid("pricing cost per month", product_id="prod_beta")
        assert len(res_beta_pricing) > 0
        for r in res_beta_pricing:
            assert r["product_id"] in ("prod_beta", "", None)
            assert "$99" not in r["chunk_text"]
            assert "Product Alpha" not in r["chunk_text"]

        # 4. Verify Global Playbook is accessible when scoped to Product Alpha
        res_global = engine.search_hybrid("cold email rule words", product_id="prod_alpha")
        assert any(r["entity_id"] == "global_rules" for r in res_global)
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)


def test_objection_playbook_isolation():
    """Verify that objection playbooks tailored for Product A do not leak into Product B outreach."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        engine = LocalEmbeddingEngine(db_path=temp_db)

        engine.index_document(
            entity_type="objection_playbook",
            entity_id="obj_alpha",
            text="Objection: already using AWS Aurora. Response: Alpha cuts Aurora IOPS costs by 60%.",
            product_id="prod_alpha",
        )
        engine.index_document(
            entity_type="objection_playbook",
            entity_id="obj_beta",
            text="Objection: already using CrowdStrike. Response: Beta provides automated agentless scanning.",
            product_id="prod_beta",
        )

        # Query objection while scoped to Product Beta
        res_beta_obj = engine.search_hybrid("already using an incumbent solution", product_id="prod_beta")
        for r in res_beta_obj:
            assert "Aurora" not in r["chunk_text"]
            assert r["product_id"] != "prod_alpha"

        # Query objection while scoped to Product Alpha
        res_alpha_obj = engine.search_hybrid("already using an incumbent solution", product_id="prod_alpha")
        for r in res_alpha_obj:
            assert "CrowdStrike" not in r["chunk_text"]
            assert r["product_id"] != "prod_beta"
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)


def test_prospect_research_isolation():
    """Verify that prospect-specific research is strictly filtered by prospect_id."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        engine = LocalEmbeddingEngine(db_path=temp_db)

        # Research on Prospect 1
        engine.index_document(
            entity_type="web_research",
            entity_id="res_01",
            text="CEO John Doe at Stark Industries announced private confidential acquisition of drones.",
            product_id="prod_defense",
            metadata={"prospect_id": "prospect_01"},
        )

        # Research on Prospect 2
        engine.index_document(
            entity_type="web_research",
            entity_id="res_02",
            text="CTO Jane Smith at Wayne Enterprises announced new satellite communications network.",
            product_id="prod_defense",
            metadata={"prospect_id": "prospect_02"},
        )

        # Search for drone acquisition scoped to Prospect 2
        res = engine.search_hybrid(
            "acquisition of drones",
            product_id="prod_defense",
            prospect_id="prospect_02",
        )
        # Must NOT leak Prospect 1's research to Prospect 2
        for r in res:
            assert r["metadata"].get("prospect_id") != "prospect_01"
            assert "Stark Industries" not in r["chunk_text"]
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)
