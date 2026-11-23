"""Tests for Dashboard Product Management APIs (EDXSO Phase 2).

Covers:
- GET /api/products
- POST /api/products
- GET /api/products/{id}
- PUT /api/products/{id}
- DELETE /api/products/{id} (archive)
- GET /api/products/{id}/icp & POST /api/products/{id}/icp
- POST /api/products/{id}/generate-icp (mocked local Ollama)
- POST /api/products/{id}/generate-intelligence (mocked local Ollama)
- POST /api/products/{id}/match-prospects
- GET /api/products/{id}/prospects
- GET /api/products/{id}/evidence
- GET /api/products/{id}/campaigns
"""

import asyncio
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import sellbird.config as sellbird_config
from sellbird.config import BrainConfig
import sellbird.dashboard as dash
from sellbird.models.product import ProductICP, ProspectProductMatch, ProductResearchEvidence
from sellbird.state import StateManager

ORIGIN = "http://127.0.0.1:5555"
CONFIG = SimpleNamespace(
    brain=BrainConfig(provider="ollama"),
    icp=SimpleNamespace(industries=["technology"], titles=["CTO"], hiring_signals=[]),
    compliance=SimpleNamespace(do_not_contact=[]),
    channels=SimpleNamespace(email=SimpleNamespace(require_approval=True)),
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db = data_dir / "sellbird.db"
    monkeypatch.setattr(dash, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(dash, "DB_PATH", db)
    monkeypatch.setattr(dash, "PID_FILE", data_dir / "sellbird.pid")
    monkeypatch.setattr(dash, "LOG_FILE", data_dir / "sellbird.log")
    monkeypatch.setattr(sellbird_config, "load_config", lambda *a, **k: CONFIG)
    dash.token_store.configure(data_dir / "dashboard-token")

    async def _init():
        state = StateManager(db_path=str(db))
        await state.init_db()

    asyncio.run(_init())
    return {"token": dash.token_store.get(), "db": db}


@pytest.fixture
def client(env):
    with TestClient(dash.app, base_url=ORIGIN,
                    headers={"Authorization": f"Bearer {env['token']}"}) as c:
        yield c


def test_get_products_initially_empty(client):
    res = client.get("/api/products")
    assert res.status_code == 200
    assert res.json() == []


def test_create_and_get_product(client):
    payload = {
        "name": "CloudGuard AI",
        "description": "Automated cloud compliance monitoring",
        "category": "Cybersecurity",
        "value_proposition": "Continuous SOC2 compliance in minutes",
        "features": ["Automated auditing", "Drift detection", "Zero agent"],
        "pricing_model": "subscription",
        "pricing_details": "$1500/mo",
    }
    create_res = client.post("/api/products", json=payload)
    assert create_res.status_code == 201
    prod = create_res.json()
    assert prod["name"] == "CloudGuard AI"
    assert prod["id"]
    assert prod["is_active"] is True

    # Get by ID
    get_res = client.get(f"/api/products/{prod['id']}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "CloudGuard AI"

    # List products
    list_res = client.get("/api/products")
    assert list_res.status_code == 200
    products = list_res.json()
    assert len(products) == 1
    assert products[0]["id"] == prod["id"]


def test_create_product_validation_error(client):
    # Missing required 'name'
    res = client.post("/api/products", json={"description": "No name provided"})
    assert res.status_code == 400
    assert "Invalid product data" in res.json()["error"]


def test_get_nonexistent_product_returns_404(client):
    res = client.get("/api/products/non-existent-id-999")
    assert res.status_code == 404
    assert res.json()["error"] == "Product not found"


def test_update_product(client):
    # Create product
    create_res = client.post("/api/products", json={"name": "DevSecOps Pro", "description": "Initial description"})
    prod_id = create_res.json()["id"]

    # Update product
    update_res = client.put(f"/api/products/{prod_id}", json={
        "name": "DevSecOps Pro Suite",
        "is_active": False,
        "features": ["Kubernetes security", "Terraform analysis"],
    })
    assert update_res.status_code == 200
    updated = update_res.json()
    assert updated["name"] == "DevSecOps Pro Suite"
    assert updated["is_active"] is False
    assert "Kubernetes security" in updated["features"]

    # Update non-existent product
    bad_res = client.put("/api/products/bad-id", json={"name": "New"})
    assert bad_res.status_code == 404


def test_archive_product(client):
    create_res = client.post("/api/products", json={"name": "Legacy Tool", "description": "To be archived"})
    prod_id = create_res.json()["id"]

    # Archive
    del_res = client.delete(f"/api/products/{prod_id}")
    assert del_res.status_code == 200
    assert del_res.json()["archived"] is True

    # Check normal listing excludes archived product
    active_res = client.get("/api/products")
    assert not any(p["id"] == prod_id for p in active_res.json())

    # Check listing with include_archived=true includes it
    all_res = client.get("/api/products?include_archived=true")
    assert any(p["id"] == prod_id for p in all_res.json())


def test_product_icp_get_and_save(client):
    create_res = client.post("/api/products", json={"name": "Fintech API", "description": "Banking APIs"})
    prod_id = create_res.json()["id"]

    # Initially None
    res = client.get(f"/api/products/{prod_id}/icp")
    assert res.status_code == 200
    assert res.json() is None

    # Save ICP
    icp_payload = {
        "target_industries": ["Fintech", "Banking", "Neobanks"],
        "company_size": "50-200",
        "job_titles": ["VP Engineering", "Head of Payments"],
        "pain_points": ["Legacy payment gateways", "High interchange fees"],
        "buying_signals": ["Expanding internationally", "Series B funding"],
        "exclusion_criteria": ["Crypto startups under 10 employees"],
    }
    save_res = client.post(f"/api/products/{prod_id}/icp", json=icp_payload)
    assert save_res.status_code == 200
    saved = save_res.json()
    assert saved["product_id"] == prod_id
    assert "Fintech" in saved["target_industries"]
    assert "VP Engineering" in saved["job_titles"]

    # Retrieve again
    get_res = client.get(f"/api/products/{prod_id}/icp")
    assert get_res.status_code == 200
    assert get_res.json()["target_industries"] == ["Fintech", "Banking", "Neobanks"]


@patch("sellbird.product_intelligence.generate_product_icp")
def test_generate_product_icp_route(mock_generate, client):
    create_res = client.post("/api/products", json={"name": "SalesBot", "description": "AI sales agent"})
    prod_id = create_res.json()["id"]

    mock_icp = ProductICP(
        product_id=prod_id,
        target_industries=["B2B SaaS", "E-commerce"],
        company_size="10-50",
        job_titles=["Head of Sales", "CRO"],
        pain_points=["Low SDR response rate"],
        buying_signals=["Hiring sales reps"],
        exclusion_criteria=["B2C companies"],
    )
    mock_generate.return_value = mock_icp

    res = client.post(f"/api/products/{prod_id}/generate-icp")
    assert res.status_code == 200
    data = res.json()
    assert data["product_id"] == prod_id
    assert "B2B SaaS" in data["target_industries"]
    assert "CRO" in data["job_titles"]


@patch("sellbird.product_intelligence.generate_product_intelligence")
def test_generate_product_intelligence_route(mock_generate, client):
    create_res = client.post("/api/products", json={"name": "DocuSign Competitor", "description": "eSignatures"})
    prod_id = create_res.json()["id"]

    mock_generate.return_value = {
        "problem_statements": ["Legacy eSignatures cost $40/user/mo"],
        "value_propositions": ["Simple, legally binding signatures for $5/user/mo"],
        "differentiators": ["Unlimited templates, self-hosted option"],
        "objection_responses": {
            "price": "We are 80% cheaper than DocuSign.",
            "compliance": "SOC2 and HIPAA compliant out of the box.",
        },
    }

    res = client.post(f"/api/products/{prod_id}/generate-intelligence")
    assert res.status_code == 200
    intel = res.json()
    assert "Legacy eSignatures cost $40/user/mo" in intel["problem_statements"]
    assert "price" in intel["objection_responses"]


@patch("sellbird.product_intelligence.match_prospect_product")
def test_match_prospects_and_retrieve_matches(mock_match, client, env):
    create_res = client.post("/api/products", json={"name": "APM Tool", "description": "App monitoring"})
    prod_id = create_res.json()["id"]

    # Insert a prospect into database
    with sqlite3.connect(env["db"]) as c:
        c.execute(
            "INSERT INTO prospects (id, company_id, first_name, last_name, title, email, status, created_at) "
            "VALUES ('p_apm', 'c_apm', 'Alice', 'Engineer', 'DevOps Lead', 'alice@apm.test', 'new', '2026-11-22T00:00:00')"
        )

    # Mock match return
    mock_match.return_value = (
        ProspectProductMatch(
            prospect_id="p_apm",
            product_id=prod_id,
            fit_score=88.5,
            matching_status="qualified",
            matching_reasons=["DevOps Lead role matches APM buyer profile exactly."],
            ai_analysis={"confidence": "high"},
        ),
        [
            ProductResearchEvidence(
                product_id=prod_id,
                prospect_id="p_apm",
                evidence_source="title_match",
                evidence_description="Role is DevOps Lead",
            )
        ],
    )

    # Trigger matching
    match_res = client.post(f"/api/products/{prod_id}/match-prospects")
    assert match_res.status_code == 200
    assert match_res.json()["matched"] >= 1

    # Retrieve matched prospects
    prospects_res = client.get(f"/api/products/{prod_id}/prospects")
    assert prospects_res.status_code == 200
    prospects_data = prospects_res.json()
    assert len(prospects_data) >= 1
    match_entry = prospects_data[0]
    assert match_entry["match"]["fit_score"] == 88.5
    assert match_entry["match"]["matching_status"] == "qualified"
    assert match_entry["prospect"]["first_name"] == "Alice"

    # Retrieve evidence
    evidence_res = client.get(f"/api/products/{prod_id}/evidence")
    assert evidence_res.status_code == 200
    ev_list = evidence_res.json()
    assert len(ev_list) >= 1
    assert ev_list[0]["evidence_source"] == "title_match"
    assert ev_list[0]["evidence_description"] == "Role is DevOps Lead"


def test_get_product_campaigns(client, env):
    create_res = client.post("/api/products", json={"name": "Security Scanner", "description": "Vulnerability scanning"})
    prod_id = create_res.json()["id"]

    # Insert a campaign linked to this product
    with sqlite3.connect(env["db"]) as c:
        c.execute(
            "INSERT INTO campaigns (id, name, status, product_id, sequence_json, prospect_ids_json, created_at) "
            "VALUES ('camp_prod1', 'Scanner Launch', 'active', ?, ?, ?, '2026-11-22T00:00:00')",
            (prod_id, json.dumps([{"step": 1, "subject": "Automate security"}]), json.dumps(["p1", "p2"]))
        )

    res = client.get(f"/api/products/{prod_id}/campaigns")
    assert res.status_code == 200
    camps = res.json()
    assert len(camps) == 1
    assert camps[0]["id"] == "camp_prod1"
    assert camps[0]["product_id"] == prod_id
    assert camps[0]["sequence"][0]["subject"] == "Automate security"
    assert camps[0]["prospect_ids"] == ["p1", "p2"]
