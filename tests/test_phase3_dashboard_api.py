"""Tests for Phase 3 Dashboard API Endpoints.

Verifies:
- POST /api/discovery/search (free sources, paid provider rejection)
- POST /api/research/crawl-company (offline and structured extraction)
- GET /api/research/cache/inspect
- POST /api/embeddings/index and POST /api/embeddings/search
- POST /api/products/{product_id}/opportunity-score
"""

import asyncio
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient

import sellbird.config as sellbird_config
from sellbird.config import BrainConfig
import sellbird.dashboard as dash
from sellbird.models.product import Product
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


def test_discovery_search_rejects_paid_providers(client):
    """Verify POST /api/discovery/search returns 400 when paid provider requested."""
    res = client.post(
        "/api/discovery/search",
        json={"provider": "serper", "industry": "Dentists"},
    )
    assert res.status_code == 400
    data = res.json()
    assert "error" in data
    assert "Prohibited discovery provider" in data["error"]


def test_discovery_search_free_providers(client):
    """Verify POST /api/discovery/search accepts free provider requests."""
    res = client.post(
        "/api/discovery/search",
        json={"provider": "osm", "industry": "software", "location": "Denver, CO", "limit": 5},
    )
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert data["is_zero_cost"] is True


def test_crawl_company_offline(client):
    """Verify POST /api/research/crawl-company runs in offline mode without network errors."""
    res = client.post(
        "/api/research/crawl-company",
        json={"domain": "example-offline-company.com", "offline_only": True},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["domain"] == "example-offline-company.com"
    assert data["from_cache"] is True
    assert "technologies" in data
    assert "decision_makers" in data


def test_inspect_research_cache(client):
    """Verify GET /api/research/cache/inspect returns structured cache entries."""
    res = client.get("/api/research/cache/inspect?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)


def test_embeddings_index_and_search_api(client):
    """Verify vector indexing and semantic search via API."""
    # 1. Index document
    idx_res = client.post(
        "/api/embeddings/index",
        json={
            "entity_type": "product_feature",
            "entity_id": "feat_01",
            "text": "Automated security scanning with zero cloud exposure for air-gapped systems.",
            "metadata": {"category": "security"},
        },
    )
    assert idx_res.status_code == 200
    assert idx_res.json()["status"] == "indexed"

    # 2. Search document
    search_res = client.post(
        "/api/embeddings/search",
        json={"query": "air-gapped security scanning", "top_k": 3},
    )
    assert search_res.status_code == 200
    results = search_res.json()["results"]
    assert len(results) >= 1
    assert results[0]["entity_id"] == "feat_01"


def test_product_opportunity_score_api(client):
    """Verify POST /api/products/{product_id}/opportunity-score returns versioned score."""
    # 1. Create product via client
    prod_res = client.post(
        "/api/products",
        json={
            "name": "Security Auditing Agent",
            "value_proposition": "Automated local penetration test reports.",
        },
    )
    assert prod_res.status_code == 201
    created_prod = prod_res.json()
    product_id = created_prod["id"]

    # 2. Calculate opportunity score
    res = client.post(
        f"/api/products/{product_id}/opportunity-score",
        json={
            "icp_fit": 85.0,
            "signals": [
                {
                    "signal_code": "HIRING_ROLE",
                    "confidence": 0.9,
                    "verification_status": "verified_observation",
                    "result": "Hiring CISO",
                }
            ],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["formula_version"] == "v1.0"
    assert data["is_provisional"] is False
    assert data["icp_component"] == 59.5  # 70% of 85.0
    assert data["opportunity_score"] > 59.5
