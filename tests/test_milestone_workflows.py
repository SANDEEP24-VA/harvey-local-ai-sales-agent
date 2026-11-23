"""Tests for Milestone API endpoints and Reply Simulator (Phases C, D, G, H of EDXSO Milestone).

Covers:
- GET /api/companies/{id}/business-profile & POST /api/companies/{id}/generate-profile
- POST /api/research/scrape & GET /api/research/cache
- POST /api/prospects/{id}/classify-signals & GET /api/prospects/{id}/signals
- POST /api/prospects/{id}/products/{product_id}/generate-problem-statement & GET .../problem-statement
- GET /api/conversations/sample-library
- POST /api/conversations/test-classify (Reply Simulator with response preview)
- GET /api/system/hardware-profile
- Offline task queue: GET /api/tasks/queue, POST /api/tasks/queue, POST /api/tasks/queue/{id}/retry
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

import sellbird.config as sellbird_config
from sellbird.config import BrainConfig
import sellbird.dashboard as dash
from sellbird.models.company import Company
from sellbird.models.product import Product, ProductCreate
from sellbird.models.prospect import Prospect
from sellbird.state import StateManager

ORIGIN = "http://127.0.0.1:5555"
CONFIG = SimpleNamespace(
    brain=BrainConfig(provider="ollama"),
    persona=SimpleNamespace(name="Ethan", company="Sellbird", role="Founder", tone="candid and brief"),
    product=SimpleNamespace(name="Sellbird AI", description="Local sales platform", key_benefits=["Autonomous prospecting"], pricing="Free local tier"),
    compliance=SimpleNamespace(do_not_contact=[]),
    channels=SimpleNamespace(email=SimpleNamespace(require_approval=True, provider="smtp")),
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
        # Seed test data
        c = Company(id="c_test_1", name="Apex Systems", domain="apexsystems.test")
        await state.add_company(c)
        p = Prospect(
            id="p_test_1",
            company_id="c_test_1",
            first_name="Marcus",
            last_name="Vance",
            title="VP Sales",
            email="marcus@apexsystems.test",
            industry="Software",
        )
        await state.add_prospect(p)
        prod = ProductCreate(
            name="ApexGuard",
            description="Automated security for Kubernetes",
            value_proposition="10x faster compliance",
        )
        await state.add_product(prod)

    asyncio.run(_init())
    return {"token": dash.token_store.get(), "db": db}


@pytest.fixture
def client(env):
    with TestClient(dash.app, base_url=ORIGIN,
                    headers={"Authorization": f"Bearer {env['token']}"}) as c:
        yield c


def test_hardware_profile_api(client):
    res = client.get("/api/system/hardware-profile")
    assert res.status_code == 200
    data = res.json()
    assert "tier" in data
    assert "ram" in data
    assert "recommended_models" in data
    assert data["default_model"] == "qwen3:4b-instruct-2507-q4_K_M"


def test_business_profile_api(client):
    # Generate business profile
    res = client.post("/api/companies/c_test_1/generate-profile")
    assert res.status_code == 200
    data = res.json()
    assert data["company_id"] == "c_test_1"
    assert data["summary"]

    # Retrieve business profile
    get_res = client.get("/api/companies/c_test_1/business-profile")
    assert get_res.status_code == 200
    assert get_res.json()["company_id"] == "c_test_1"


def test_research_scrape_and_cache_api(client):
    # Scrape with offline_only to test fallback / cache
    res = client.post("/api/research/scrape", json={"url": "https://example.com/test", "offline_only": True})
    assert res.status_code == 200
    data = res.json()
    assert data["url"] == "https://example.com/test"

    # List cache
    cache_res = client.get("/api/research/cache")
    assert cache_res.status_code == 200
    assert isinstance(cache_res.json(), list)


def test_classify_signals_api(client):
    res = client.post(
        "/api/prospects/p_test_1/classify-signals",
        json={"research_text": "We are hiring Senior Engineers on WordPress.", "source_url": "https://apexsystems.test"},
    )
    assert res.status_code == 200
    signals = res.json()
    assert isinstance(signals, list)
    assert len(signals) > 0

    # Retrieve signals
    get_res = client.get("/api/prospects/p_test_1/signals")
    assert get_res.status_code == 200
    assert len(get_res.json()) >= len(signals)


def test_problem_statement_api(client):
    # Fetch existing product
    prods_res = client.get("/api/products")
    prod_id = prods_res.json()[0]["id"]

    # Generate problem statement
    res = client.post(
        f"/api/prospects/p_test_1/products/{prod_id}/generate-problem-statement",
        json={"research_text": "Apex Systems is modernizing enterprise infrastructure."},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["prospect_id"] == "p_test_1"
    assert data["product_id"] == prod_id
    assert data["business_problem"]
    assert len(data["pain_points"]) > 0

    # Retrieve problem statement
    get_res = client.get(f"/api/prospects/p_test_1/products/{prod_id}/problem-statement")
    assert get_res.status_code == 200
    assert get_res.json()["business_problem"] == data["business_problem"]


def test_sample_library_and_reply_simulator_api(client):
    # 1. Sample library
    lib_res = client.get("/api/conversations/sample-library")
    assert lib_res.status_code == 200
    samples = lib_res.json()
    assert len(samples) >= 5

    # 2. Reply Simulator - Opt out
    opt_out_res = client.post(
        "/api/conversations/test-classify",
        json={"reply_text": "Please unsubscribe me from this mailing list."},
    )
    assert opt_out_res.status_code == 200
    opt_data = opt_out_res.json()
    assert opt_data["intent"] == "unsubscribe"
    assert opt_data["is_opt_out"] is True
    assert opt_data["requires_human"] is True
    assert "opted out" in opt_data["preview_response"].lower()

    # 3. Reply Simulator - Interested
    interested_res = client.post(
        "/api/conversations/test-classify",
        json={"reply_text": "Sounds great! Do you have time for a demo on Thursday?"},
    )
    assert interested_res.status_code == 200
    int_data = interested_res.json()
    assert int_data["intent"] == "interested"
    assert int_data["is_opt_out"] is False


def test_offline_task_queue_api(client):
    # Queue task
    create_res = client.post(
        "/api/tasks/queue",
        json={"task_type": "profile_enrichment", "payload": {"prospect_id": "p_test_1"}},
    )
    assert create_res.status_code == 200
    task_id = create_res.json()["id"]

    # List queue
    list_res = client.get("/api/tasks/queue")
    assert list_res.status_code == 200
    tasks = list_res.json()
    assert any(t["id"] == task_id for t in tasks)

    # Retry task
    retry_res = client.post(f"/api/tasks/queue/{task_id}/retry")
    assert retry_res.status_code == 200
    assert retry_res.json()["status"] == "pending"
