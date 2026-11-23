"""Tests for SQLite Migration v18 and new Milestone CRUD operations."""

import pytest
from sellbird.models.business_profile import CompanyBusinessProfile
from sellbird.models.company import Company
from sellbird.models.product import ProductCreate
from sellbird.models.prospect import Prospect
from sellbird.models.research import (
    ClassifiedBuyingSignal,
    OfflineTask,
    ProspectProblemStatement,
    ResearchCacheItem,
)
from sellbird.state import StateManager


@pytest.mark.asyncio
async def test_business_profile_crud(tmp_path):
    state = StateManager(str(tmp_path / "test.db"))
    await state.init_db()

    # Add company
    c = Company(id="c_bp1", name="Apex Cloud", domain="apexcloud.test")
    await state.add_company(c)

    # Save business profile
    bp = CompanyBusinessProfile(
        company_id="c_bp1",
        summary="Apex Cloud provides enterprise cloud orchestration solutions.",
        business_model="B2B SaaS subscription",
        core_offerings=["Kubernetes Manager", "Multi-Cloud Cost Guard"],
        target_markets=["Enterprise IT", "Financial Services"],
        tech_stack=["Go", "Python", "Kubernetes", "AWS"],
        operational_scale="Mid-Market (150-500 employees)",
        raw_profile={"confidence": 0.95, "source": "local_ai"},
    )
    prof_id = await state.save_business_profile(bp)
    assert prof_id

    # Retrieve business profile
    retrieved = await state.get_business_profile("c_bp1")
    assert retrieved is not None
    assert retrieved.company_id == "c_bp1"
    assert "enterprise cloud orchestration" in retrieved.summary
    assert "Kubernetes Manager" in retrieved.core_offerings
    assert "Go" in retrieved.tech_stack
    assert retrieved.raw_profile.get("confidence") == 0.95

    # Update business profile via upsert
    bp.summary = "Apex Cloud is an industry-leading Kubernetes management platform."
    await state.save_business_profile(bp)
    updated = await state.get_business_profile("c_bp1")
    assert "industry-leading" in updated.summary


@pytest.mark.asyncio
async def test_research_cache_crud(tmp_path):
    state = StateManager(str(tmp_path / "test.db"))
    await state.init_db()

    cache_item = ResearchCacheItem(
        url="https://apexcloud.test/careers",
        domain="apexcloud.test",
        content_hash="hash12345",
        content_type="text/html",
        raw_content="<html><body><h1>Hiring DevOps Leads</h1></body></html>",
        extracted_text="Hiring DevOps Leads",
        metadata={"title": "Careers at Apex", "roles_detected": 1},
        status_code=200,
    )

    item_id = await state.save_research_cache(cache_item)
    assert item_id

    # Retrieve by URL
    retrieved = await state.get_research_cache("https://apexcloud.test/careers")
    assert retrieved is not None
    assert retrieved.domain == "apexcloud.test"
    assert retrieved.extracted_text == "Hiring DevOps Leads"
    assert retrieved.metadata.get("roles_detected") == 1

    # List items
    items = await state.list_research_cache(domain="apexcloud.test")
    assert len(items) == 1
    assert items[0].url == "https://apexcloud.test/careers"


@pytest.mark.asyncio
async def test_classified_buying_signals_crud(tmp_path):
    state = StateManager(str(tmp_path / "test.db"))
    await state.init_db()

    # Add prospect
    p = Prospect(id="p_sig1", email="dan@apex.test", status="new")
    await state.add_prospect(p)

    sig = ClassifiedBuyingSignal(
        prospect_id="p_sig1",
        product_id="prod_sec1",
        signal_code="HIRING_ROLE",
        category="Hiring",
        result="Actively recruiting Senior Cloud Security Engineers",
        confidence=0.92,
        supporting_evidence="Job listing 'Senior Cloud Security Engineer' on /careers",
        source_url="https://apexcloud.test/careers",
        verification_status="verified_observation",
    )
    sig_id = await state.add_classified_buying_signal(sig)
    assert sig_id

    # Retrieve signals
    signals = await state.get_classified_buying_signals("p_sig1")
    assert len(signals) == 1
    s = signals[0]
    assert s.signal_code == "HIRING_ROLE"
    assert s.confidence == 0.92
    assert s.verification_status == "verified_observation"
    assert "Senior Cloud Security Engineer" in s.supporting_evidence


@pytest.mark.asyncio
async def test_prospect_problem_statements_crud(tmp_path):
    state = StateManager(str(tmp_path / "test.db"))
    await state.init_db()

    # Add product & prospect
    prod_id = await state.add_product(ProductCreate(name="SecScan Pro", description="Security tool"))
    p = Prospect(id="p_ps1", email="cto@apex.test", status="new")
    await state.add_prospect(p)

    stmt = ProspectProblemStatement(
        prospect_id="p_ps1",
        product_id=prod_id,
        business_problem="Multi-cloud Kubernetes cluster drift leads to security audit delays.",
        pain_points=["Manual compliance audits", "Lack of runtime security monitoring"],
        business_impact="SOC2 audit cycle extended by 6 weeks with risk of non-compliance fines.",
        product_benefits=["Real-time drift detection", "Instant automated compliance reporting"],
        value_proposition="Continuous compliance visibility without agent overhead.",
        personalisation_angle="Saw Apex is expanding its cloud infrastructure across AWS and Azure.",
        recommendation_reason="SecScan Pro directly addresses Kubernetes drift mentioned in recent hiring specs.",
        evidence_citations=[
            {"source": "https://apexcloud.test/careers", "quote": "Scaling multi-cloud Kubernetes"}
        ],
    )
    stmt_id = await state.save_prospect_problem_statement(stmt)
    assert stmt_id

    # Retrieve
    retrieved = await state.get_prospect_problem_statement("p_ps1", prod_id)
    assert retrieved is not None
    assert "cluster drift" in retrieved.business_problem
    assert len(retrieved.pain_points) == 2
    assert len(retrieved.evidence_citations) == 1


@pytest.mark.asyncio
async def test_offline_task_queue_crud(tmp_path):
    state = StateManager(str(tmp_path / "test.db"))
    await state.init_db()

    task = OfflineTask(
        task_type="business_profile",
        payload={"company_id": "c_bp1", "domain": "apexcloud.test"},
        status="pending",
        retry_count=0,
    )
    task_id = await state.queue_offline_task(task)
    assert task_id

    # Retrieve pending tasks
    pending = await state.get_pending_offline_tasks()
    assert len(pending) == 1
    assert pending[0].id == task_id
    assert pending[0].task_type == "business_profile"

    # Update status to processing / completed
    ok = await state.update_offline_task_status(task_id, status="completed")
    assert ok

    # Pending list should now be empty
    pending_now = await state.get_pending_offline_tasks()
    assert len(pending_now) == 0
