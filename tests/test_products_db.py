"""Tests for multi-product database architecture, migration v17, and CRUD operations."""

import pytest
import aiosqlite
from sellbird.models.campaign import Campaign
from sellbird.models.conversation import Conversation
from sellbird.models.product import (
    Product,
    ProductCreate,
    ProductICP,
    ProductResearchEvidence,
    ProspectProductMatch,
)
from sellbird.models.prospect import Prospect
from sellbird.state import MIGRATIONS, StateManager


@pytest.mark.asyncio
async def test_migration_v17_fresh_database(tmp_path):
    """Migration v17 runs cleanly on a fresh database."""
    db_file = tmp_path / "fresh.db"
    state = StateManager(str(db_file))
    await state.init_db()

    async with aiosqlite.connect(str(db_file)) as db:
        async with db.execute("PRAGMA user_version") as cursor:
            (version,) = await cursor.fetchone()
        assert version >= 17

        # Verify tables exist
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
            tables = {row[0] for row in await cursor.fetchall()}

        assert "products" in tables
        assert "product_icps" in tables
        assert "prospect_product_matches" in tables
        assert "product_research_evidence" in tables

        # Verify columns on existing tables
        async with db.execute("PRAGMA table_info(campaigns)") as cursor:
            campaign_cols = {row[1] for row in await cursor.fetchall()}
        assert "product_id" in campaign_cols

        async with db.execute("PRAGMA table_info(outbox)") as cursor:
            outbox_cols = {row[1] for row in await cursor.fetchall()}
        assert "product_id" in outbox_cols

        async with db.execute("PRAGMA table_info(conversations)") as cursor:
            convo_cols = {row[1] for row in await cursor.fetchall()}
        assert "product_id" in convo_cols


@pytest.mark.asyncio
async def test_migration_v17_from_v16(tmp_path):
    """Migration v17 successfully upgrades a v16 database while preserving data."""
    db_file = tmp_path / "v16.db"

    # Seed v1..v16
    async with aiosqlite.connect(str(db_file)) as db:
        for script in MIGRATIONS[:16]:
            await db.executescript(script)
        await db.execute("PRAGMA user_version = 16")

        # Insert historical prospect, campaign, and conversation
        await db.execute(
            "INSERT INTO prospects (id, first_name, last_name, email) VALUES ('p1', 'Alice', 'Smith', 'alice@test.com')"
        )
        await db.execute(
            "INSERT INTO campaigns (id, name, status) VALUES ('c1', 'Legacy Campaign', 'active')"
        )
        await db.execute(
            "INSERT INTO conversations (id, prospect_id, status) VALUES ('conv1', 'p1', 'open')"
        )
        await db.commit()

    # Now run init_db via StateManager which should migrate to v17
    state = StateManager(str(db_file))
    await state.init_db()

    async with aiosqlite.connect(str(db_file)) as db:
        async with db.execute("PRAGMA user_version") as cursor:
            (version,) = await cursor.fetchone()
        assert version >= 17

        # Verify historical records remain intact and product_id is NULL
        async with db.execute("SELECT id, name, product_id FROM campaigns WHERE id = 'c1'") as cursor:
            c_row = await cursor.fetchone()
            assert c_row[0] == "c1"
            assert c_row[1] == "Legacy Campaign"
            assert c_row[2] is None

        async with db.execute("SELECT id, prospect_id, product_id FROM conversations WHERE id = 'conv1'") as cursor:
            conv_row = await cursor.fetchone()
            assert conv_row[0] == "conv1"
            assert conv_row[1] == "p1"
            assert conv_row[2] is None


@pytest.mark.asyncio
async def test_product_crud_and_archival(tmp_path):
    """Test creating, reading, listing, updating, and archiving products."""
    db_file = tmp_path / "products.db"
    state = StateManager(str(db_file))
    await state.init_db()

    # Create Product 1
    p1 = ProductCreate(
        name="SalesBot Enterprise",
        sku="SB-ENT-001",
        category="AI Software",
        description="Autonomous sales prospecting agent",
        pricing_model="subscription",
        pricing_details="$500/month per seat",
        features=["Auto discovery", "Email writing", "Objection handling"],
        benefits=["3x lead volume", "Save 15 hrs/week"],
        target_industries=["B2B SaaS", "Tech"],
        target_customer_sizes=["10-500 employees"],
        target_job_titles=["VP Sales", "Head of Sales"],
        pain_points=["Manual prospecting takes too long"],
        value_proposition="Scale outreach with local AI",
        competitors=["Apollo", "Outreach"],
        objection_responses={"too expensive": "ROI pays back in 2 weeks"},
        entry_offer="14-day free pilot",
        is_active=True,
    )
    p1_id = await state.add_product(p1)
    assert p1_id

    # Create Product 2
    p2 = ProductCreate(
        name="LeadScraper Pro",
        sku="LS-PRO-002",
        category="Data Scraping",
        description="Local web scraping intelligence tool",
        is_active=True,
    )
    p2_id = await state.add_product(p2)
    assert p2_id

    # Get Product
    retrieved = await state.get_product(p1_id)
    assert retrieved is not None
    assert retrieved.name == "SalesBot Enterprise"
    assert retrieved.sku == "SB-ENT-001"
    assert "Auto discovery" in retrieved.features
    assert "B2B SaaS" in retrieved.target_industries
    assert retrieved.objection_responses.get("too expensive") == "ROI pays back in 2 weeks"
    assert retrieved.is_active is True

    # List active products
    active = await state.list_products(include_archived=False)
    assert len(active) == 2

    # Update product
    updated = await state.update_product(
        p1_id,
        description="Updated autonomous sales prospecting agent",
        features=["Auto discovery", "Email writing", "Objection handling", "Ollama native"],
    )
    assert updated is True

    retrieved2 = await state.get_product(p1_id)
    assert retrieved2.description == "Updated autonomous sales prospecting agent"
    assert len(retrieved2.features) == 4
    assert "Ollama native" in retrieved2.features

    # Archive Product 2 (soft-delete)
    archived = await state.archive_product(p2_id)
    assert archived is True

    # Check listing without archived
    active_now = await state.list_products(include_archived=False)
    assert len(active_now) == 1
    assert active_now[0].id == p1_id

    # Check listing with archived
    all_prods = await state.list_products(include_archived=True)
    assert len(all_prods) == 2


@pytest.mark.asyncio
async def test_product_icp_operations(tmp_path):
    """Test product ICP storage and retrieval."""
    db_file = tmp_path / "icp.db"
    state = StateManager(str(db_file))
    await state.init_db()

    prod_id = await state.add_product(ProductCreate(name="CloudBackup"))

    icp = ProductICP(
        product_id=prod_id,
        target_industries=["Healthcare", "Finance"],
        company_size="50-1000",
        revenue_range="$5M-$50M",
        job_titles=["CTO", "CISO", "IT Director"],
        technologies=["AWS", "Azure", "Kubernetes"],
        business_characteristics=["Strict compliance requirements", "Multi-region"],
        pain_points=["Ransomware risk", "Slow disaster recovery"],
        buying_signals=["Recent cyber security incident", "Hiring DevOps"],
        exclusion_criteria=["Consumer goods", "Under 10 employees"],
    )

    icp_id = await state.add_or_update_product_icp(icp)
    assert icp_id

    fetched = await state.get_product_icp(prod_id)
    assert fetched is not None
    assert fetched.product_id == prod_id
    assert "Healthcare" in fetched.target_industries
    assert "CTO" in fetched.job_titles
    assert "Ransomware risk" in fetched.pain_points
    assert "Under 10 employees" in fetched.exclusion_criteria


@pytest.mark.asyncio
async def test_prospect_product_matches_many_to_many(tmp_path):
    """Test that a prospect can match multiple products with distinct fit scores and analyses."""
    db_file = tmp_path / "matches.db"
    state = StateManager(str(db_file))
    await state.init_db()

    # Create 2 products
    prod_a = await state.add_product(ProductCreate(name="Product Alpha"))
    prod_b = await state.add_product(ProductCreate(name="Product Beta"))

    # Add a prospect
    prospect = Prospect(
        id="pros_1",
        first_name="Jane",
        last_name="Doe",
        email="jane@example.com",
        company="Acme Corp",
        title="VP Operations",
        industry="Logistics",
    )
    await state.add_prospect(prospect)

    # Match prospect with Product A (high fit)
    match_a = ProspectProductMatch(
        prospect_id=prospect.id,
        product_id=prod_a,
        fit_score=92.5,
        matching_reasons=["Industry match (Logistics)", "Executive buyer persona"],
        matching_status="qualified",
        ai_analysis={"confidence": 0.95, "angle": "Efficiency automation"},
    )
    await state.add_prospect_product_match(match_a)

    # Match prospect with Product B (moderate fit)
    match_b = ProspectProductMatch(
        prospect_id=prospect.id,
        product_id=prod_b,
        fit_score=68.0,
        matching_reasons=["Secondary need"],
        matching_status="matched",
        ai_analysis={"confidence": 0.70, "angle": "Cost reduction"},
    )
    await state.add_prospect_product_match(match_b)

    # Query matches for prospect
    matches = await state.get_prospect_product_matches(prospect_id=prospect.id)
    assert len(matches) == 2
    # Ordered by fit_score DESC
    assert matches[0].product_id == prod_a
    assert matches[0].fit_score == 92.5
    assert matches[1].product_id == prod_b
    assert matches[1].fit_score == 68.0

    # Query with min_score filter
    high_fits = await state.get_prospect_product_matches(prospect_id=prospect.id, min_score=80.0)
    assert len(high_fits) == 1
    assert high_fits[0].product_id == prod_a

    # Query single match
    single = await state.get_prospect_product_match(prospect.id, prod_a)
    assert single is not None
    assert single.fit_score == 92.5
    assert "Executive buyer persona" in single.matching_reasons


@pytest.mark.asyncio
async def test_product_research_evidence(tmp_path):
    """Test storing traceable research evidence for prospect-product match."""
    db_file = tmp_path / "evidence.db"
    state = StateManager(str(db_file))
    await state.init_db()

    prod_id = await state.add_product(ProductCreate(name="SecurityScanner"))
    prospect = Prospect(id="p_ev", first_name="Bob", email="bob@cloudsec.test")
    await state.add_prospect(prospect)

    ev1 = ProductResearchEvidence(
        prospect_id=prospect.id,
        product_id=prod_id,
        evidence_source="Company Careers Page",
        evidence_url="https://cloudsec.test/careers/devops",
        evidence_description="Job listing specifies hiring 3 Kubernetes security engineers",
        buying_signal="hiring_security",
    )
    await state.add_product_research_evidence(ev1)

    ev2 = ProductResearchEvidence(
        prospect_id=prospect.id,
        product_id=prod_id,
        evidence_source="Website Tech Stack",
        evidence_url="https://cloudsec.test",
        evidence_description="Detected unpatched WordPress installation",
        buying_signal="outdated_software",
    )
    await state.add_product_research_evidence(ev2)

    evidence_list = await state.get_product_research_evidence(prod_id, prospect.id)
    assert len(evidence_list) == 2
    sources = {e.evidence_source for e in evidence_list}
    assert "Company Careers Page" in sources
    assert "Website Tech Stack" in sources


@pytest.mark.asyncio
async def test_product_associated_campaigns_and_outbox(tmp_path):
    """Test linking campaigns and outbox items to specific products."""
    db_file = tmp_path / "workflows.db"
    state = StateManager(str(db_file))
    await state.init_db()

    prod_id = await state.add_product(ProductCreate(name="Product Alpha"))

    # Create campaign linked to product
    campaign = Campaign(
        id="c_alpha",
        name="Alpha Outbound Q1",
        status="draft",
        product_id=prod_id,
    )
    cid = await state.add_campaign(campaign)
    assert cid == "c_alpha"

    campaigns = await state.get_campaigns_by_status("draft")
    alpha_campaign = next(c for c in campaigns if c.id == "c_alpha")
    assert alpha_campaign.product_id == prod_id

    # Queue outbox item linked to product
    item_id = await state.add_outbox_item(
        prospect_id="p1",
        to_email="test@domain.test",
        subject="Hello",
        body="World",
        send_at="2026-01-01T00:00:00",
        campaign_id=cid,
        product_id=prod_id,
    )
    assert item_id

    item = await state.get_outbox_item(item_id)
    assert item is not None
    assert item["product_id"] == prod_id
