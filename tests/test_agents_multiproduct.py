"""Tests for Multi-Product Sales Agent Integration (EDXSO Phase 2).

Verifies:
1. Scout: matches newly researched prospects against active products in the catalog
   and persists matches and research evidence.
2. Writer: resolves product context for prospect batches, drafts emails using
   product features/value props, and persists product_id to campaigns and outbox items.
3. Handler: uses product-specific objection responses and sales context when drafting
   replies, while preserving unsubscribe safeguards.
4. Analyst: aggregates multi-product performance and conversion analytics.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from sellbird.agents.analyst import Analyst
from sellbird.agents.handler import Handler
from sellbird.agents.scout import Scout
from sellbird.agents.writer import Writer
from sellbird.brain import Brain
from sellbird.config import BrainConfig
from sellbird.models.campaign import Campaign
from sellbird.models.conversation import Conversation, Message
from sellbird.models.product import ProductCreate, ProductICP, ProspectProductMatch
from sellbird.models.prospect import Prospect
from sellbird.state import StateManager


@pytest.fixture
def test_config():
    return SimpleNamespace(
        brain=BrainConfig(provider="ollama"),
        product=SimpleNamespace(
            name="Default Product",
            description="Default general solution",
            pitch="Grow faster",
            objection_responses={"expensive": "Default price objection"},
            offer=SimpleNamespace(entry="Free audit"),
        ),
        persona=SimpleNamespace(
            name="Alex Agent",
            role="Sales Specialist",
            company="Sellbird Technologies",
            tone="professional and helpful",
        ),
        icp=SimpleNamespace(
            industries=["software"],
            titles=["Director", "Lead", "Manager"],
            company_sizes=["10-50"],
            locations=["US"],
            hiring_signals=[],
        ),
        channels=SimpleNamespace(
            email=SimpleNamespace(
                provider="mock",
                require_approval=True,
                from_email="alex@sellbird.example",
                domain="sellbird.example",
            )
        ),
        compliance=SimpleNamespace(do_not_contact=[]),
    )


@pytest.mark.asyncio
async def test_scout_matches_prospects_against_active_products(tmp_path, test_config):
    state_mgr = StateManager(db_path=str(tmp_path / "test_scout.db"))
    await state_mgr.init_db()

    # 1. Add two products to state
    prod_cyber_id = await state_mgr.add_product(
        ProductCreate(
            name="CloudArmor",
            description="Cloud container and Kubernetes security",
            category="Cybersecurity",
            value_proposition="Stop runtime threats in real-time",
            features=["Runtime detection", "Vulnerability management"],
            pricing_details="$1200/mo",
        )
    )
    prod_hr_id = await state_mgr.add_product(
        ProductCreate(
            name="TalentFlow",
            description="Recruiting pipeline ATS",
            category="HR Tech",
            value_proposition="Hire 2x faster with pipeline automation",
            features=["Candidate scoring", "Automated scheduling"],
            pricing_details="$500/mo",
        )
    )

    # 2. Add ICPs
    await state_mgr.add_or_update_product_icp(
        ProductICP(
            product_id=prod_cyber_id,
            target_industries=["Cybersecurity", "Fintech", "Software"],
            company_size="50-200",
            job_titles=["Chief Information Security Officer", "CISO", "Security Director"],
            pain_points=["Container vulnerabilities", "Cloud breaches"],
            buying_signals=["Migrating to Kubernetes"],
            exclusion_criteria=[],
        )
    )
    await state_mgr.add_or_update_product_icp(
        ProductICP(
            product_id=prod_hr_id,
            target_industries=["Staffing", "Recruiting", "Technology"],
            company_size="20-100",
            job_titles=["VP of Talent", "Recruiting Manager", "Head of People"],
            pain_points=["Manual applicant tracking"],
            buying_signals=["Rapid headcount growth"],
            exclusion_criteria=[],
        )
    )

    # 3. Create prospects
    ciso_prospect = Prospect(
        id="p_ciso",
        company_id="c_fintech",
        first_name="Marcus",
        last_name="Vance",
        email="marcus@fintech.test",
        title="Chief Information Security Officer",
        industry="Fintech",
        company="Fintech Secure",
        status="new",
    )
    hr_prospect = Prospect(
        id="p_hr",
        company_id="c_staff",
        first_name="Sarah",
        last_name="Connor",
        email="sarah@staff.test",
        title="VP of Talent",
        industry="Recruiting",
        company="StaffCorp",
        status="new",
    )

    # Save prospects in state
    await state_mgr.add_prospect(ciso_prospect)
    await state_mgr.add_prospect(hr_prospect)

    brain = Brain(state_mgr, test_config.brain)

    async def mock_match_think(prompt, **kwargs):
        if "CloudArmor" in prompt and "Chief Information Security Officer" in prompt:
            return {
                "fit_score": 92.0,
                "matching_status": "qualified",
                "matching_reasons": ["Chief Information Security Officer title matches CloudArmor"],
                "buying_signals": ["Migrating to Kubernetes"],
                "pain_points_addressed": ["Container vulnerabilities"],
                "evidence": [
                    {
                        "source": "Title Match",
                        "description": "CISO aligns directly with cloud security",
                        "signal": "Migrating to Kubernetes",
                    }
                ],
            }
        return {
            "fit_score": 25.0,
            "matching_status": "disqualified",
            "matching_reasons": ["Low relevance for target profile"],
            "buying_signals": [],
            "pain_points_addressed": [],
            "evidence": [],
        }

    brain.think_json = AsyncMock(side_effect=mock_match_think)
    scout = Scout(brain, state_mgr, test_config, SimpleNamespace())

    # 4. Scout evaluates contacts against products
    await scout.match_prospects_against_products([ciso_prospect, hr_prospect])

    # 5. Verify matches
    ciso_matches = await state_mgr.get_prospect_product_matches(prospect_id=ciso_prospect.id)
    assert len(ciso_matches) == 2
    cyber_match = next(m for m in ciso_matches if m.product_id == prod_cyber_id)
    hr_match = next(m for m in ciso_matches if m.product_id == prod_hr_id)

    # CISO should fit CloudArmor significantly better than TalentFlow
    assert cyber_match.fit_score > hr_match.fit_score
    assert cyber_match.matching_status in ("matched", "qualified")

    # Verify research evidence is stored
    evidence = await state_mgr.get_product_research_evidence(product_id=prod_cyber_id, prospect_id=ciso_prospect.id)
    assert len(evidence) >= 1
    assert any("title" in e.evidence_source.lower() for e in evidence)


@pytest.mark.asyncio
async def test_writer_product_resolution_and_outbox_staging(tmp_path, test_config):
    state_mgr = StateManager(db_path=str(tmp_path / "test_writer.db"))
    await state_mgr.init_db()

    # 1. Add product
    prod_id = await state_mgr.add_product(
        ProductCreate(
            name="DataStream Pro",
            description="Real-time ETL data pipeline tool",
            category="Data Infrastructure",
            value_proposition="Zero-maintenance CDC from Postgres to Snowflake",
            features=["Sub-second latency", "Schema evolution"],
            pricing_details="$800/mo",
        )
    )
    product = await state_mgr.get_product(prod_id)

    # 2. Add prospect and high-fit match
    p = Prospect(
        id="p_data",
        company_id="c_analytics",
        first_name="David",
        last_name="Kim",
        email="david@analytics.test",
        title="Lead Data Engineer",
        company="Analytics Co",
        status="new",
    )
    await state_mgr.add_prospect(p)
    await state_mgr.add_prospect_product_match(
        ProspectProductMatch(
            prospect_id=p.id,
            product_id=prod_id,
            fit_score=92.0,
            matching_status="qualified",
            matching_reasons=["Lead Data Engineer is ideal buyer for DataStream Pro"],
            ai_analysis={},
        )
    )

    brain = Brain(state_mgr, test_config.brain)
    writer = Writer(brain, state_mgr, test_config, SimpleNamespace())

    # 3. Test _resolve_product_for_batch
    resolved = await writer._resolve_product_for_batch([p])
    assert resolved is not None
    assert resolved.id == prod_id
    assert resolved.name == "DataStream Pro"

    # 4. Test _personalize_first_emails stages outbox with product_id
    with patch.object(writer, "_write_personal_email", new=AsyncMock(return_value={
        "subject": "CDC for Analytics Co data stack",
        "body": "David, saw your team is scaling Postgres."
    })):
        await writer._personalize_first_emails("camp_stream_1", [p], product=product)

    # 5. Check outbox
    async with state_mgr._connect() as db:
        async with db.execute("SELECT campaign_id, product_id, subject, body, status FROM outbox WHERE prospect_id = ?", (p.id,)) as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "camp_stream_1"
            assert row[1] == prod_id
            assert "CDC for Analytics Co" in row[2]
            assert row[4] == "pending_review"


@pytest.mark.asyncio
async def test_handler_product_objection_and_context(tmp_path, test_config):
    state_mgr = StateManager(db_path=str(tmp_path / "test_handler.db"))
    await state_mgr.init_db()

    # 1. Add product with specific objection response
    prod_id = await state_mgr.add_product(
        ProductCreate(
            name="SecureVault",
            description="Hardware security module in the cloud",
            category="Security",
            value_proposition="FIPS 140-3 Level 4 key management",
            objection_responses={
                "expensive": "SecureVault includes unlimited key rotations and dedicated HSM instances, cutting HSM operating costs by 65%.",
                "migration": "We offer white-glove migration assistance and zero-downtime key import.",
            },
        )
    )

    # 2. Add campaign and conversation associated with this product
    await state_mgr.add_campaign(
        Campaign(
            id="camp_sec",
            name="SecureVault Outreach",
            status="active",
            sequence=[],
            product_id=prod_id,
        )
    )

    p = Prospect(
        id="p_hsm",
        company_id="c_bank",
        first_name="Elena",
        last_name="Rostova",
        email="elena@bank.test",
        title="Security Architect",
        company="Global Bank",
        status="contacted",
    )
    await state_mgr.add_prospect(p)

    convo = Conversation(
        id="conv_sec_1",
        prospect_id=p.id,
        campaign_id="camp_sec",
        channel="email",
        status="open",
        stage="objection",
        product_id=prod_id,
        thread=[Message(sender="assistant", content="Would you like to explore SecureVault?")],
    )
    await state_mgr.add_conversation(convo)

    brain = Brain(state_mgr, test_config.brain)
    handler = Handler(brain, state_mgr, test_config, SimpleNamespace(instantly_api_key=""))

    # 3. Simulate prospect objection: "This sounds too expensive for our branch"
    reply_text = "This sounds interesting, but dedicated HSMs are typically too expensive for us."

    captured_prompts = []

    async def fake_think(prompt, **kwargs):
        captured_prompts.append(prompt)
        return "Elena, I completely understand budget concerns. SecureVault includes unlimited key rotations, cutting HSM operating costs by 65%."

    with patch.object(brain, "think", side_effect=fake_think):
        draft = await handler._generate_response(intent="objection", reply_text=reply_text, prospect=p, convo=convo)

    assert draft is not None
    assert len(captured_prompts) == 1
    prompt_used = captured_prompts[0]
    assert "SecureVault" in prompt_used
    assert "cutting HSM operating costs by 65%" in prompt_used

    # 4. Verify unsubscribe safeguard: if prospect unsubscribes, Handler opts them out and never pitches
    await handler._process_reply(
        lead_email=p.email,
        reply_text="Please stop emailing me and unsubscribe immediately.",
        reply_uuid="reply_unsub_1",
        campaign_id="camp_sec",
    )
    updated_prospect = await state_mgr.get_prospect(p.id)
    assert updated_prospect.status == "opted_out"
    assert await state_mgr.is_suppressed(p.email) is not None
    # Ensure no outgoing email was staged
    async with state_mgr._connect() as db:
        async with db.execute("SELECT COUNT(*) FROM outbox WHERE to_email = ?", (p.email,)) as cursor:
            count = (await cursor.fetchone())[0]
            assert count == 0


@pytest.mark.asyncio
async def test_analyst_product_performance_metrics(tmp_path, test_config):
    state_mgr = StateManager(db_path=str(tmp_path / "test_analyst.db"))
    await state_mgr.init_db()

    # 1. Add two products
    p1_id = await state_mgr.add_product(ProductCreate(name="Product Alpha", description="Alpha Solution"))
    p2_id = await state_mgr.add_product(ProductCreate(name="Product Beta", description="Beta Solution"))

    # 2. Add prospect matches
    await state_mgr.add_prospect_product_match(
        ProspectProductMatch(prospect_id="pr1", product_id=p1_id, fit_score=80.0, matching_status="qualified", matching_reasons=[], ai_analysis={})
    )
    await state_mgr.add_prospect_product_match(
        ProspectProductMatch(prospect_id="pr2", product_id=p1_id, fit_score=70.0, matching_status="matched", matching_reasons=[], ai_analysis={})
    )
    await state_mgr.add_prospect_product_match(
        ProspectProductMatch(prospect_id="pr3", product_id=p2_id, fit_score=85.0, matching_status="qualified", matching_reasons=[], ai_analysis={})
    )

    # 3. Add outbox sent records
    async with state_mgr._connect() as db:
        await db.execute(
            "INSERT INTO outbox (id, prospect_id, campaign_id, step, to_email, subject, body, send_at, status, product_id) "
            "VALUES ('ob1', 'pr1', 'c1', 1, 'pr1@test', 'Sub', 'Body', '2026-11-22', 'sent', ?)",
            (p1_id,)
        )
        await db.execute(
            "INSERT INTO outbox (id, prospect_id, campaign_id, step, to_email, subject, body, send_at, status, product_id) "
            "VALUES ('ob2', 'pr2', 'c1', 1, 'pr2@test', 'Sub', 'Body', '2026-11-22', 'sent', ?)",
            (p1_id,)
        )
        await db.execute(
            "INSERT INTO outbox (id, prospect_id, campaign_id, step, to_email, subject, body, send_at, status, product_id) "
            "VALUES ('ob3', 'pr3', 'c2', 1, 'pr3@test', 'Sub', 'Body', '2026-11-22', 'sent', ?)",
            (p2_id,)
        )

        # 4. Add conversations
        # Product 1: 1 interested, 1 meeting_booked -> 2 positive
        await db.execute(
            "INSERT INTO conversations (id, prospect_id, channel, status, intent, product_id) "
            "VALUES ('cv1', 'pr1', 'email', 'open', 'interested', ?)",
            (p1_id,)
        )
        await db.execute(
            "INSERT INTO conversations (id, prospect_id, channel, status, intent, product_id) "
            "VALUES ('cv2', 'pr2', 'email', 'meeting_booked', 'interested', ?)",
            (p1_id,)
        )
        # Product 2: 1 objection -> 0 positive
        await db.execute(
            "INSERT INTO conversations (id, prospect_id, channel, status, intent, product_id) "
            "VALUES ('cv3', 'pr3', 'email', 'open', 'objection', ?)",
            (p2_id,)
        )
        await db.commit()

    analyst = Analyst(state_mgr)

    perf = await analyst._product_performance()
    assert len(perf) == 2

    alpha = next(p for p in perf if p["product_id"] == p1_id)
    beta = next(p for p in perf if p["product_id"] == p2_id)

    assert alpha["product_name"] == "Product Alpha"
    assert alpha["matched_prospects"] == 2
    assert alpha["emails_sent"] == 2
    assert alpha["conversations"] == 2
    assert alpha["positive_replies"] == 2

    assert beta["product_name"] == "Product Beta"
    assert beta["matched_prospects"] == 1
    assert beta["emails_sent"] == 1
    assert beta["conversations"] == 1
    assert beta["positive_replies"] == 0
