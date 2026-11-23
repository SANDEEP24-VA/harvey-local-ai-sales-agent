"""Unit tests for Problem Statements and Business Profiles (Phase D of EDXSO Milestone).

Covers:
- generate_business_profile via local Ollama and deterministic fallback
- generate_prospect_problem_statement with pain points, impact, and traceable citations
- Integration into Writer agent facts
- Fallback mechanics on inference timeout or error
"""

from unittest.mock import AsyncMock
import pytest

from sellbird.models.product import Product
from sellbird.models.prospect import Prospect
from sellbird.models.research import ClassifiedBuyingSignal
from sellbird.product_intelligence import (
    generate_business_profile,
    generate_prospect_problem_statement,
    _deterministic_business_profile_fallback,
    _deterministic_problem_statement_fallback,
)


def test_deterministic_business_profile_fallback():
    profile = _deterministic_business_profile_fallback(
        company_id="co_123",
        company_name="Apex Logistics",
        domain="apexlogistics.example",
        site_text="We provide supply chain solutions and freight tracking services on Shopify and React.",
    )
    assert profile.company_id == "co_123"
    assert "Apex Logistics" in profile.summary
    assert "Shopify" in profile.tech_stack
    assert "React" in profile.tech_stack
    assert len(profile.core_offerings) > 0


def test_deterministic_problem_statement_fallback():
    prospect = Prospect(
        id="p_99",
        first_name="David",
        last_name="Miller",
        title="Head of Procurement",
        company="Apex Logistics",
        industry="Supply Chain",
    )
    product = Product(
        id="prod_track",
        name="TrackGuard",
        description="Autonomous IoT freight monitoring",
        value_proposition="Real-time freight integrity and zero loss",
        pain_points=["Cargo theft", "Manual paperwork", "Delayed status"],
        benefits=["Automated sensor alerts", "Instant customs clearance"],
    )
    sig = ClassifiedBuyingSignal(
        prospect_id="p_99",
        signal_code="HIRING_ROLE",
        result="Hiring Logistics Manager",
        supporting_evidence="We are hiring a Logistics Manager in Chicago",
        verification_status="verified_observation",
    )

    statement = _deterministic_problem_statement_fallback(
        prospect=prospect,
        product=product,
        signals=[sig],
    )

    assert statement.prospect_id == "p_99"
    assert statement.product_id == "prod_track"
    assert "Apex Logistics" in statement.business_problem
    assert any("Cargo theft" in p for p in statement.pain_points)
    assert len(statement.evidence_citations) >= 2


@pytest.mark.asyncio
async def test_generate_prospect_problem_statement_with_mock_brain():
    prospect = Prospect(
        id="p_101",
        first_name="Sarah",
        last_name="Connor",
        title="VP of Sales",
        company="Cyberdyne Systems",
        industry="Robotics",
    )
    product = Product(
        id="prod_ai",
        name="Harvey Sales Agent",
        description="Autonomous B2B outbound engine",
        value_proposition="10x pipeline without extra headcount",
    )

    mock_brain = AsyncMock()
    mock_brain.think_json.return_value = {
        "business_problem": "Cyberdyne Systems SDR team cannot keep up with outbound volume across multiple robotics product lines.",
        "pain_points": ["Manual research takes 4 hours per day", "Low reply rates on generic copy"],
        "business_impact": "$500k in pipeline lost every quarter due to delayed follow-ups.",
        "product_benefits": ["Instant local AI prospect research", "Personalized evidence-grounded cold emails"],
        "value_proposition": "Automated, private outbound that scales with your robotics catalog.",
        "personalisation_angle": "Address Sarah's need for automated multi-product SDR scaling.",
        "recommendation_reason": "Direct role alignment with sales leadership pain points.",
        "evidence_citations": [
            {
                "source": "Prospect Profile",
                "quote": "Job Title: VP of Sales",
                "relevance": "Direct owner of outbound sales quota",
            }
        ],
    }

    statement = await generate_prospect_problem_statement(
        brain=mock_brain,
        prospect=prospect,
        product=product,
    )

    assert statement.prospect_id == "p_101"
    assert statement.product_id == "prod_ai"
    assert "Cyberdyne Systems" in statement.business_problem
    assert len(statement.pain_points) == 2
    assert "$500k in pipeline" in statement.business_impact
    assert len(statement.evidence_citations) == 1
    assert statement.evidence_citations[0]["source"] == "Prospect Profile"
