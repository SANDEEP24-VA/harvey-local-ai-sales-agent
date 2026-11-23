"""Unit tests for local AI product intelligence, ICP generation, and prospect matching."""

import pytest
from unittest.mock import AsyncMock

from sellbird.brain import Brain
from sellbird.models.product import Product, ProductICP
from sellbird.models.prospect import Prospect
from sellbird.product_intelligence import (
    generate_product_icp,
    generate_product_intelligence,
    match_prospect_product,
    _deterministic_heuristic_match,
)
from sellbird.state import StateManager


@pytest.fixture
def product():
    return Product(
        id="prod_test_1",
        name="AutoCloser AI",
        category="Sales Automation",
        description="Local AI sales copilot for engineering-led teams",
        pricing_model="tiered",
        pricing_details="$300-$1000/month",
        features=["Automated prospecting", "Local Ollama inference", "Reply classification"],
        benefits=["Triple meeting book rates", "100% private data with zero cloud LLMs"],
        target_industries=["Software", "Cybersecurity", "Fintech"],
        target_customer_sizes=["10-250 employees"],
        target_job_titles=["Founder", "VP Sales", "Head of Business Development"],
        pain_points=["Sales reps spend 70% time on admin", "Data privacy concerns with OpenAI/Claude"],
        value_proposition="Close more deals completely on-prem with local AI.",
        competitors=["Apollo", "Salesloft"],
        objection_responses={"privacy risk": "Everything runs locally on your machine via Ollama."},
        entry_offer="14-day local trial",
        is_active=True,
    )


@pytest.fixture
def prospect():
    return Prospect(
        id="pros_test_1",
        first_name="Marcus",
        last_name="Vance",
        email="marcus@securecore.tech",
        title="Head of Business Development",
        seniority="director",
        company="SecureCore",
        industry="Cybersecurity",
        company_size="50-100 employees",
        personalization_notes="Specializes in zero-trust network architectures.",
        source="company_discovery",
        source_url="https://securecore.tech/team",
    )


@pytest.mark.asyncio
async def test_generate_product_icp_success(tmp_path, product):
    """Test ICP generation with valid mocked Ollama response."""
    state = StateManager(str(tmp_path / "test.db"))
    brain = Brain(state)

    mock_icp_data = {
        "target_industries": ["Cybersecurity", "Fintech", "Healthtech"],
        "company_size": "20-200 employees",
        "revenue_range": "$3M-$25M",
        "job_titles": ["VP Sales", "Head of BD", "Founder"],
        "technologies": ["CRM", "Linux", "Ollama"],
        "business_characteristics": ["Data sensitivity", "Outbound focused"],
        "pain_points": ["Cloud LLM compliance blockers", "Lead research fatigue"],
        "buying_signals": ["Hiring SDRs", "Recent SOC2 audit"],
        "exclusion_criteria": ["B2C companies", "Solo operators"],
    }

    brain.think_json = AsyncMock(return_value=mock_icp_data)

    icp = await generate_product_icp(brain, product)
    assert icp is not None
    assert icp.product_id == product.id
    assert "Cybersecurity" in icp.target_industries
    assert "VP Sales" in icp.job_titles
    assert "Cloud LLM compliance blockers" in icp.pain_points
    assert "B2C companies" in icp.exclusion_criteria


@pytest.mark.asyncio
async def test_generate_product_icp_failure_safe(tmp_path, product):
    """Test safe handling when Ollama returns None or fails."""
    state = StateManager(str(tmp_path / "test.db"))
    brain = Brain(state)
    brain.think_json = AsyncMock(side_effect=RuntimeError("Ollama daemon down"))

    icp = await generate_product_icp(brain, product)
    assert icp is None


@pytest.mark.asyncio
async def test_generate_product_intelligence_success(tmp_path, product):
    """Test generating product marketing intelligence."""
    state = StateManager(str(tmp_path / "test.db"))
    brain = Brain(state)

    mock_intel = {
        "problem_statements": [
            "Outbound sales data sent to cloud LLMs leaks confidential customer IP.",
            "Manual lead personalization takes SDRs 4 hours per day.",
        ],
        "value_propositions": [
            "100% air-gapped local AI sales agent that protects customer confidentiality.",
        ],
        "differentiators": [
            "Runs natively on Ollama without any cloud API tokens.",
        ],
        "objection_responses": {
            "our data is too sensitive": "AutoCloser runs entirely on your local workstation; zero bits leave.",
        },
    }

    brain.think_json = AsyncMock(return_value=mock_intel)

    intel = await generate_product_intelligence(brain, product)
    assert intel is not None
    assert len(intel["problem_statements"]) == 2
    assert "our data is too sensitive" in intel["objection_responses"]
    assert "Runs natively on Ollama" in intel["differentiators"][0]


@pytest.mark.asyncio
async def test_match_prospect_product_ai_success(tmp_path, product, prospect):
    """Test AI prospect-product matching with structured analysis and evidence."""
    state = StateManager(str(tmp_path / "test.db"))
    brain = Brain(state)

    mock_match_data = {
        "fit_score": 88,
        "status": "qualified",
        "matching_reasons": [
            "Exact decision-maker persona (Head of BD)",
            "Industry match (Cybersecurity with data sensitivity needs)",
        ],
        "analysis": {
            "verified_facts": [
                "Role: Head of Business Development",
                "Industry: Cybersecurity",
            ],
            "inferred_points": [
                "Company likely handles confidential client infrastructure data",
            ],
            "missing_information": [
                "Current outbound software stack not confirmed",
            ],
            "key_pain_point": "Data confidentiality during sales prospecting",
            "recommended_angle": "Emphasize zero-cloud Ollama privacy",
        },
        "evidence": [
            {
                "source": "Role & Industry Analysis",
                "url": "https://securecore.tech/team",
                "description": "Leader in cybersecurity outbound growth",
                "buying_signal": "high_authority",
            }
        ],
    }

    brain.think_json = AsyncMock(return_value=mock_match_data)

    match, evidence = await match_prospect_product(brain, prospect, product)
    assert match is not None
    assert match.prospect_id == prospect.id
    assert match.product_id == product.id
    assert match.fit_score == 88.0
    assert match.matching_status == "qualified"
    assert len(match.matching_reasons) == 2
    assert match.ai_analysis["key_pain_point"] == "Data confidentiality during sales prospecting"

    assert len(evidence) == 1
    assert evidence[0].evidence_source == "Role & Industry Analysis"
    assert evidence[0].buying_signal == "high_authority"


@pytest.mark.asyncio
async def test_match_prospect_product_fallback_on_error(tmp_path, product, prospect):
    """Test that when Ollama fails, a safe heuristic match is produced without crashing."""
    state = StateManager(str(tmp_path / "test.db"))
    brain = Brain(state)
    brain.think_json = AsyncMock(side_effect=Exception("Ollama timeout"))

    match, evidence = await match_prospect_product(brain, prospect, product)
    assert match is not None
    assert match.prospect_id == prospect.id
    assert match.product_id == product.id
    # Heuristic should match title ("Head of Business Development") and industry ("Cybersecurity")
    assert match.fit_score >= 60.0
    assert match.ai_analysis.get("source") == "heuristic_fallback"
    assert len(evidence) >= 1
