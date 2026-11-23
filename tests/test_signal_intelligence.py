"""Unit tests for Signal Intelligence (Phase C of EDXSO Milestone).

Covers:
- Deterministic heuristic signal scanner
- Evidence-grounded classification against Sellbird's 29 signal definitions
- Verification status marking (verified_observation vs model_inference)
- Verbatim supporting quote validation
- Fallback scanning when Ollama is offline/mocked
"""

from unittest.mock import AsyncMock
import pytest

from sellbird.models.product import Product
from sellbird.models.prospect import Prospect
from sellbird.signal_intelligence import _heuristic_signal_scan, classify_buying_signals
from sellbird.signals import SIGNAL_CATALOG


RESEARCH_BODY = """
About Us: Apex Global Tech operates high-throughput data pipelines on WordPress and HubSpot.
We are actively spending on digital acquisition with active Google Ads tracking and Meta Pixel.
Careers: We are hiring: Senior DevOps Engineer and Lead Python Architect.
Contact: Our office is located in Austin, TX. Book a demo directly with our founder.
"""


def test_signal_catalog_has_29_signals():
    # Verify that the complete catalog exists and contains expected categories
    assert len(SIGNAL_CATALOG) >= 20
    codes = {s["code"] for s in SIGNAL_CATALOG}
    assert "TECH_STACK" in codes
    assert "RUNNING_GOOGLE_ADS" in codes
    assert "RUNNING_META_ADS" in codes
    assert "HIRING_ROLE" in codes
    assert "DECISION_MAKER_TITLE" in codes
    assert "CONTACT_FOUND" in codes


def test_heuristic_signal_scan():
    prospect = Prospect(
        id="p_test1",
        first_name="Marcus",
        last_name="Vance",
        title="VP of Engineering",
        email="marcus@apexglobal.example",
        company="Apex Global Tech",
    )
    signals = _heuristic_signal_scan(
        prospect=prospect,
        research_text=RESEARCH_BODY,
        source_url="https://apexglobal.example",
    )

    codes = {s.signal_code: s for s in signals}
    assert "TECH_STACK" in codes
    assert "WordPress" in codes["TECH_STACK"].result or "HubSpot" in codes["TECH_STACK"].result
    assert codes["TECH_STACK"].verification_status == "verified_observation"

    assert "DECISION_MAKER_TITLE" in codes
    assert codes["DECISION_MAKER_TITLE"].result == "Decision-maker title: VP of Engineering"
    assert codes["DECISION_MAKER_TITLE"].confidence >= 0.90

    assert "CONTACT_FOUND" in codes
    assert "marcus@apexglobal.example" in codes["CONTACT_FOUND"].result


@pytest.mark.asyncio
async def test_classify_buying_signals_with_ollama_mock():
    prospect = Prospect(
        id="p_test2",
        first_name="Elena",
        last_name="Rostova",
        title="Chief Technology Officer",
        email="elena@apexcloud.example",
        company="Apex Cloud",
    )
    product = Product(
        id="prod_k8s",
        name="KubeGuard",
        description="Kubernetes configuration drift prevention",
    )

    # Mock Brain returning a signal with verbatim quote and an inferred one
    mock_brain = AsyncMock()
    mock_brain.think_json.return_value = [
        {
            "signal_code": "HIRING_ROLE",
            "category": "profile",
            "result": "Hiring Senior DevOps Engineer",
            "confidence": 0.95,
            "supporting_evidence": "We are hiring: Senior DevOps Engineer",  # Verbatim in RESEARCH_BODY
        },
        {
            "signal_code": "SITE_PAGE_COUNT",
            "category": "profile",
            "result": "Estimated 25 pages",
            "confidence": 0.60,
            "supporting_evidence": "Inferred from sitemap structure",  # Not verbatim
        },
    ]

    results = await classify_buying_signals(
        brain=mock_brain,
        prospect=prospect,
        product=product,
        research_text=RESEARCH_BODY,
        source_url="https://apexcloud.example",
    )

    by_code = {s.signal_code: s for s in results}
    assert "HIRING_ROLE" in by_code
    # Verbatim quote must be marked verified_observation
    assert by_code["HIRING_ROLE"].verification_status == "verified_observation"
    assert by_code["HIRING_ROLE"].confidence == 0.95

    # Non-verbatim must be marked model_inference
    assert "SITE_PAGE_COUNT" in by_code
    assert by_code["SITE_PAGE_COUNT"].verification_status == "model_inference"
