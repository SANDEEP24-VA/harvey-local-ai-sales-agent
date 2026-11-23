"""Tests for Phase 3 Evidence-Grounded Buying Signal Intelligence.

Verifies:
- All 29 buying signal definitions are supported
- Heuristic scanning across technical, growth, organizational, and discovery signals
- Strict evidence quote verification and status assignment
"""

import pytest

from sellbird.models.prospect import Prospect
from sellbird.research_engine import (
    STATUS_DETERMINISTIC_HEURISTIC,
    STATUS_MODEL_INFERENCE,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED_OBSERVATION,
)
from sellbird.signal_intelligence import (
    _CATALOG_BY_CODE,
    _heuristic_signal_scan,
    classify_buying_signals,
)
from sellbird.signals import SIGNAL_CATALOG


def test_catalog_has_29_signal_definitions():
    """Verify that all 29 signal definitions exist and have required metadata."""
    assert len(SIGNAL_CATALOG) == 29
    assert len(_CATALOG_BY_CODE) == 29

    expected_codes = {
        "SERP_RANK", "IN_LOCAL_PACK", "MAPS_RANK", "FOUND_IN_SERP",
        "NO_WEBSITE", "UNCLAIMED_LISTING", "REVIEW_RATING", "REVIEW_COUNT",
        "INCUMBENT_AGENCY", "RUNNING_GOOGLE_ADS", "RUNNING_META_ADS",
        "TECH_STACK", "NO_SCHEMA_MARKUP", "NO_ONLINE_BOOKING",
        "SITE_PAGE_COUNT", "BLOG_STALE", "BLOCKS_AI_CRAWLERS",
        "YC_BACKED", "ACCELERATOR", "FUNDING_ROUND", "HIRING_ROLE",
        "OPEN_ROLES", "CONTACT_FOUND", "DECISION_MAKER_TITLE",
        "REGISTRY_VERIFIED", "LIKELY_OWNER", "EMAIL_PATTERN",
        "EMAIL_STATUS", "CONTACT_FORM_URL",
    }
    assert set(_CATALOG_BY_CODE.keys()) == expected_codes


def test_heuristic_signal_scan_detects_signals():
    """Verify _heuristic_signal_scan accurately detects tech, hiring, funding, and owner signals."""
    prospect = Prospect(
        id="p123",
        company="NexTech Solutions",
        title="Founder and CEO",
        first_name="Elena",
        last_name="Rostova",
        email="elena.rostova@nextech.io",
        source="Google Search",
        source_url="https://nextech.io",
    )

    research_text = """
    NexTech Solutions is backed by Y Combinator W22 and recently completed a $3.5M Seed round.
    We are hiring senior full-stack software engineers to expand our core platform.
    Our web application is built with React, Next.js, and Stripe for payments.
    Site designed by StudioPixel Agency.
    Book a demo with our team at https://nextech.io/contact.
    """

    signals = _heuristic_signal_scan(
        prospect=prospect,
        research_text=research_text,
        source_url="https://nextech.io",
    )

    detected_codes = {s.signal_code for s in signals}

    assert "TECH_STACK" in detected_codes
    assert "HIRING_ROLE" in detected_codes
    assert "DECISION_MAKER_TITLE" in detected_codes
    assert "CONTACT_FOUND" in detected_codes
    assert "YC_BACKED" in detected_codes
    assert "FUNDING_ROUND" in detected_codes
    assert "INCUMBENT_AGENCY" in detected_codes
    assert "LIKELY_OWNER" in detected_codes
    assert "EMAIL_PATTERN" in detected_codes

    # Verify that all returned signals have verified or deterministic verification status
    for s in signals:
        assert s.verification_status in (STATUS_VERIFIED_OBSERVATION, STATUS_DETERMINISTIC_HEURISTIC)
        assert len(s.supporting_evidence) > 0


@pytest.mark.asyncio
async def test_classify_buying_signals_fallback():
    """Verify classify_buying_signals executes heuristic fallback when brain fails."""
    from types import SimpleNamespace
    from sellbird.brain import Brain
    from sellbird.state import StateManager

    # Brain with empty in-memory state
    state = StateManager(db_path=":memory:")
    brain = Brain(state=state)
    prospect = Prospect(
        id="p456",
        company="Apex Logistics",
        title="Director of Operations",
        first_name="David",
        last_name="Kim",
        email="david@apexlogistics.com",
    )

    signals = await classify_buying_signals(
        brain=brain,
        prospect=prospect,
        research_text="We are hiring: Dispatch Manager, Freight Broker. Connect with us at calendly.com/apex.",
    )

    assert len(signals) >= 2
    codes = {s.signal_code for s in signals}
    assert "HIRING_ROLE" in codes
    assert "CONTACT_FOUND" in codes
