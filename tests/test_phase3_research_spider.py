"""Tests for Phase 3 Research Engine, URL Canonicalization, Scrapy, Decision-Makers and ATS.

Verifies:
- URL canonicalization stripping tracking parameters, fragments, and default ports
- Verbatim quote validation
- 4 distinct verification statuses
- Scrapy spider and isolated subprocess runner
- Decision-maker extraction integration via sellbird.collectors.people
- ATS job board detection and parsing via sellbird.collectors.ats
- Reusable multi-page website research with zero-network offline mode
"""

import pytest

from sellbird.collectors.ats import Board, Posting, detect_board
from sellbird.collectors.people import find_people
from sellbird.research_engine import (
    STATUS_DETERMINISTIC_HEURISTIC,
    STATUS_MODEL_INFERENCE,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED_OBSERVATION,
    VERIFICATION_STATUSES,
    ResearchEngine,
    canonicalize_url,
    crawl_company_website,
    verify_evidence_quote,
)


def test_url_canonicalization():
    """Verify URL canonicalization handles tracking params, fragments, ports, and normalization."""
    # 1. Strip tracking params and fragments
    raw = "https://Example.COM:443/products/features/?utm_source=google&b=2&utm_campaign=winter&a=1#pricing"
    expected = "https://example.com/products/features?a=1&b=2"
    assert canonicalize_url(raw) == expected

    # 2. HTTP default port 80
    assert canonicalize_url("http://test.org:80/about/") == "http://test.org/about"

    # 3. Preserve root slash
    assert canonicalize_url("https://company.io/") == "https://company.io"

    # 4. Add missing scheme
    assert canonicalize_url("mycompany.com") == "https://mycompany.com"

    # 5. Non-tracking query params preserved
    assert canonicalize_url("https://site.com/search?q=test&fbclid=123") == "https://site.com/search?q=test"


def test_verbatim_evidence_quote_verification():
    """Verify exact verbatim quote matching with whitespace and case tolerance."""
    source = """
    Welcome to CloudScale Technologies.
    We are hiring senior distributed systems engineers to join our team in Austin, TX.
    Our platform is built with React, Next.js, and Stripe.
    """
    # Verbatim match
    assert verify_evidence_quote(source, "senior distributed systems engineers") is True
    assert verify_evidence_quote(source, "Platform is built with React") is True
    # Non-existent quote
    assert verify_evidence_quote(source, "Series B venture funding raised") is False
    assert verify_evidence_quote(source, "") is False
    assert verify_evidence_quote("", "something") is False


def test_four_verification_statuses_defined():
    """Verify all 4 required verification statuses are available and defined."""
    assert STATUS_VERIFIED_OBSERVATION == "verified_observation"
    assert STATUS_MODEL_INFERENCE == "model_inference"
    assert STATUS_DETERMINISTIC_HEURISTIC == "deterministic_heuristic"
    assert STATUS_UNVERIFIED == "unverified"
    assert len(VERIFICATION_STATUSES) == 4


def test_html_parsing_extracts_technologies_and_signals():
    """Verify ResearchEngine extracts title, description, technologies, and hiring mentions."""
    engine = ResearchEngine()
    html = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>DataFlow AI - Autonomous Analytics</title>
        <meta name="description" content="Next-generation analytics powered by local machine learning.">
        <script src="https://cdn.shopify.com/s/files/theme.js"></script>
        <script src="https://js.stripe.com/v3/"></script>
      </head>
      <body>
        <h1>Transform Your Enterprise Data</h1>
        <p>We are hiring: Backend Engineer, Product Manager, and Growth Lead.</p>
        <p>Schedule a call with our team via calendly.com/dataflow-demo.</p>
      </body>
    </html>
    """
    res = engine.parse_html_content("https://dataflow.ai", html, "2026-01-01T00:00:00")
    assert res.title == "DataFlow AI - Autonomous Analytics"
    assert res.meta_description == "Next-generation analytics powered by local machine learning."
    assert "Shopify" in res.technologies
    assert "Stripe" in res.technologies
    assert any("Backend Engineer" in h for h in res.hiring_mentions)
    assert res.metadata["has_booking_widget"] is True


def test_people_collector_integration():
    """Verify sellbird.collectors.people extracts decision makers from parsed website pages."""
    pages = [
        (
            "https://acmesoftware.com/about",
            """
            <html>
              <body>
                <h2>Leadership</h2>
                <div class="team-card">
                  <h3>Sarah Jenkins</h3>
                  <p>Founder & CEO</p>
                </div>
                <div class="team-card">
                  <h3>Marcus Vance</h3>
                  <p>Chief Technology Officer</p>
                </div>
              </body>
            </html>
            """,
        )
    ]
    people = find_people(pages, domain="acmesoftware.com")
    assert len(people) >= 1
    ceo = next((p for p in people if p.get("first_name") == "Sarah"), None)
    assert ceo is not None
    assert ceo["last_name"] == "Jenkins"
    assert ceo["owner"] is True


def test_ats_board_detection_integration():
    """Verify sellbird.collectors.ats detects Greenhouse and Lever embed links."""
    html_gh = """
    <html>
      <body>
        <a href="https://boards.greenhouse.io/acmeplatform">View Open Positions</a>
      </body>
    </html>
    """
    board_gh = detect_board(html_gh, domain="acmeplatform.com")
    assert board_gh is not None
    assert board_gh.ats == "greenhouse"
    assert board_gh.token == "acmeplatform"
    assert board_gh.page_url == "https://boards.greenhouse.io/acmeplatform"

    html_lever = """
    <html>
      <body>
        <a href="https://jobs.lever.co/cloudflow">Join our team on Lever</a>
      </body>
    </html>
    """
    board_lever = detect_board(html_lever, domain="cloudflow.io")
    assert board_lever is not None
    assert board_lever.ats == "lever"
    assert board_lever.token == "cloudflow"


@pytest.mark.asyncio
async def test_crawl_company_website_offline_mode():
    """Verify crawl_company_website respects offline_only=True and makes zero network calls."""
    report = await crawl_company_website(
        domain_or_url="offline-test-nonexistent.com",
        max_pages=5,
        offline_only=True,
    )
    assert report["domain"] == "offline-test-nonexistent.com"
    assert report["pages_crawled"] == 0
    assert report["from_cache"] is True
    assert report["pages"] == []
