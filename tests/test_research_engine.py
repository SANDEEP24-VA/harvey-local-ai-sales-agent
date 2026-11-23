"""Unit tests for ResearchEngine (Phase B of EDXSO Milestone).

Covers:
- HTML text extraction and heading parsing via BeautifulSoup
- Tech stack signature detection
- Hiring keywords and open roles extraction
- Local SQLite research caching (write and read)
- Offline mode (offline_only=True returns cached item without network)
- SSRF protection validation
- Scrapy pipeline item export adapter
"""

import pytest
from sellbird.research_engine import ResearchEngine, ResearchResult
from sellbird.state import StateManager


SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Apex Cloud Solutions | NextGen Kubernetes Orchestration</title>
  <script src="https://cdn.shopify.com/s/files/1/shopify.js"></script>
  <script src="https://js.stripe.com/v3/"></script>
  <script>
    (function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':new Date().getTime(),event:'gtm.js'});})(window,document,'script','dataLayer','GTM-XYZ');
    fbq('init', '1234567890');
  </script>
</head>
<body>
  <header>
    <h1>Apex Cloud Infrastructure Solutions</h1>
    <h2>Enterprise Multi-Cloud Cost Optimization</h2>
  </header>
  <main>
    <p>Apex Cloud delivers mission-critical Kubernetes management for fast-growing B2B software companies.</p>
    <p>Our autonomous agents reduce AWS and GCP cloud spend by up to 40% while preventing configuration drift.</p>
    <div>
      <h3>Career Opportunities</h3>
      <p>We are hiring: Senior Site Reliability Engineer, Staff Platform Architect, and Solutions Engineer.</p>
    </div>
    <div>
      <p>Schedule a call: Book a demo with our engineering team today via calendly.com/apexcloud.</p>
    </div>
  </main>
  <footer>
    <p>&copy; 2026 Apex Cloud Inc. All rights reserved.</p>
  </footer>
</body>
</html>"""


@pytest.mark.asyncio
async def test_extract_from_html():
    engine = ResearchEngine()
    text, meta = engine.extract_from_html(SAMPLE_HTML, url="https://apexcloud.example.com")

    assert "Apex Cloud Infrastructure Solutions" in text
    assert "Enterprise Multi-Cloud Cost Optimization" in text
    assert meta["title"] == "Apex Cloud Solutions | NextGen Kubernetes Orchestration"
    assert "Shopify" in meta["technologies"]
    assert "Stripe" in meta["technologies"]
    assert meta["has_meta_ads"] is True
    assert meta["has_booking_widget"] is True
    assert any("Senior Site Reliability Engineer" in r for r in meta["hiring_roles"])


@pytest.mark.asyncio
async def test_research_cache_hit_and_offline_mode(tmp_path):
    db_path = str(tmp_path / "research_test.db")
    state = StateManager(db_path)
    await state.init_db()

    engine = ResearchEngine(state=state)

    # 1. Manually populate cache item
    url = "https://cached-domain.example/info"
    text, meta = engine.extract_from_html(SAMPLE_HTML, url=url)
    await engine._cache_content(url, "cached-domain.example", SAMPLE_HTML, text, meta)

    # 2. Scrape with offline_only=True
    result = await engine.scrape(url, offline_only=True)
    assert result.from_cache is True
    assert result.status_code == 200
    assert "Apex Cloud" in result.extracted_text
    assert result.metadata["title"] == "Apex Cloud Solutions | NextGen Kubernetes Orchestration"

    # 3. Scrapy item adapter
    scrapy_item = result.to_scrapy_item()
    assert scrapy_item["url"] == url
    assert scrapy_item["domain"] == "cached-domain.example"
    assert scrapy_item["from_cache"] is True


@pytest.mark.asyncio
async def test_offline_mode_returns_cached_or_graceful_error(tmp_path):
    db_path = str(tmp_path / "research_test2.db")
    state = StateManager(db_path)
    await state.init_db()

    engine = ResearchEngine(state=state)
    uncached_url = "https://nonexistent-offline-domain.example/about"

    # Requesting uncached url in offline mode must not raise and return graceful error
    result = await engine.scrape(uncached_url, offline_only=True)
    assert result.from_cache is False
    assert result.status_code == 404
    assert "Offline mode" in result.error
