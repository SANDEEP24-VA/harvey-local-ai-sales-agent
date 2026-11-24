"""Tests for Phase 3 Free Multi-Source Prospect Discovery.

Verifies:
- Only permitted free discovery providers (osm, ddg) are active
- Prohibited paid providers (DataForSEO, Serper, Scavio, TinyFish) are strictly rejected
- Scout search restricts to free backends
- OpenStreetMap and DuckDuckGo discovery runners execute cleanly
"""

import asyncio
from types import SimpleNamespace
import pytest

from sellbird.agents.scout import Scout
from sellbird.brain import Brain
from sellbird.collectors.discover import (
    PERMITTED_FREE_PROVIDERS,
    PROHIBITED_PAID_PROVIDERS,
    PROVIDERS,
    DuckDuckGoDiscovery,
    OpenStreetMap,
    estimate_cost,
    run_discovery,
)
from sellbird.config import load_config
from sellbird.state import StateManager


def test_only_free_providers_registered():
    """Verify that only free zero-cost discovery providers are available in PROVIDERS."""
    assert set(PROVIDERS.keys()) == {"osm", "ddg"}
    assert isinstance(PROVIDERS["osm"], OpenStreetMap)
    assert isinstance(PROVIDERS["ddg"], DuckDuckGoDiscovery)
    assert "dataforseo_listings" not in PROVIDERS
    assert "dataforseo_serp" not in PROVIDERS
    assert "serper" not in PROVIDERS
    assert "scavio_maps" not in PROVIDERS
    assert "tinyfish" not in PROVIDERS


def test_prohibited_paid_providers_raise_value_error():
    """Verify that attempting to use any prohibited paid provider raises ValueError."""
    for paid_key in PROHIBITED_PAID_PROVIDERS:
        with pytest.raises(ValueError, match="disabled"):
            estimate_cost(paid_key, queries=[])


@pytest.mark.asyncio
async def test_run_discovery_rejects_paid_providers():
    """Verify run_discovery rejects prohibited paid provider keys."""
    cfg = load_config()
    state = StateManager(db_path=":memory:")
    for paid_key in ("serper", "dataforseo_listings", "scavio_maps", "tinyfish"):
        with pytest.raises(ValueError, match="disabled"):
            await run_discovery(
                state=state,
                config=cfg,
                provider_key=paid_key,
            )


def test_scout_search_backends_restricted_to_free(monkeypatch):
    """Verify Scout._web_search excludes paid backends even if paid keys are in env."""
    from unittest.mock import AsyncMock

    cfg = load_config()
    state = StateManager(db_path=":memory:")
    scout = Scout(
        brain=Brain(state=state),
        state=state,
        config=cfg,
        env=SimpleNamespace(serper_api_key="paid_key_123", tinyfish_api_key="paid_key_456"),
    )

    serper_called = False

    async def mock_serper(*a, **k):
        nonlocal serper_called
        serper_called = True
        return [("https://example.com", "Paid result")]

    monkeypatch.setattr(scout, "_search_serper", mock_serper)
    monkeypatch.setattr(scout, "_search_duckduckgo", AsyncMock(return_value=[("https://free.org", "Free result")]))

    results = asyncio.run(scout._web_search("test query"))
    assert not serper_called
    assert results == [("https://free.org", "Free result")]


def test_osm_tag_map_expanded():
    """Verify OpenStreetMap maps physical business premises and trade tags, while software terms map to []."""
    osm = OpenStreetMap()
    # Physical premises and trades map to appropriate OSM tags (craft=, shop=, amenity=, office=)
    assert "roof" in osm.TAG_MAP
    assert "craft=roofer" in osm.TAG_MAP["roof"]
    assert "plumb" in osm.TAG_MAP
    assert "craft=plumber" in osm.TAG_MAP["plumb"]
    assert "dentist" in osm.TAG_MAP
    assert "amenity=dentist" in osm.TAG_MAP["dentist"]
    assert "lawyer" in osm.TAG_MAP
    assert "office=lawyer" in osm.TAG_MAP["lawyer"]
    # Intangible software terms that have no physical storefront/premises map to []
    assert osm._tags("software") == []
    assert osm._tags("saas") == []
    assert osm._tags("ai") == []
    assert osm._tags("b2b software") == []
