"""Evidence-grounded web research engine for prospect intelligence.

Combines:
- BeautifulSoup (DOM parsing, text extraction, structured schema.org / JSON-LD)
- Playwright (dynamic JS-rendered pages when requested)
- Scrapy item compatibility / pipeline adapters
- SSRF prevention via PublicHTTPTransport
- Persistent local SQLite caching for offline reproducibility
- Robots.txt checking and polite domain rate-limiting
- Zero paid API dependencies
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import logging
import re
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
import urllib.robotparser

from bs4 import BeautifulSoup
import httpx

from sellbird.collectors.ats import detect_board, read_board, Board, Posting
from sellbird.collectors.people import find_people
from sellbird.models.company import Company
from sellbird.models.research import ResearchCacheItem
from sellbird.public_http import public_http_client, UnsafeURLError
from sellbird.spiders.runner import run_spider_subprocess
from sellbird.state import StateManager
from sellbird.untrusted import UNTRUSTED_RULE

logger = logging.getLogger("sellbird.research_engine")

# 4 distinct evidence verification statuses
STATUS_VERIFIED_OBSERVATION = "verified_observation"
STATUS_MODEL_INFERENCE = "model_inference"
STATUS_DETERMINISTIC_HEURISTIC = "deterministic_heuristic"
STATUS_UNVERIFIED = "unverified"
VERIFICATION_STATUSES = (
    STATUS_VERIFIED_OBSERVATION,
    STATUS_MODEL_INFERENCE,
    STATUS_DETERMINISTIC_HEURISTIC,
    STATUS_UNVERIFIED,
)

# Known marketing/tracking query parameters stripped during canonicalization
TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id",
    "fbclid", "gclid", "gad_source", "_ga", "_gl", "mc_cid", "mc_eid",
    "ref", "ref_src", "yclid", "msclkid",
})


def canonicalize_url(url: str) -> str:
    """Canonicalize a URL: strip fragments, tracking parameters, default ports, and normalize scheme/host."""
    if not url:
        return ""
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"https://{url}"

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Strip default ports
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    # Normalize path: collapse slashes, strip trailing slash
    path = parsed.path
    path = re.sub(r"/+", "/", path)
    if path == "/":
        path = ""
    elif len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    # Strip tracking parameters, sort remaining query parameters
    query_parts = []
    if parsed.query:
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        filtered_pairs = [
            (k, v) for k, v in query_pairs
            if k.lower() not in TRACKING_PARAMS
        ]
        if filtered_pairs:
            filtered_pairs.sort(key=lambda x: x[0])
            query_parts = [urlencode(filtered_pairs)]

    clean_query = "&".join(query_parts)
    return urlunparse((scheme, netloc, path, "", clean_query, ""))


def verify_evidence_quote(source_text: str, quote: str) -> bool:
    """Verify that an extracted quote or signal evidence appears verbatim in the source text."""
    if not quote or not source_text:
        return False
    norm_quote = re.sub(r"\s+", " ", quote).strip().lower()
    norm_source = re.sub(r"\s+", " ", source_text).strip().lower()
    if not norm_quote:
        return False
    return norm_quote in norm_source


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


class OfflineCacheMissError(Exception):
    """Raised when offline research requested a URL not present in the local cache."""
    pass


# Known web technology fingerprints
TECH_FINGERPRINTS = {
    "WordPress": [re.compile(r"wp-content", re.I), re.compile(r"wp-includes", re.I)],
    "Shopify": [re.compile(r"cdn\.shopify\.com", re.I), re.compile(r"Shopify\.theme", re.I)],
    "React": [re.compile(r"react(?:\.production|\.development)?\.js", re.I), re.compile(r"data-reactroot", re.I), re.compile(r"_next/", re.I)],
    "Next.js": [re.compile(r"/_next/static", re.I), re.compile(r"__NEXT_DATA__", re.I)],
    "Vue.js": [re.compile(r"vue(?:\.runtime)?\.js", re.I), re.compile(r"data-v-[a-f0-9]", re.I)],
    "HubSpot": [re.compile(r"js\.hs-scripts\.com", re.I), re.compile(r"hbspt", re.I)],
    "Stripe": [re.compile(r"js\.stripe\.com", re.I)],
    "Google Analytics": [re.compile(r"google-analytics\.com", re.I), re.compile(r"googletagmanager\.com", re.I)],
    "Datadog": [re.compile(r"datadoghq-browser", re.I)],
    "Tailwind CSS": [re.compile(r"tailwind", re.I)],
    "Bootstrap": [re.compile(r"bootstrap(?:\.min)?\.css", re.I)],
    "AWS S3/CloudFront": [re.compile(r"s3\.amazonaws\.com", re.I), re.compile(r"cloudfront\.net", re.I)],
}

# Hiring & expansion keyword signals
HIRING_KEYWORDS = [
    re.compile(r"\b(we're hiring|we are hiring|join our team|careers|open positions|current openings)\b", re.I),
    re.compile(r"\b(software engineer|frontend engineer|backend engineer|devops|sdr|account executive|ciso|product manager)\b", re.I),
]


class ResearchResult:
    """Structured extraction result from a researched web destination."""

    def __init__(
        self,
        url: str,
        domain: str,
        content_hash: str,
        title: str,
        meta_description: str,
        extracted_text: str,
        technologies: list[str],
        hiring_mentions: list[str],
        business_signals: list[dict],
        source_html: str,
        is_cached: bool,
        retrieved_at: str,
        status_code: int = 200,
        error: str = "",
    ):
        self.url = url
        self.domain = domain
        self.content_hash = content_hash
        self.title = title
        self.meta_description = meta_description
        self.extracted_text = extracted_text
        self.technologies = technologies
        self.hiring_mentions = hiring_mentions
        self.business_signals = business_signals
        self.source_html = source_html
        self.is_cached = is_cached
        self.retrieved_at = retrieved_at
        self.status_code = status_code
        self.error = error

    @property
    def from_cache(self) -> bool:
        return self.is_cached

    @property
    def metadata(self) -> dict:
        return {
            "title": self.title,
            "description": self.meta_description,
            "technologies": self.technologies,
            "hiring_roles": self.hiring_mentions,
            "business_signals": self.business_signals,
            "has_meta_ads": "Meta Pixel" in self.technologies or "Meta" in self.technologies or "fbq" in self.source_html.lower(),
            "has_booking_widget": any(w in self.extracted_text.lower() or w in self.source_html.lower() for w in ["calendly.com", "acuityscheduling.com", "book a demo", "schedule a call"]),
        }

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "domain": self.domain,
            "content_hash": self.content_hash,
            "title": self.title,
            "meta_description": self.meta_description,
            "extracted_text": self.extracted_text[:2000],
            "technologies": self.technologies,
            "hiring_mentions": self.hiring_mentions,
            "business_signals": self.business_signals,
            "is_cached": self.is_cached,
            "retrieved_at": self.retrieved_at,
            "status_code": self.status_code,
            "error": self.error,
        }

    def to_scrapy_item(self) -> dict:
        """Compatibility export for Scrapy pipeline consumers."""
        return {
            "url": self.url,
            "domain": self.domain,
            "checksum": self.content_hash,
            "title": self.title,
            "description": self.meta_description,
            "technologies": self.technologies,
            "hiring": self.hiring_mentions,
            "signals": self.business_signals,
            "crawled_at": self.retrieved_at,
            "from_cache": self.is_cached,
        }


class ResearchEngine:
    """Evidence-grounded web researcher with persistent caching and rate-limiting."""

    def __init__(self, state: StateManager | None = None, rate_limit_seconds: float = 1.0):
        self.state = state
        self.rate_limit_seconds = rate_limit_seconds
        self._last_request_times: dict[str, float] = {}
        self._robots_parsers: dict[str, urllib.robotparser.RobotFileParser] = {}

    def extract_from_html(self, html: str, url: str = "https://example.com") -> tuple[str, dict]:
        """Convenience method to extract plain text and metadata dictionary from HTML."""
        res = self.parse_html_content(url=url, html=html, retrieved_at=_utcnow_iso())
        return res.extracted_text, res.metadata

    async def scrape(
        self,
        url: str,
        offline_only: bool = False,
        force_refresh: bool = False,
        render_js: bool = False,
    ) -> ResearchResult:
        """Safe scraping entrypoint returning ResearchResult with status codes and graceful offline errors."""
        try:
            return await self.research_url(
                url=url,
                force_refresh=force_refresh,
                render_js=render_js,
                offline_only=offline_only,
            )
        except OfflineCacheMissError as e:
            norm_url = self._normalize_url(url)
            return ResearchResult(
                url=norm_url,
                domain=urlparse(norm_url).netloc.lower(),
                content_hash="",
                title="",
                meta_description="",
                extracted_text="",
                technologies=[],
                hiring_mentions=[],
                business_signals=[],
                source_html="",
                is_cached=False,
                retrieved_at=_utcnow_iso(),
                status_code=404,
                error=f"Offline mode: {e}",
            )

    async def _cache_content(
        self,
        url: str,
        domain: str,
        html: str,
        text: str,
        metadata: dict | None = None,
    ) -> None:
        """Manually populate local cache item (useful for tests and seed data)."""
        if not self.state:
            return
        content_hash = hashlib.sha256(html.encode("utf-8")).hexdigest()
        cache_item = ResearchCacheItem(
            url=self._normalize_url(url),
            domain=domain,
            content_hash=content_hash,
            content_type="text/html",
            raw_content=html,
            extracted_text=text,
            metadata=metadata or {},
            status_code=200,
        )
        await self.state.save_research_cache(cache_item)

    def _normalize_url(self, url: str) -> str:
        return canonicalize_url(url)

    async def _check_robots_txt(self, url: str) -> bool:
        """Check robots.txt compliance for the target URL."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if not domain:
            return True

        if domain in self._robots_parsers:
            rp = self._robots_parsers[domain]
            return rp.can_fetch("SellbirdResearcher/1.0", url)

        robots_url = f"{parsed.scheme}://{domain}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        try:
            async with public_http_client(timeout=5.0) as client:
                res = await client.get(robots_url)
                if res.status_code == 200:
                    rp.parse(res.text.splitlines())
                else:
                    rp.allow_all = True
        except Exception:
            rp.allow_all = True

        self._robots_parsers[domain] = rp
        return rp.can_fetch("SellbirdResearcher/1.0", url)

    async def _rate_limit(self, domain: str) -> None:
        """Ensure respectful request pacing per domain."""
        now = time.monotonic()
        last = self._last_request_times.get(domain, 0.0)
        elapsed = now - last
        if elapsed < self.rate_limit_seconds:
            await asyncio.sleep(self.rate_limit_seconds - elapsed)
        self._last_request_times[domain] = time.monotonic()

    def parse_html_content(self, url: str, html: str, retrieved_at: str, is_cached: bool = False) -> ResearchResult:
        """Extract structured facts, technologies, and hiring signals from HTML using BeautifulSoup."""
        soup = BeautifulSoup(html, "html.parser")
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Content hash
        content_hash = hashlib.sha256(html.encode("utf-8")).hexdigest()

        # Title & Meta Description
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        meta_desc = ""
        desc_tag = soup.find("meta", attrs={"name": re.compile(r"description", re.I)}) or \
                   soup.find("meta", attrs={"property": re.compile(r"og:description", re.I)})
        if desc_tag and desc_tag.get("content"):
            meta_desc = desc_tag["content"].strip()

        # Clean script/style tags for text extraction
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        raw_text = soup.get_text(separator=" ", strip=True)
        # Collapse whitespace
        extracted_text = re.sub(r"\s+", " ", raw_text)

        # Detect technologies
        technologies = []
        for tech, patterns in TECH_FINGERPRINTS.items():
            for pat in patterns:
                if pat.search(html):
                    technologies.append(tech)
                    break

        # Detect hiring mentions
        hiring_mentions = []
        role_list_match = re.search(r"(?:we are hiring|we're hiring|careers|open positions)[:\s]+([^.]+)", extracted_text, re.I)
        if role_list_match:
            raw_roles = re.split(r",|\band\b", role_list_match.group(1))
            for r in raw_roles:
                cleaned = r.strip(" :.-")
                if cleaned and len(cleaned) < 60:
                    hiring_mentions.append(cleaned)

        for pat in HIRING_KEYWORDS:
            matches = pat.findall(extracted_text)
            if matches:
                for m in set(matches[:3]):
                    hiring_mentions.append(m if isinstance(m, str) else m[0])

        # Extract structured headings / value signals
        business_signals = []
        for h in soup.find_all(["h1", "h2"])[:10]:
            h_text = h.get_text(strip=True)
            if len(h_text) > 8:
                business_signals.append({"tag": h.name, "text": h_text})

        return ResearchResult(
            url=url,
            domain=domain,
            content_hash=content_hash,
            title=title,
            meta_description=meta_desc,
            extracted_text=extracted_text,
            technologies=sorted(list(set(technologies))),
            hiring_mentions=sorted(list(set(hiring_mentions))),
            business_signals=business_signals,
            source_html=html,
            is_cached=is_cached,
            retrieved_at=retrieved_at,
        )

    async def research_url(
        self,
        url: str,
        force_refresh: bool = False,
        render_js: bool = False,
        offline_only: bool = False,
    ) -> ResearchResult:
        """Evidence-grounded web research with persistent caching, SSRF guards, and offline support."""
        norm_url = self._normalize_url(url)
        parsed = urlparse(norm_url)
        domain = parsed.netloc.lower()

        # 1. Check local research cache
        cached = await self.state.get_research_cache(norm_url)
        if cached and not force_refresh:
            logger.debug(f"ResearchEngine: Cache hit for {norm_url}")
            return self.parse_html_content(
                url=norm_url,
                html=cached.raw_content,
                retrieved_at=cached.retrieved_at.isoformat() if hasattr(cached.retrieved_at, "isoformat") else str(cached.retrieved_at),
                is_cached=True,
            )

        if offline_only:
            if cached:
                return self.parse_html_content(
                    url=norm_url,
                    html=cached.raw_content,
                    retrieved_at=cached.retrieved_at.isoformat() if hasattr(cached.retrieved_at, "isoformat") else str(cached.retrieved_at),
                    is_cached=True,
                )
            raise OfflineCacheMissError(f"URL {norm_url} not found in offline research cache.")

        # 2. Check robots.txt
        allowed = await self._check_robots_txt(norm_url)
        if not allowed:
            logger.warning(f"ResearchEngine: URL blocked by robots.txt: {norm_url}")
            if cached:
                return self.parse_html_content(
                    url=norm_url, html=cached.raw_content, retrieved_at=str(cached.retrieved_at), is_cached=True
                )
            raise PermissionError(f"Research on {norm_url} prohibited by website robots.txt.")

        # 3. Rate limit per domain
        await self._rate_limit(domain)

        html = ""
        status_code = 200
        now_str = _utcnow_iso()

        # 4. Fetch via Playwright if dynamic JS rendering requested, else SSRF-safe HTTP
        if render_js:
            try:
                from playwright.async_api import async_playwright
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page()
                    await page.goto(norm_url, timeout=15000, wait_until="domcontentloaded")
                    html = await page.content()
                    await browser.close()
            except Exception as e:
                logger.warning(f"Playwright render failed for {norm_url}: {e}; falling back to HTTP.")
                html = ""

        if not html:
            # SSRF-guarded HTTP request
            try:
                async with public_http_client(timeout=15.0) as client:
                    resp = await client.get(norm_url)
                    status_code = resp.status_code
                    html = resp.text
            except (httpx.ConnectError, httpx.TimeoutException, UnsafeURLError) as e:
                # If network fails but cache exists, return cached
                if cached:
                    logger.info(f"Network error ({e}); using cached research for {norm_url}")
                    return self.parse_html_content(
                        url=norm_url, html=cached.raw_content, retrieved_at=str(cached.retrieved_at), is_cached=True
                    )
                raise

        # 5. Parse structured result
        res = self.parse_html_content(url=norm_url, html=html, retrieved_at=now_str, is_cached=False)

        # 6. Save into persistent research cache
        cache_item = ResearchCacheItem(
            url=norm_url,
            domain=domain,
            content_hash=res.content_hash,
            content_type="text/html",
            raw_content=html,
            extracted_text=res.extracted_text,
            metadata={
                "title": res.title,
                "meta_description": res.meta_description,
                "technologies": res.technologies,
                "hiring_mentions": res.hiring_mentions,
            },
            status_code=status_code,
        )
        await self.state.save_research_cache(cache_item)
        return res

    async def research_company(
        self, company: Company, max_pages: int = 3, offline_only: bool = False
    ) -> list[ResearchResult]:
        """Deep research of a company's web presence across homepage, about, and careers."""
        base_url = company.website or (f"https://{company.domain}" if company.domain else "")
        if not base_url:
            return []

        norm_base = self._normalize_url(base_url)
        results = []

        # Target paths
        target_urls = [norm_base]
        if max_pages > 1:
            target_urls.append(urljoin(norm_base, "/about"))
        if max_pages > 2:
            target_urls.append(urljoin(norm_base, "/careers"))

        seen_hashes = set()
        for target in target_urls:
            try:
                res = await self.research_url(target, offline_only=offline_only)
                if res.content_hash not in seen_hashes:
                    seen_hashes.add(res.content_hash)
                    results.append(res)
            except Exception as e:
                logger.debug(f"ResearchEngine: failed researching {target}: {e}")

        return results

    async def crawl_company_website(
        self,
        domain_or_url: str,
        max_pages: int = 8,
        use_scrapy: bool = True,
        render_js: bool = False,
        offline_only: bool = False,
    ) -> dict[str, Any]:
        """Crawl key public pages of a company website and return structured research."""
        return await crawl_company_website(
            domain_or_url=domain_or_url,
            max_pages=max_pages,
            use_scrapy=use_scrapy,
            render_js=render_js,
            offline_only=offline_only,
            state=self.state,
        )


async def crawl_company_website(
    domain_or_url: str,
    max_pages: int = 8,
    use_scrapy: bool = True,
    render_js: bool = False,
    offline_only: bool = False,
    state: StateManager | None = None,
) -> dict[str, Any]:
    """Reusable multi-page company research orchestrator.

    Crawls or reads from cache key pages of a company website (Homepage, About,
    Products/Services, Pricing, Careers, Team, Contact).
    Extracts technologies, hiring mentions, ATS job boards, and executive decision-makers.
    """
    clean_target = (domain_or_url or "").strip()
    if not clean_target:
        return {
            "domain": "",
            "base_url": "",
            "pages_crawled": 0,
            "pages": [],
            "technologies": [],
            "hiring_roles": [],
            "decision_makers": [],
            "ats_board": None,
            "signals": [],
            "from_cache": True,
        }

    if "://" in clean_target:
        domain = urlparse(clean_target).netloc.lower().lstrip("www.")
        base_url = clean_target
    else:
        domain = clean_target.split("/")[0].lower().lstrip("www.")
        base_url = f"https://{domain}"

    canonical_base = canonicalize_url(base_url)
    engine = ResearchEngine(state=state)

    # 1. Offline Mode: STRICT ZERO NETWORK
    if offline_only:
        cached_items = []
        if state:
            cached_items = await state.list_research_cache(domain=domain, limit=max_pages)
            if not cached_items and domain.startswith("www."):
                cached_items = await state.list_research_cache(domain=domain[4:], limit=max_pages)
            elif not cached_items:
                cached_items = await state.list_research_cache(domain=f"www.{domain}", limit=max_pages)

        if not cached_items:
            return {
                "domain": domain,
                "base_url": canonical_base,
                "pages_crawled": 0,
                "pages": [],
                "technologies": [],
                "hiring_roles": [],
                "decision_makers": [],
                "ats_board": None,
                "signals": [],
                "from_cache": True,
            }

        parsed_results = []
        pages_tuples = []
        all_tech = set()
        all_hiring = []
        all_signals = []
        detected_board = None

        for item in cached_items:
            res = engine.parse_html_content(
                url=item.url,
                html=item.raw_content,
                retrieved_at=item.retrieved_at.isoformat() if hasattr(item.retrieved_at, "isoformat") else str(item.retrieved_at),
                is_cached=True,
            )
            parsed_results.append(res)
            pages_tuples.append((item.url, item.raw_content))
            all_tech.update(res.technologies)
            all_hiring.extend(res.hiring_mentions)
            all_signals.extend(res.business_signals)
            if not detected_board and item.raw_content:
                detected_board = detect_board(item.raw_content, domain=domain)

        decision_makers = find_people(pages_tuples, domain=domain) if pages_tuples else []
        ats_info = None
        if detected_board:
            ats_info = {
                "ats": detected_board.ats,
                "token": detected_board.token,
                "label": detected_board.label,
                "page_url": detected_board.page_url,
            }

        return {
            "domain": domain,
            "base_url": canonical_base,
            "pages_crawled": len(parsed_results),
            "pages": [r.to_dict() for r in parsed_results],
            "technologies": sorted(list(all_tech)),
            "hiring_roles": [{"title": h} for h in sorted(list(set(all_hiring)))],
            "decision_makers": decision_makers,
            "ats_board": ats_info,
            "signals": all_signals[:20],
            "from_cache": True,
        }

    # 2. Online Mode
    research_results: list[ResearchResult] = []
    seen_hashes: set[str] = set()
    pages_tuples: list[tuple[str, str]] = []

    # Check robots.txt first
    allowed = await engine._check_robots_txt(canonical_base)
    if not allowed:
        logger.warning(f"Crawl prohibited by robots.txt for {canonical_base}")
        return {
            "domain": domain,
            "base_url": canonical_base,
            "pages_crawled": 0,
            "pages": [],
            "technologies": [],
            "hiring_roles": [],
            "decision_makers": [],
            "ats_board": None,
            "signals": [],
            "from_cache": False,
            "error": "Prohibited by robots.txt",
        }

    # Try Scrapy subprocess runner if requested
    if use_scrapy:
        try:
            spider_items = await run_spider_subprocess(
                start_url=canonical_base,
                domain=domain,
                max_pages=max_pages,
                timeout=30.0,
            )
            for item in spider_items:
                item_url = canonicalize_url(item.get("url", ""))
                html = item.get("html", "")
                if not html and item.get("text"):
                    html = f"<html><head><title>{item.get('title', '')}</title></head><body><h1>{item.get('title', '')}</h1><p>{item.get('text', '')}</p></body></html>"

                res = engine.parse_html_content(
                    url=item_url,
                    html=html,
                    retrieved_at=_utcnow_iso(),
                    is_cached=False,
                )
                if res.content_hash not in seen_hashes:
                    seen_hashes.add(res.content_hash)
                    research_results.append(res)
                    if html:
                        pages_tuples.append((item_url, html))
                    if state:
                        await engine._cache_content(
                            url=item_url,
                            domain=domain,
                            html=html,
                            text=res.extracted_text,
                            metadata=res.metadata,
                        )
        except Exception as e:
            logger.warning(f"Scrapy runner error ({e}); falling back to HTTP multi-page fetch.")

    # If Scrapy didn't return pages or was disabled, fetch via HTTP
    if not research_results:
        target_paths = [
            "/",
            "/about",
            "/company",
            "/services",
            "/products",
            "/pricing",
            "/careers",
            "/team",
            "/contact",
        ]
        target_urls = [canonicalize_url(urljoin(canonical_base, p)) for p in target_paths][:max_pages]

        for target in target_urls:
            try:
                res = await engine.research_url(
                    target,
                    render_js=render_js,
                    offline_only=False,
                )
                if res.content_hash not in seen_hashes:
                    seen_hashes.add(res.content_hash)
                    research_results.append(res)
                    if res.source_html:
                        pages_tuples.append((res.url, res.source_html))
            except Exception as e:
                logger.debug(f"crawl_company_website: error researching {target}: {e}")

    # Extract decision makers using sellbird.collectors.people
    decision_makers = []
    if pages_tuples:
        try:
            decision_makers = find_people(pages_tuples, domain=domain)
        except Exception as e:
            logger.warning(f"Decision maker extraction error: {e}")

    # Detect ATS job boards and fetch live open roles
    detected_board = None
    open_roles = []
    for res in research_results:
        if res.source_html:
            board = detect_board(res.source_html, domain=domain)
            if board:
                detected_board = board
                break

    ats_info = None
    if detected_board:
        ats_info = {
            "ats": detected_board.ats,
            "token": detected_board.token,
            "label": detected_board.label,
            "page_url": detected_board.page_url,
        }
        try:
            async with public_http_client(timeout=10.0) as client:
                postings = await read_board(client, detected_board)
                if postings:
                    for p in postings[:25]:
                        open_roles.append({
                            "title": p.title,
                            "department": p.department,
                            "location": p.location,
                            "posted": p.posted,
                            "url": p.url,
                            "summary": p.summary(),
                        })
        except Exception as e:
            logger.debug(f"ATS live feed fetch failed ({e})")

    # Aggregate technologies, hiring mentions, business signals
    all_tech = set()
    all_hiring = []
    all_signals = []
    for r in research_results:
        all_tech.update(r.technologies)
        all_hiring.extend(r.hiring_mentions)
        all_signals.extend(r.business_signals)

    if not open_roles and all_hiring:
        open_roles = [{"title": h} for h in sorted(list(set(all_hiring)))]

    return {
        "domain": domain,
        "base_url": canonical_base,
        "pages_crawled": len(research_results),
        "pages": [r.to_dict() for r in research_results],
        "technologies": sorted(list(all_tech)),
        "hiring_roles": open_roles,
        "decision_makers": decision_makers,
        "ats_board": ats_info,
        "signals": all_signals[:20],
        "from_cache": all(r.is_cached for r in research_results) if research_results else False,
    }
