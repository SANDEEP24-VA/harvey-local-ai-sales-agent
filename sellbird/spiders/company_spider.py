"""Structured company website spider using Scrapy.

Complies strictly with robots.txt, respects per-domain rate limits,
applies bounded crawl depths, and extracts clean sectioned page content.
"""

import re
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import scrapy


class CompanyWebsiteSpider(scrapy.Spider):
    """Scrapy spider crawling key public pages of a single company domain."""

    name = "company_website"

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 0.5,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "DEPTH_LIMIT": 2,
        "CLOSESPIDER_PAGECOUNT": 10,
        "USER_AGENT": "SellbirdSpider/1.0 (+https://sellbird.ai/bot; research)",
        "LOG_LEVEL": "WARNING",
    }

    # Patterns matching target subpages
    PATH_PATTERNS = {
        "about": re.compile(r"/(about|about-us|company|story|who-we-are)$", re.I),
        "products_services": re.compile(r"/(services|solutions|products|platform|features)$", re.I),
        "pricing": re.compile(r"/(pricing|plans|packages)$", re.I),
        "careers": re.compile(r"/(careers|jobs|join-us|work-with-us)$", re.I),
        "team": re.compile(r"/(team|our-team|leadership|people|founders)$", re.I),
        "contact": re.compile(r"/(contact|contact-us|get-in-touch)$", re.I),
    }

    def __init__(self, start_url: str = "", domain: str = "", max_pages: int = 8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_url = start_url
        if not domain and start_url:
            self.domain = urlparse(start_url).netloc.lower().lstrip("www.")
        else:
            self.domain = domain.lower().lstrip("www.")

        if self.domain:
            self.allowed_domains = [self.domain, f"www.{self.domain}"]
        else:
            self.allowed_domains = []

        self.max_pages = int(max_pages)
        self.pages_crawled = 0
        self.visited_paths = set()

    def start_requests(self):
        if not self.start_url and self.domain:
            self.start_url = f"https://{self.domain}"
        if self.start_url:
            yield scrapy.Request(
                self.start_url,
                callback=self.parse,
                errback=self.errback,
                meta={"page_type": "homepage", "depth": 0},
            )

    def _classify_path(self, path: str) -> str:
        clean = path.rstrip("/")
        for ptype, pat in self.PATH_PATTERNS.items():
            if pat.search(clean):
                return ptype
        return "other"

    def parse(self, response):
        if not isinstance(response, scrapy.http.HtmlResponse):
            return

        self.pages_crawled += 1
        url = response.url
        page_type = response.meta.get("page_type", "other")

        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        desc = ""
        meta_desc = soup.find("meta", attrs={"name": re.compile(r"description", re.I)}) or \
                    soup.find("meta", attrs={"property": re.compile(r"og:description", re.I)})
        if meta_desc and meta_desc.get("content"):
            desc = meta_desc["content"].strip()

        # Headings
        headings = []
        for h in soup.find_all(["h1", "h2", "h3"])[:15]:
            h_text = h.get_text(strip=True)
            if len(h_text) > 4:
                headings.append({"tag": h.name, "text": h_text})

        # Strip unneeded tags
        for t in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
            t.decompose()

        raw_text = soup.get_text(separator=" ", strip=True)
        cleaned_text = re.sub(r"\s+", " ", raw_text)

        yield {
            "url": url,
            "domain": self.domain,
            "page_type": page_type,
            "title": title,
            "meta_description": desc,
            "text": cleaned_text,
            "headings": headings,
            "html": response.text,
            "status_code": response.status,
            "depth": response.meta.get("depth", 0),
        }

        # Follow links if not yet at max pages
        if self.pages_crawled < self.max_pages and response.meta.get("depth", 0) < 2:
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
                    continue

                full_url = urljoin(response.url, href)
                parsed = urlparse(full_url)
                netloc = parsed.netloc.lower().lstrip("www.")
                if netloc != self.domain:
                    continue

                path = parsed.path.rstrip("/")
                if not path or path in self.visited_paths:
                    continue

                ptype = self._classify_path(path)
                if ptype != "other" or response.meta.get("depth", 0) == 0:
                    self.visited_paths.add(path)
                    yield response.follow(
                        full_url,
                        callback=self.parse,
                        errback=self.errback,
                        meta={"page_type": ptype, "depth": response.meta.get("depth", 0) + 1},
                    )

    def errback(self, failure):
        self.logger.warning(f"Scrapy request error for {self.domain}: {failure}")
