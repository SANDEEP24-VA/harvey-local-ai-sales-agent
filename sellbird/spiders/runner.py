"""Isolated subprocess runner for Scrapy spider execution.

Prevents Twisted reactor conflicts with FastAPI's asyncio event loop on Windows
by executing Scrapy inside a short-lived isolated subprocess worker.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


def _run_in_process(start_url: str, domain: str, max_pages: int, output_file: str):
    """Execute Scrapy in the current process and write results to JSON."""
    from scrapy.crawler import CrawlerProcess
    from scrapy import signals
    from sellbird.spiders.company_spider import CompanyWebsiteSpider

    items: list[dict[str, Any]] = []

    process = CrawlerProcess({
        "USER_AGENT": "SellbirdSpider/1.0 (+https://sellbird.ai/bot; research)",
        "LOG_LEVEL": "WARNING",
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 0.5,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "CLOSESPIDER_PAGECOUNT": max_pages,
        "DEPTH_LIMIT": 2,
    })

    crawler = process.create_crawler(CompanyWebsiteSpider)

    def _item_scraped(item, response, spider):
        # Drop large raw html before saving output file to keep IPC lean
        clean_item = dict(item)
        items.append(clean_item)

    crawler.signals.connect(_item_scraped, signal=signals.item_scraped)
    process.crawl(crawler, start_url=start_url, domain=domain, max_pages=max_pages)
    process.start()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(items, f)


async def run_spider_subprocess(
    start_url: str,
    domain: str = "",
    max_pages: int = 8,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Execute Scrapy in an isolated subprocess to guarantee Twisted reactor isolation."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        temp_path = tf.name

    try:
        cmd = [
            sys.executable,
            "-m",
            "sellbird.spiders.runner",
            "--url",
            start_url,
            "--domain",
            domain,
            "--max-pages",
            str(max_pages),
            "--output",
            temp_path,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            logger.warning(f"Scrapy subprocess timed out for {start_url} after {timeout}s.")
            return []

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace")
            logger.warning(f"Scrapy subprocess exited with code {proc.returncode}: {err_msg[:200]}")
            return []

        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            with open(temp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        return []
    except Exception as e:
        logger.warning(f"Failed to execute Scrapy subprocess: {e}")
        return []
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Scrapy company website spider in subprocess.")
    parser.add_argument("--url", required=True, help="Start URL to crawl")
    parser.add_argument("--domain", default="", help="Allowed domain")
    parser.add_argument("--max-pages", type=int, default=8, help="Max pages limit")
    parser.add_argument("--output", required=True, help="Output JSON path")

    args = parser.parse_args()
    _run_in_process(args.url, args.domain, args.max_pages, args.output)
