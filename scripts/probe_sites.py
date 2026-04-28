"""Probe candidate sites with AsyncWebCrawler to see which return real content.

Usage: uv run python scripts/probe_sites.py
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

CANDIDATES = [
    # gov / no anti-bot
    ("SEC EDGAR full-text search", "https://efts.sec.gov/LATEST/search-index?q=%22semiconductor+supply+chain%22&dateRange=custom&startdt=2025-01-01&enddt=2026-04-27&forms=8-K"),
    ("SEC EDGAR press releases", "https://www.sec.gov/news/pressreleases"),
    ("Federal Reserve press", "https://www.federalreserve.gov/newsevents/pressreleases.htm"),
    ("Treasury press", "https://home.treasury.gov/news/press-releases"),
    # news / generally scrapable
    ("Hacker News front", "https://news.ycombinator.com/"),
    ("AP News business", "https://apnews.com/hub/business"),
    ("NPR business", "https://www.npr.org/sections/business/"),
    ("CNBC markets", "https://www.cnbc.com/markets/"),
    ("Yahoo Finance topic", "https://finance.yahoo.com/topic/stock-market-news/"),
    ("MarketWatch latest", "https://www.marketwatch.com/latest-news"),
    ("Reuters markets", "https://www.reuters.com/markets/"),
    # press release wires (lots of links, easy structure)
    ("PR Newswire financial", "https://www.prnewswire.com/news-releases/financial-services-latest-news/"),
    ("Business Wire", "https://www.businesswire.com/portal/site/home/news/"),
    ("GlobeNewswire", "https://www.globenewswire.com/en/news-feed"),
    # finance aggregators
    ("Finviz news", "https://finviz.com/news.ashx"),
    ("Investing.com news", "https://www.investing.com/news/stock-market-news"),
    ("Seeking Alpha market news", "https://seekingalpha.com/market-news"),
]


@dataclass
class Probe:
    label: str
    url: str
    status: str
    chars: int
    links: int
    sample_links: list[str]
    blocked_signal: str | None


BLOCK_PATTERNS = [
    re.compile(r"are you a (?:human|robot)", re.I),
    re.compile(r"verifying you are human", re.I),
    re.compile(r"access denied", re.I),
    re.compile(r"oops, something went wrong", re.I),
    re.compile(r"please enable (?:js|javascript)", re.I),
    re.compile(r"unsupported browser", re.I),
    re.compile(r"captcha", re.I),
    re.compile(r"cloudflare", re.I),
]


def detect_block(text: str) -> str | None:
    for pat in BLOCK_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


async def probe_one(crawler: AsyncWebCrawler, label: str, url: str) -> Probe:
    cfg = CrawlerRunConfig(page_timeout=20000, verbose=False)
    try:
        result = await crawler.arun(url=url, config=cfg)
    except Exception as exc:
        return Probe(label, url, f"error: {type(exc).__name__}: {exc}", 0, 0, [], None)

    if not result.success:
        return Probe(label, url, f"failed: {result.error_message}", 0, 0, [], None)

    md = result.markdown or ""
    if isinstance(md, object) and hasattr(md, "raw_markdown"):
        md = md.raw_markdown or str(md)
    md = str(md)
    block = detect_block(md[:4000])

    internal = result.links.get("internal", []) if result.links else []
    samples = []
    for link in internal[:20]:
        href = link.get("href") if isinstance(link, dict) else str(link)
        if href:
            samples.append(href)

    return Probe(
        label=label,
        url=url,
        status=f"ok HTTP {result.status_code}",
        chars=len(md),
        links=len(internal),
        sample_links=samples[:5],
        blocked_signal=block,
    )


async def main() -> None:
    bcfg = BrowserConfig(headless=True, verbose=False)
    async with AsyncWebCrawler(config=bcfg) as crawler:
        for label, url in CANDIDATES:
            p = await probe_one(crawler, label, url)
            verdict = "GOOD" if (p.chars > 1500 and p.links > 10 and not p.blocked_signal) else (
                "BLOCKED" if p.blocked_signal else "WEAK" if p.chars else "FAIL"
            )
            print(f"[{verdict:7}] {p.label:32} {p.status:18} chars={p.chars:>6} links={p.links:>4}"
                  + (f"  blocked='{p.blocked_signal}'" if p.blocked_signal else ""))
            for s in p.sample_links[:3]:
                print(f"            link: {s[:100]}")


if __name__ == "__main__":
    asyncio.run(main())
