"""Run the production crawl path against candidate sites for a real query and
print what the saved rows would actually look like.

Usage: uv run python scripts/validate_sites.py "semiconductor supply chain"
"""
from __future__ import annotations

import asyncio
import re
import sys

from crawl4ai import AdaptiveConfig, AdaptiveCrawler, AsyncWebCrawler

CANDIDATES = [
    "https://www.cnbc.com/markets/",
    "https://www.investing.com/news/stock-market-news",
    "https://finviz.com/news.ashx",
    "https://seekingalpha.com/market-news",
    "https://apnews.com/hub/business",
    "https://www.npr.org/sections/business/",
    "https://www.prnewswire.com/news-releases/financial-services-latest-news/",
]

NAV_TOKENS = re.compile(
    r"(skip to (?:navigation|main content|right column)|sign in|subscribe|"
    r"shopping\.|newsletters|horoscope|games)",
    re.I,
)


def junk_ratio(text: str) -> float:
    if not text:
        return 1.0
    nav_hits = len(NAV_TOKENS.findall(text))
    words = max(len(text.split()), 1)
    return nav_hits / (words / 200)


async def validate(start_url: str, query: str) -> None:
    cfg = AdaptiveConfig(
        confidence_threshold=0.7, max_pages=30, top_k_links=3, strategy="statistical"
    )
    print(f"\n=== {start_url} ===")
    try:
        async with AsyncWebCrawler() as crawler:
            adaptive = AdaptiveCrawler(crawler, cfg)
            await adaptive.digest(start_url=start_url, query=query)
            pages = adaptive.get_relevant_content(top_k=3)
    except Exception as exc:
        print(f"  ERROR: {type(exc).__name__}: {exc}")
        return

    if not pages:
        print("  no pages returned")
        return

    for i, page in enumerate(pages):
        page = dict(page) if not isinstance(page, dict) else page
        url = page.get("url") or page.get("source_url") or "?"
        body = page.get("markdown") or page.get("content") or page.get("text") or ""
        body = str(body)
        score = page.get("score") or page.get("relevance_score")
        snippet = re.sub(r"\s+", " ", body[:400])
        print(f"\n  [{i}] url={url}")
        print(f"      score={score} chars={len(body)} junk_ratio={junk_ratio(body):.2f}")
        print(f"      preview: {snippet[:300]}")


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "semiconductor supply chain"
    print(f"query: {query!r}")
    for url in CANDIDATES:
        await validate(url, query)


if __name__ == "__main__":
    asyncio.run(main())
