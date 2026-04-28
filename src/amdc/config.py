"""Crawler targets and tunables. Edit SITES to swap target domains."""

from __future__ import annotations

SITES: list[dict] = [
    {
        # CNBC: articles live under dated paths or /video/<date>/
        "url": "https://www.cnbc.com/markets/",
        "url_patterns": [
            "*/2025/*",
            "*/2026/*",
            "*/video/2025/*",
            "*/video/2026/*",
        ],
        "include_external": False,
    },
    {
        # Investing.com: drill into specific news subsections + analysis
        "url": "https://www.investing.com/news/stock-market-news",
        "url_patterns": ["*/news/*-news/*", "*/analysis/*"],
        "include_external": False,
    },
    {
        # Finviz news.ashx is an outbound-link aggregator; allow externals
        "url": "https://finviz.com/news.ashx",
        "url_patterns": ["*"],
        "include_external": True,
    },
]

TEXT_CHAR_CAP: int = 8000

# BestFirst deep-crawl tunables (per site)
DEEP_CRAWL_MAX_DEPTH: int = 4  # hub → articles → linked articles
DEEP_CRAWL_MAX_PAGES: int = 100  # ceiling per site
# Post-fetch gate: keep a page if EITHER raw body clears MIN_RAW or BM25-trimmed
# body clears MIN_FIT. Two floors so over-aggressive BM25 trim on long articles
# doesn't drop them.
MIN_FIT_MARKDOWN_CHARS: int = 200
MIN_RAW_MARKDOWN_CHARS: int = 800
CONCURRENCY_PER_SITE: int = 10  # parallel page fetches inside one site's deep crawl

# Cross-site parallelism
PARALLEL_SITES: bool = True  # asyncio.gather across SITES instead of sequential

# Content filter + stealth
BM25_THRESHOLD: float = 0.3
PAGE_TIMEOUT_MS: int = 20_000  # per-page browser timeout
STEALTH: bool = True  # keep magic + override_navigator
SIMULATE_USER: bool = False  # disabled: mouse-move overhead rarely changes outcome
