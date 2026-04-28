"""Run BestFirstCrawlingStrategy over the configured sites with parallelism.

Parallelism layers:
  * Cross-site: all SITES are crawled concurrently via asyncio.gather, sharing
    one AsyncWebCrawler (one browser process, multiple contexts).
  * Intra-site: BestFirstCrawlingStrategy fetches up to CONCURRENCY_PER_SITE
    pages in parallel (controlled by CrawlerRunConfig.semaphore_count).

Performance knobs:
  * BrowserConfig.text_mode + light_mode  — text-only rendering, faster paint.
  * CrawlerRunConfig.cache_mode=ENABLED   — re-runs reuse fetched pages.
  * wait_for_images=False, exclude_external_images=True, exclude_external_links=True.
  * remove_overlay_elements=True          — auto-dismiss cookie popups.

Query-driven components (unchanged):
  * KeywordRelevanceScorer  — orders link discovery toward query-relevant URLs.
  * BM25ContentFilter       — content pruning inside each fetched page.
"""
from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
)
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (
    ContentTypeFilter,
    DomainFilter,
    FilterChain,
    URLPatternFilter,
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from amdc.config import (
    BM25_THRESHOLD,
    CONCURRENCY_PER_SITE,
    DEEP_CRAWL_MAX_DEPTH,
    DEEP_CRAWL_MAX_PAGES,
    MIN_FIT_MARKDOWN_CHARS,
    MIN_RAW_MARKDOWN_CHARS,
    PAGE_TIMEOUT_MS,
    PARALLEL_SITES,
    SIMULATE_USER,
    SITES,
    STEALTH,
)

log = logging.getLogger(__name__)


def _domain(url: str) -> str:
    return urlparse(url).netloc


def _md_fields(markdown_obj) -> tuple[str, str]:
    """Return (fit_markdown, raw_markdown) regardless of whether markdown is a
    MarkdownGenerationResult or a plain string."""
    if markdown_obj is None:
        return "", ""
    fit = getattr(markdown_obj, "fit_markdown", None)
    raw = getattr(markdown_obj, "raw_markdown", None)
    if fit is None and raw is None:
        s = str(markdown_obj)
        return "", s
    return (fit or ""), (raw or "")


def _build_run_config(site: dict, query: str) -> CrawlerRunConfig:
    netloc = _domain(site["url"])
    keywords = [k for k in query.split() if k] or [query]

    domain_filter = (
        DomainFilter(allowed_domains=[netloc])
        if not site["include_external"]
        else DomainFilter()
    )
    filters = FilterChain(
        [
            domain_filter,
            URLPatternFilter(patterns=site["url_patterns"]),
            ContentTypeFilter(allowed_types=["text/html"]),
        ]
    )
    scorer = KeywordRelevanceScorer(keywords=keywords, weight=1.0)
    strategy = BestFirstCrawlingStrategy(
        max_depth=DEEP_CRAWL_MAX_DEPTH,
        max_pages=DEEP_CRAWL_MAX_PAGES,
        url_scorer=scorer,
        filter_chain=filters,
        include_external=site["include_external"],
    )
    md_gen = DefaultMarkdownGenerator(
        content_filter=BM25ContentFilter(
            user_query=query, bm25_threshold=BM25_THRESHOLD
        )
    )
    return CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        markdown_generator=md_gen,
        cache_mode=CacheMode.ENABLED,
        semaphore_count=CONCURRENCY_PER_SITE,
        magic=STEALTH,
        simulate_user=SIMULATE_USER,
        override_navigator=STEALTH,
        page_timeout=PAGE_TIMEOUT_MS,
        wait_for_images=False,
        exclude_external_images=True,
        exclude_external_links=not site["include_external"],
        remove_overlay_elements=True,
        stream=True,
    )


async def _crawl_one(crawler: AsyncWebCrawler, site: dict, query: str) -> list[dict]:
    start_url = site["url"]
    netloc = _domain(start_url)
    run_cfg = _build_run_config(site, query)

    out: list[dict] = []
    try:
        async for result in await crawler.arun(url=start_url, config=run_cfg):
            if not getattr(result, "success", False):
                continue
            # Skip the hub/start URL itself — it's nav, not an article.
            if result.url.rstrip("/") == start_url.rstrip("/"):
                continue
            fit, raw = _md_fields(getattr(result, "markdown", None))
            # Keep a page if either the raw body is substantial OR the
            # BM25-filtered body has enough query-relevant content. Gating on
            # fit alone drops long articles where BM25 over-trims.
            if len(raw) < MIN_RAW_MARKDOWN_CHARS and len(fit) < MIN_FIT_MARKDOWN_CHARS:
                continue
            metadata = getattr(result, "metadata", None) or {}
            score = metadata.get("score") if isinstance(metadata, dict) else None
            out.append(
                {
                    "url": result.url,
                    "fit_markdown": fit,
                    "raw_markdown": raw,
                    "metadata": metadata,
                    "score": score,
                    "_source_start_url": start_url,
                    "_source_domain": netloc,
                }
            )
    except Exception as exc:
        log.error("site failed (%s): %s", start_url, exc, exc_info=True)
    return out


async def crawl_all(query: str) -> list[dict]:
    browser = BrowserConfig(
        headless=True,
        viewport_width=1920,
        viewport_height=1080,
        user_agent_mode="random" if STEALTH else None,
        text_mode=True,
        light_mode=True,
        verbose=False,
    )

    async with AsyncWebCrawler(config=browser) as crawler:
        if PARALLEL_SITES:
            tasks = [_crawl_one(crawler, site, query) for site in SITES]
            log.info("crawling %d sites in parallel for query=%r", len(SITES), query)
            per_site = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            per_site = []
            for site in SITES:
                log.info("crawling %s for query=%r", site["url"], query)
                per_site.append(await _crawl_one(crawler, site, query))

    results: list[dict] = []
    for site, res in zip(SITES, per_site):
        if isinstance(res, Exception):
            log.error("site task raised (%s): %s", site["url"], res)
            continue
        log.info("got %d pages from %s", len(res), site["url"])
        results.extend(res)
    return results
