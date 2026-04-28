from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from amdc import crawler as crawler_module
from amdc.crawler import _crawl_one, crawl_all


@dataclass
class FakeMarkdown:
    fit_markdown: str = ""
    raw_markdown: str = ""


@dataclass
class FakeResult:
    url: str
    success: bool = True
    markdown: FakeMarkdown = field(default_factory=FakeMarkdown)
    metadata: dict[str, Any] | None = None


class FakeCrawler:
    """Drop-in for AsyncWebCrawler exposing only `arun`."""

    def __init__(
        self,
        results_per_url: dict[str, list[FakeResult]] | None = None,
        raise_on: dict[str, Exception] | None = None,
    ) -> None:
        self._results = results_per_url or {}
        self._raise = raise_on or {}

    async def arun(self, *, url: str, config: Any):  # noqa: D401 - mirrors crawl4ai
        if url in self._raise:
            raise self._raise[url]
        results = self._results.get(url, [])

        async def _gen():
            for r in results:
                yield r

        return _gen()


def _site(url: str = "https://cnbc.com/markets") -> dict:
    return {
        "url": url,
        "url_patterns": ["*"],
        "include_external": False,
    }


def _run(coro):
    return asyncio.run(coro)


def test_crawl_one_keeps_long_raw_pages_and_counts_zero_dropped() -> None:
    site = _site()
    results = [
        FakeResult(
            url="https://cnbc.com/markets/article-1",
            markdown=FakeMarkdown(fit_markdown="", raw_markdown="x" * 1000),
            metadata={"score": 0.9},
        )
    ]
    crawler = FakeCrawler({site["url"]: results})

    records, stats = _run(_crawl_one(crawler, site, "markets"))

    assert len(records) == 1
    assert records[0]["url"] == "https://cnbc.com/markets/article-1"
    assert records[0]["_source_domain"] == "cnbc.com"
    assert stats["pages_kept"] == 1
    assert stats["pages_dropped"] == 0
    assert stats["error"] is None
    assert stats["site"] == "cnbc.com"


def test_crawl_one_drops_pages_below_min_thresholds() -> None:
    site = _site()
    results = [
        FakeResult(
            url="https://cnbc.com/markets/short",
            markdown=FakeMarkdown(fit_markdown="too short", raw_markdown="also short"),
        )
    ]
    crawler = FakeCrawler({site["url"]: results})

    records, stats = _run(_crawl_one(crawler, site, "markets"))

    assert records == []
    assert stats["pages_kept"] == 0
    assert stats["pages_dropped"] == 1


def test_crawl_one_keeps_pages_with_long_fit_even_if_raw_short() -> None:
    site = _site()
    results = [
        FakeResult(
            url="https://cnbc.com/markets/long-fit",
            markdown=FakeMarkdown(fit_markdown="y" * 300, raw_markdown="short"),
        )
    ]
    crawler = FakeCrawler({site["url"]: results})

    records, stats = _run(_crawl_one(crawler, site, "markets"))

    assert len(records) == 1
    assert stats["pages_kept"] == 1


def test_crawl_one_skips_start_url_without_counting_drop() -> None:
    site = _site()
    results = [
        FakeResult(url=site["url"], markdown=FakeMarkdown(raw_markdown="x" * 1000)),
        FakeResult(
            url="https://cnbc.com/markets/keep",
            markdown=FakeMarkdown(raw_markdown="x" * 1000),
        ),
    ]
    crawler = FakeCrawler({site["url"]: results})

    records, stats = _run(_crawl_one(crawler, site, "markets"))

    assert len(records) == 1
    assert stats["pages_kept"] == 1
    assert stats["pages_dropped"] == 0


def test_crawl_one_skips_unsuccessful_results() -> None:
    site = _site()
    results = [
        FakeResult(
            url="https://cnbc.com/markets/fail",
            success=False,
            markdown=FakeMarkdown(raw_markdown="x" * 1000),
        )
    ]
    crawler = FakeCrawler({site["url"]: results})

    records, stats = _run(_crawl_one(crawler, site, "markets"))

    assert records == []
    assert stats["pages_kept"] == 0
    assert stats["pages_dropped"] == 0


def test_crawl_one_captures_arun_exception_in_stats() -> None:
    site = _site()
    crawler = FakeCrawler(raise_on={site["url"]: RuntimeError("network")})

    records, stats = _run(_crawl_one(crawler, site, "markets"))

    assert records == []
    assert stats["error"] is not None
    assert "network" in stats["error"]


def test_crawl_all_aggregates_per_site_records_and_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sites = [
        {"url": "https://cnbc.com/", "url_patterns": ["*"], "include_external": False},
        {
            "url": "https://finviz.com/",
            "url_patterns": ["*"],
            "include_external": False,
        },
    ]
    monkeypatch.setattr(crawler_module, "SITES", fake_sites)

    class FakeBrowserCrawler:
        def __init__(self, config: Any) -> None:
            self.config = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> None:
            return None

    monkeypatch.setattr(crawler_module, "AsyncWebCrawler", FakeBrowserCrawler)

    async def fake_crawl_one(crawler: Any, site: dict, query: str):
        if site["url"].startswith("https://cnbc"):
            return (
                [{"url": "https://cnbc.com/a", "_source_domain": "cnbc.com"}],
                {
                    "site": "cnbc.com",
                    "pages_kept": 1,
                    "pages_dropped": 0,
                    "error": None,
                },
            )
        return (
            [],
            {"site": "finviz.com", "pages_kept": 0, "pages_dropped": 0, "error": None},
        )

    monkeypatch.setattr(crawler_module, "_crawl_one", fake_crawl_one)

    records, site_stats = _run(crawl_all("markets"))

    assert len(records) == 1
    assert records[0]["url"] == "https://cnbc.com/a"
    assert {s["site"] for s in site_stats} == {"cnbc.com", "finviz.com"}


def test_crawl_all_synthesizes_error_stats_when_gather_returns_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sites = [
        {"url": "https://cnbc.com/", "url_patterns": ["*"], "include_external": False},
    ]
    monkeypatch.setattr(crawler_module, "SITES", fake_sites)

    class FakeBrowserCrawler:
        def __init__(self, config: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> None:
            return None

    monkeypatch.setattr(crawler_module, "AsyncWebCrawler", FakeBrowserCrawler)

    async def boom(crawler: Any, site: dict, query: str):
        raise RuntimeError("crashed inside _crawl_one")

    monkeypatch.setattr(crawler_module, "_crawl_one", boom)

    records, site_stats = _run(crawl_all("markets"))

    assert records == []
    assert len(site_stats) == 1
    assert site_stats[0]["error"] is not None
    assert "crashed" in site_stats[0]["error"]
    assert site_stats[0]["pages_kept"] == 0
