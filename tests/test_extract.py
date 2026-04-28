from __future__ import annotations

from amdc.config import TEXT_CHAR_CAP
from amdc.extract import (
    _extract_body,
    _extract_date,
    _extract_score,
    _extract_title,
    _extract_url,
    _first,
    normalize,
)


def test_first_returns_first_truthy_value() -> None:
    assert _first(None, "", "second", "third") == "second"
    assert _first(None, "", 0) is None
    assert _first("only") == "only"


def test_extract_title_prefers_explicit_then_meta_then_first_body_line() -> None:
    page = {"title": "Headline"}
    assert _extract_title(page, "irrelevant body") == "Headline"

    page = {"metadata": {"og:title": "OG Title"}}
    assert _extract_title(page, "body") == "OG Title"

    page = {"metadata": {"title": "Meta Title"}}
    assert _extract_title(page, "body") == "Meta Title"

    page = {}
    body = "\n   \n# First real line\nSecond line\n"
    assert _extract_title(page, body) == "First real line"

    assert _extract_title({}, "") is None


def test_extract_title_truncates_long_first_body_line_to_200() -> None:
    long_line = "x" * 500
    assert _extract_title({}, long_line) == "x" * 200


def test_extract_date_prefers_explicit_then_meta_then_regex() -> None:
    page = {"date_published": "2026-04-28"}
    assert _extract_date(page, "") == "2026-04-28"

    page = {"metadata": {"article:published_time": "2026-01-15T00:00:00Z"}}
    assert _extract_date(page, "") == "2026-01-15T00:00:00Z"

    page = {"metadata": {"date": "May 3, 2026"}}
    assert _extract_date(page, "") == "May 3, 2026"


def test_extract_date_finds_iso_date_in_body() -> None:
    assert _extract_date({}, "Published 2026-04-28 by author") == "2026-04-28"


def test_extract_date_finds_named_month_date_in_body() -> None:
    assert _extract_date({}, "Posted Jan 5, 2026 in markets") == "Jan 5, 2026"


def test_extract_date_finds_slash_date_in_body() -> None:
    assert _extract_date({}, "see 4/28/2026 update") == "4/28/2026"


def test_extract_date_returns_none_when_no_match() -> None:
    assert _extract_date({}, "no dates anywhere here") is None


def test_extract_body_prefers_fit_then_raw_then_others_and_caps_length() -> None:
    page = {"fit_markdown": "fit", "raw_markdown": "raw", "markdown": "md"}
    assert _extract_body(page) == "fit"

    page = {"raw_markdown": "raw"}
    assert _extract_body(page) == "raw"

    page = {"content": "content fallback"}
    assert _extract_body(page) == "content fallback"

    big = "x" * (TEXT_CHAR_CAP + 100)
    assert len(_extract_body({"raw_markdown": big})) == TEXT_CHAR_CAP

    assert _extract_body({}) == ""


def test_extract_url_prefers_url_then_source_url_then_start_url() -> None:
    assert _extract_url({"url": "u", "source_url": "s"}) == "u"
    assert _extract_url({"source_url": "s", "_source_start_url": "start"}) == "s"
    assert _extract_url({"_source_start_url": "start"}) == "start"
    assert _extract_url({}) is None


def test_extract_score_handles_floats_strings_and_garbage() -> None:
    assert _extract_score({"relevance_score": 1.5}) == 1.5
    assert _extract_score({"score": "2.0"}) == 2.0
    assert _extract_score({"score": "not a number"}) is None
    assert _extract_score({}) is None


def test_normalize_builds_full_record_with_query_and_crawled_at() -> None:
    pages = [
        {
            "url": "https://cnbc.com/a",
            "fit_markdown": "Headline article body",
            "raw_markdown": "raw fallback",
            "metadata": {"og:title": "OG Headline"},
            "_source_domain": "cnbc.com",
            "score": 1.2,
        }
    ]
    out = normalize(pages, query="markets")

    assert len(out) == 1
    rec = out[0]
    assert rec["title"] == "OG Headline"
    assert rec["text"] == "Headline article body"
    assert rec["source_url"] == "https://cnbc.com/a"
    assert rec["source_domain"] == "cnbc.com"
    assert rec["relevance_score"] == 1.2
    assert rec["query"] == "markets"
    assert rec["crawled_at"].endswith("+00:00")
    assert rec["date_published"] is None


def test_normalize_handles_empty_input() -> None:
    assert normalize([], query="anything") == []
