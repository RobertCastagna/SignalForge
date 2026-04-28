"""Normalize raw crawl4ai page payloads into flat records for polars."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from amdc.config import TEXT_CHAR_CAP

_DATE_PATTERNS = [
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    re.compile(
        r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b"
    ),
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
]


def _first(*vals):
    for v in vals:
        if v:
            return v
    return None


def _extract_title(page: dict, body: str) -> str | None:
    meta = page.get("metadata") or {}
    title = _first(
        page.get("title"),
        meta.get("og:title"),
        meta.get("title"),
    )
    if title:
        return str(title).strip()
    for line in body.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:200]
    return None


def _extract_date(page: dict, body: str) -> str | None:
    meta = page.get("metadata") or {}
    explicit = _first(
        page.get("date_published"),
        meta.get("article:published_time"),
        meta.get("og:article:published_time"),
        meta.get("date"),
    )
    if explicit:
        return str(explicit)
    for pat in _DATE_PATTERNS:
        m = pat.search(body)
        if m:
            return m.group(1)
    return None


def _extract_body(page: dict) -> str:
    # Prefer the BM25-filtered fit_markdown. Fall back to raw_markdown when fit
    # is empty: the crawler gate now admits long articles whose BM25 trim
    # collapsed fit to nothing, so without this fallback those rows would have
    # empty text. Raw includes some site nav/footer chrome — downstream
    # chunking deals with it.
    body = _first(
        page.get("fit_markdown"),
        page.get("raw_markdown"),
        page.get("markdown"),
        page.get("content"),
        page.get("text"),
    )
    return (str(body)[:TEXT_CHAR_CAP]) if body else ""


def _extract_url(page: dict) -> str | None:
    return _first(
        page.get("url"), page.get("source_url"), page.get("_source_start_url")
    )


def _extract_score(page: dict) -> float | None:
    score = _first(page.get("relevance_score"), page.get("score"))
    try:
        return float(score) if score is not None else None
    except (TypeError, ValueError):
        return None


def normalize(raw_pages: list[dict], *, query: str) -> list[dict]:
    crawled_at = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
    for page in raw_pages:
        body = _extract_body(page)
        records.append(
            {
                "title": _extract_title(page, body),
                "date_published": _extract_date(page, body),
                "text": body,
                "source_url": _extract_url(page),
                "source_domain": page.get("_source_domain"),
                "relevance_score": _extract_score(page),
                "query": query,
                "crawled_at": crawled_at,
            }
        )
    return records
