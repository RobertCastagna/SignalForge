"""Pandera schema for the Bronze scrapes table."""

from __future__ import annotations

from urllib.parse import urlparse

import pandera.polars as pa
import polars as pl
from pandera.polars import Column, DataFrameSchema

from amdc.config import BM25_THRESHOLD, SITES, TEXT_CHAR_CAP
from amdc_lake.quality.checks import text_is_not_error_page, text_passes_junk


def _allowed_domains() -> list[str]:
    """Derive allowed root domains from amdc.config.SITES."""
    return sorted({urlparse(site["url"]).netloc.removeprefix("www.") for site in SITES})


def bronze_schema() -> DataFrameSchema:
    return DataFrameSchema(
        {
            "bronze_id": Column(
                pl.Utf8,
                checks=[pa.Check.str_matches(r"^bronze_[0-9a-f]{64}$")],
                nullable=False,
                unique=True,
            ),
            "source_url": Column(
                pl.Utf8,
                checks=[pa.Check.str_matches(r"^https?://")],
                nullable=False,
            ),
            "source_domain": Column(
                pl.Utf8,
                checks=[pa.Check.isin(_allowed_domains())],
                nullable=False,
            ),
            "title": Column(
                pl.Utf8,
                checks=[pa.Check.str_length(max_value=200)],
                nullable=True,
            ),
            "text": Column(
                pl.Utf8,
                checks=[
                    pa.Check.str_length(min_value=50, max_value=TEXT_CHAR_CAP),
                    pa.Check(
                        text_passes_junk,
                        element_wise=True,
                        name="url_junk_ratio_le_0.5",
                        error="text is more than 50% URL-structured (junk)",
                    ),
                    pa.Check(
                        text_is_not_error_page,
                        element_wise=True,
                        name="text_not_error_page",
                        error="text matches error/paywall sentinel",
                    ),
                ],
                nullable=False,
            ),
            "relevance_score": Column(
                pl.Float64,
                checks=[pa.Check.ge(BM25_THRESHOLD)],
                nullable=True,
            ),
            "query": Column(
                pl.Utf8,
                checks=[pa.Check.str_length(min_value=1)],
                nullable=False,
            ),
            "crawled_at": Column(pl.Utf8, nullable=False),
            "ingested_at": Column(pl.Utf8, nullable=False),
            "date_published": Column(pl.Utf8, nullable=True),
            "source_file": Column(pl.Utf8, nullable=False),
        },
        strict=False,
    )
