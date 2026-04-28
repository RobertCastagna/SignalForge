"""Persist normalized records to parquet and provide a downstream read helper."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl


def write_parquet(records: list[dict], data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = data_dir / f"market_data_{ts}.parquet"
    df = pl.DataFrame(records) if records else pl.DataFrame(
        schema={
            "title": pl.Utf8,
            "date_published": pl.Utf8,
            "text": pl.Utf8,
            "source_url": pl.Utf8,
            "source_domain": pl.Utf8,
            "relevance_score": pl.Float64,
            "query": pl.Utf8,
            "crawled_at": pl.Utf8,
        }
    )
    df.write_parquet(path)
    return path


def read_parquet_demo(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)
