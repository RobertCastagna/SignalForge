from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from amdc.store import read_parquet_demo, write_parquet


def _sample_record() -> dict:
    return {
        "title": "Headline",
        "date_published": "2026-04-28",
        "text": "body",
        "source_url": "https://cnbc.com/a",
        "source_domain": "cnbc.com",
        "relevance_score": 1.0,
        "query": "markets",
        "crawled_at": "2026-04-28T00:00:00+00:00",
    }


def test_write_parquet_creates_data_dir_and_utc_filename(tmp_path: Path) -> None:
    data_dir = tmp_path / "out"
    path = write_parquet([_sample_record()], data_dir)

    assert path.parent == data_dir
    assert data_dir.is_dir()
    assert re.fullmatch(r"market_data_\d{8}T\d{6}Z\.parquet", path.name)


def test_write_parquet_round_trips_records_via_read_helper(tmp_path: Path) -> None:
    rec = _sample_record()
    path = write_parquet([rec], tmp_path)
    df = read_parquet_demo(path)

    assert df.height == 1
    assert df.select("title").item() == "Headline"
    assert df.select("source_domain").item() == "cnbc.com"
    assert df.select("relevance_score").item() == 1.0


def test_write_parquet_writes_typed_empty_frame_when_no_records(tmp_path: Path) -> None:
    path = write_parquet([], tmp_path)
    df = pl.read_parquet(path)

    assert df.is_empty()
    expected = {
        "title",
        "date_published",
        "text",
        "source_url",
        "source_domain",
        "relevance_score",
        "query",
        "crawled_at",
    }
    assert set(df.columns) == expected
    assert df.schema["relevance_score"] == pl.Float64
