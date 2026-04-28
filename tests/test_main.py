from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from deltalake import DeltaTable
from typer.testing import CliRunner

from amdc import main as main_module
from amdc.main import app
from amdc_lake.paths import pipeline_runs_path


def _patch_crawl_all(
    monkeypatch: pytest.MonkeyPatch,
    raw: list[dict],
    site_stats: list[dict],
) -> None:
    async def fake_crawl_all(query: str):
        return raw, site_stats

    monkeypatch.setattr(main_module, "crawl_all", fake_crawl_all)


def _kept_page(url: str = "https://cnbc.com/a") -> dict:
    return {
        "url": url,
        "fit_markdown": "Headline body content",
        "raw_markdown": "x" * 1000,
        "metadata": {"og:title": "Headline"},
        "score": 1.0,
        "_source_start_url": "https://cnbc.com/",
        "_source_domain": "cnbc.com",
    }


def test_amdc_run_writes_parquet_and_no_pipeline_run_without_lake_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    data_dir = tmp_path / "data"
    _patch_crawl_all(
        monkeypatch,
        raw=[_kept_page()],
        site_stats=[
            {"site": "cnbc.com", "pages_kept": 1, "pages_dropped": 0, "error": None}
        ],
    )

    result = runner.invoke(app, ["markets", "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    parquets = list(data_dir.glob("market_data_*.parquet"))
    assert len(parquets) == 1
    df = pl.read_parquet(parquets[0])
    assert df.height == 1
    assert df.select("query").item() == "markets"


def test_amdc_run_with_lake_dir_appends_success_pipeline_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    data_dir = tmp_path / "data"
    lake_dir = tmp_path / "lake"
    _patch_crawl_all(
        monkeypatch,
        raw=[_kept_page()],
        site_stats=[
            {"site": "cnbc.com", "pages_kept": 1, "pages_dropped": 0, "error": None}
        ],
    )

    result = runner.invoke(
        app,
        ["markets", "--data-dir", str(data_dir), "--lake-dir", str(lake_dir)],
    )

    assert result.exit_code == 0, result.output
    runs = pl.from_arrow(
        DeltaTable(str(pipeline_runs_path(lake_dir))).to_pyarrow_table()
    )
    crawl_runs = runs.filter(pl.col("stage") == "crawl")
    assert crawl_runs.height == 1
    row = crawl_runs.row(0, named=True)
    assert row["status"] == "success"
    assert row["rows_out"] == 1
    assert row["query"] == "markets"


def test_amdc_run_records_partial_when_one_site_errored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    data_dir = tmp_path / "data"
    lake_dir = tmp_path / "lake"
    _patch_crawl_all(
        monkeypatch,
        raw=[_kept_page()],
        site_stats=[
            {"site": "cnbc.com", "pages_kept": 1, "pages_dropped": 0, "error": None},
            {
                "site": "finviz.com",
                "pages_kept": 0,
                "pages_dropped": 0,
                "error": "RuntimeError('boom')",
            },
        ],
    )

    result = runner.invoke(
        app,
        ["markets", "--data-dir", str(data_dir), "--lake-dir", str(lake_dir)],
    )

    assert result.exit_code == 0, result.output
    runs = pl.from_arrow(
        DeltaTable(str(pipeline_runs_path(lake_dir))).to_pyarrow_table()
    )
    crawl_row = runs.filter(pl.col("stage") == "crawl").row(0, named=True)
    assert crawl_row["status"] == "partial"


def test_amdc_run_records_fail_when_zero_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    data_dir = tmp_path / "data"
    lake_dir = tmp_path / "lake"
    _patch_crawl_all(
        monkeypatch,
        raw=[],
        site_stats=[
            {
                "site": "cnbc.com",
                "pages_kept": 0,
                "pages_dropped": 0,
                "error": "RuntimeError('x')",
            }
        ],
    )

    result = runner.invoke(
        app,
        ["markets", "--data-dir", str(data_dir), "--lake-dir", str(lake_dir)],
    )

    assert result.exit_code == 0, result.output
    runs = pl.from_arrow(
        DeltaTable(str(pipeline_runs_path(lake_dir))).to_pyarrow_table()
    )
    crawl_row = runs.filter(pl.col("stage") == "crawl").row(0, named=True)
    assert crawl_row["status"] == "fail"
    assert crawl_row["rows_out"] == 0
