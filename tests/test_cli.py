from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from deltalake import DeltaTable
from typer.testing import CliRunner

from amdc_lake.bronze import BRONZE_SCHEMA, write_bronze
from amdc_lake.cli import app as lake_app
from amdc_lake.constants import EMBEDDING_DIM
from amdc_lake.paths import (
    bronze_scrapes_path,
    pipeline_runs_path,
    silver_chunks_path,
    silver_pages_path,
)


class _FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(range(1, len(text.split()) + 1))

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok{token_id}" for token_id in token_ids)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 0


class _FakeEmbedder:
    tokenizer = _FakeTokenizer()
    max_length = 512

    def embed(self, texts: list[str], *, batch_size: int = 8) -> list[list[float]]:
        return [[float(i)] + [0.0] * (EMBEDDING_DIM - 1) for i, _ in enumerate(texts)]


def _bronze_row() -> dict:
    return {
        "title": "Fed decision",
        "date_published": "2026-04-28",
        "text": "Markets watch the Federal Reserve decision today and tomorrow.",
        "source_url": "https://cnbc.com/fed",
        "source_domain": "cnbc.com",
        "relevance_score": 1.0,
        "query": "Fed rate decision",
        "crawled_at": "2026-04-28T00:00:00+00:00",
    }


def test_cli_init_creates_all_lake_layers(tmp_path: Path) -> None:
    runner = CliRunner()
    lake_dir = tmp_path / "lake"

    result = runner.invoke(lake_app, ["init", "--lake-dir", str(lake_dir)])

    assert result.exit_code == 0, result.output
    for layer in ("bronze", "silver", "gold", "_quality", "_pipeline"):
        assert (lake_dir / layer).is_dir()


def test_cli_bronze_backfill_writes_rows_and_pipeline_run(tmp_path: Path) -> None:
    runner = CliRunner()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    lake_dir = tmp_path / "lake"
    pl.DataFrame([_bronze_row()]).write_parquet(input_dir / "market_data_x.parquet")

    result = runner.invoke(
        lake_app,
        [
            "bronze-backfill",
            "--input-dir",
            str(input_dir),
            "--lake-dir",
            str(lake_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "bronze rows" in result.output

    bronze = pl.from_arrow(
        DeltaTable(str(bronze_scrapes_path(lake_dir))).to_pyarrow_table()
    )
    assert bronze.height == 1

    runs = pl.from_arrow(
        DeltaTable(str(pipeline_runs_path(lake_dir))).to_pyarrow_table()
    )
    assert runs.filter(pl.col("stage") == "bronze").height == 1


def test_cli_bronze_backfill_no_validate_skips_quality_run(tmp_path: Path) -> None:
    runner = CliRunner()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    lake_dir = tmp_path / "lake"
    pl.DataFrame([_bronze_row()]).write_parquet(input_dir / "market_data_x.parquet")

    result = runner.invoke(
        lake_app,
        [
            "bronze-backfill",
            "--input-dir",
            str(input_dir),
            "--lake-dir",
            str(lake_dir),
            "--no-validate",
        ],
    )

    assert result.exit_code == 0, result.output
    quality_path = lake_dir / "_quality" / "runs"
    assert not quality_path.exists() or not any(quality_path.iterdir())


def test_cli_silver_build_writes_pages_and_chunks_with_mocked_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    lake_dir = tmp_path / "lake"

    bronze = pl.DataFrame(
        [
            {
                "bronze_id": "bronze_a",
                "title": "Useful",
                "date_published": "2026-04-28",
                "text": "alpha beta gamma delta epsilon",
                "source_url": "https://cnbc.com/a",
                "source_domain": "cnbc.com",
                "relevance_score": 1.0,
                "query": "markets",
                "crawled_at": "2026-04-28T00:00:00+00:00",
                "source_file": "f.parquet",
                "ingested_at": "2026-04-28T00:00:00+00:00",
            }
        ],
        schema=BRONZE_SCHEMA,
    )
    write_bronze(bronze, lake_dir)

    monkeypatch.setattr(
        "amdc_lake.embedder.BgeM3Embedder", lambda device=None: _FakeEmbedder()
    )

    result = runner.invoke(
        lake_app,
        [
            "silver-build",
            "--lake-dir",
            str(lake_dir),
            "--batch-size",
            "2",
            "--chunk-tokens",
            "3",
            "--chunk-overlap",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert silver_pages_path(lake_dir).is_dir()
    assert silver_chunks_path(lake_dir).is_dir()

    runs = pl.from_arrow(
        DeltaTable(str(pipeline_runs_path(lake_dir))).to_pyarrow_table()
    )
    stages = set(runs.get_column("stage").to_list())
    assert "silver_pages" in stages
    assert "silver_chunks" in stages


def test_cli_quality_check_exits_one_when_no_bronze_table(tmp_path: Path) -> None:
    runner = CliRunner()
    lake_dir = tmp_path / "lake"

    result = runner.invoke(
        lake_app,
        ["quality-check", "--lake-dir", str(lake_dir)],
    )

    assert result.exit_code == 1
    assert "No Bronze table" in result.output


def test_cli_quality_check_passes_on_clean_bronze_table(tmp_path: Path) -> None:
    runner = CliRunner()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    lake_dir = tmp_path / "lake"
    pl.DataFrame([_bronze_row()]).write_parquet(input_dir / "market_data_x.parquet")
    runner.invoke(
        lake_app,
        ["bronze-backfill", "--input-dir", str(input_dir), "--lake-dir", str(lake_dir)],
    )

    result = runner.invoke(
        lake_app,
        ["quality-check", "--lake-dir", str(lake_dir)],
    )

    assert result.exit_code == 0, result.output
    assert "status" in result.output
