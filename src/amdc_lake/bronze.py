"""Bronze Delta ingestion for normalized scrape parquet files."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import polars as pl

from amdc_lake.ids import sha256_id
from amdc_lake.observability import record_run
from amdc_lake.paths import bronze_scrapes_path, ensure_layers

log = logging.getLogger(__name__)

BRONZE_SCHEMA: dict[str, pl.DataType] = {
    "bronze_id": pl.Utf8,
    "title": pl.Utf8,
    "date_published": pl.Utf8,
    "text": pl.Utf8,
    "source_url": pl.Utf8,
    "source_domain": pl.Utf8,
    "relevance_score": pl.Float64,
    "query": pl.Utf8,
    "crawled_at": pl.Utf8,
    "source_file": pl.Utf8,
    "ingested_at": pl.Utf8,
}

WriteMode = Literal["append", "overwrite"]


def _empty_bronze() -> pl.DataFrame:
    return pl.DataFrame(schema=BRONZE_SCHEMA)


def _normalize_frame(
    df: pl.DataFrame, *, source_file: Path, ingested_at: str
) -> pl.DataFrame:
    for name, dtype in BRONZE_SCHEMA.items():
        if name not in df.columns and name not in {
            "bronze_id",
            "source_file",
            "ingested_at",
        }:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(name))

    df = df.with_columns(
        pl.lit(str(source_file)).alias("source_file"),
        pl.lit(ingested_at).alias("ingested_at"),
        pl.col("title").cast(pl.Utf8, strict=False),
        pl.col("date_published").cast(pl.Utf8, strict=False),
        pl.col("text").cast(pl.Utf8, strict=False),
        pl.col("source_url").cast(pl.Utf8, strict=False),
        pl.col("source_domain").cast(pl.Utf8, strict=False),
        pl.col("relevance_score").cast(pl.Float64, strict=False),
        pl.col("query").cast(pl.Utf8, strict=False),
        pl.col("crawled_at").cast(pl.Utf8, strict=False),
    )
    df = df.with_columns(
        pl.struct(["source_url", "query", "crawled_at", "title", "text"])
        .map_elements(
            lambda row: sha256_id(
                row["source_url"],
                row["query"],
                row["crawled_at"],
                row["title"],
                row["text"],
                prefix="bronze",
            ),
            return_dtype=pl.Utf8,
        )
        .alias("bronze_id")
    )
    return df.select(list(BRONZE_SCHEMA))


def load_parquet_dir(input_dir: Path) -> pl.DataFrame:
    files = sorted(input_dir.glob("market_data_*.parquet"))
    if not files:
        return _empty_bronze()

    ingested_at = datetime.now(timezone.utc).isoformat()
    frames = [
        _normalize_frame(
            pl.read_parquet(path), source_file=path, ingested_at=ingested_at
        )
        for path in files
    ]
    return pl.concat(frames, how="vertical_relaxed").unique(
        subset=["bronze_id"], keep="first"
    )


def write_bronze(
    df: pl.DataFrame, lake_dir: Path, *, mode: WriteMode = "overwrite"
) -> Path:
    ensure_layers(lake_dir)
    target = bronze_scrapes_path(lake_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.select(list(BRONZE_SCHEMA)).write_delta(str(target), mode=mode)
    return target


def backfill_parquet(
    input_dir: Path,
    lake_dir: Path,
    *,
    mode: WriteMode = "overwrite",
    validate: bool = True,
) -> tuple[Path, int]:
    source_files = len(sorted(input_dir.glob("market_data_*.parquet")))
    df = load_parquet_dir(input_dir)
    rows_loaded = df.height
    log.info(
        "bronze: loaded %d rows from %d parquet file(s)", rows_loaded, source_files
    )

    with record_run("bronze", lake_dir, rows_in=rows_loaded, logger=log) as handle:
        rows_quarantined = 0
        quality_run_id: str | None = None
        if validate and not df.is_empty():
            from amdc_lake.quality.metrics import append_run
            from amdc_lake.quality.quarantine import write_quarantine
            from amdc_lake.quality.runner import run_bronze_checks

            result = run_bronze_checks(df, lake_dir)
            write_quarantine(result.failures, lake_dir)
            append_run(result, lake_dir)
            quality_run_id = result.run_id
            if not result.failures.is_empty():
                rows_quarantined = result.failures.height
                df = df.filter(
                    ~pl.col("bronze_id").is_in(result.failures.get_column("bronze_id"))
                )
        target = write_bronze(df, lake_dir, mode=mode)
        rows_written = df.height
        if rows_quarantined > 0:
            handle.set_status("partial")
        handle.set_rows_out(rows_written)
        handle.update_details(
            source_files=source_files,
            rows_quarantined=rows_quarantined,
            validate=validate,
            mode=mode,
            quality_run_id=quality_run_id,
        )
    return target, rows_written
