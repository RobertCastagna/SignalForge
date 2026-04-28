"""Quarantine writer for failed Bronze rows."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from amdc_lake.bronze import BRONZE_SCHEMA
from amdc_lake.paths import bronze_scrapes_quarantine_path, ensure_layers

QUARANTINE_SCHEMA: dict[str, pl.DataType] = {
    **BRONZE_SCHEMA,
    "_failure_reasons": pl.List(pl.Utf8),
    "_quality_run_id": pl.Utf8,
}


def write_quarantine(failures: pl.DataFrame, lake_dir: Path) -> Path | None:
    if failures.is_empty():
        return None
    ensure_layers(lake_dir)
    target = bronze_scrapes_quarantine_path(lake_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    _align(failures, QUARANTINE_SCHEMA).write_delta(str(target), mode="append")
    return target


def _align(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    for name, dtype in schema.items():
        if name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(name))
    casts = [
        pl.col(name).cast(dtype, strict=False).alias(name)
        for name, dtype in schema.items()
    ]
    return df.with_columns(casts).select(list(schema))
