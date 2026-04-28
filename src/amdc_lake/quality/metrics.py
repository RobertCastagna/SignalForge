"""Append-only metrics table for quality run history."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from amdc_lake.paths import ensure_layers, quality_runs_path

if TYPE_CHECKING:
    from amdc_lake.quality.runner import QualityResult

RUNS_SCHEMA: dict[str, pl.DataType] = {
    "run_id": pl.Utf8,
    "layer": pl.Utf8,
    "started_at": pl.Utf8,
    "finished_at": pl.Utf8,
    "rows_in": pl.Int64,
    "rows_passed": pl.Int64,
    "rows_failed": pl.Int64,
    "status": pl.Utf8,
    "check_summary": pl.Utf8,
    "drift_report": pl.Utf8,
    "lake_dir": pl.Utf8,
}


def append_run(result: "QualityResult", lake_dir: Path) -> Path:
    ensure_layers(lake_dir)
    target = quality_runs_path(lake_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = pl.DataFrame(
        {
            "run_id": [result.run_id],
            "layer": [result.layer],
            "started_at": [result.started_at],
            "finished_at": [result.finished_at],
            "rows_in": [result.rows_in],
            "rows_passed": [result.rows_passed],
            "rows_failed": [result.rows_failed],
            "status": [result.status],
            "check_summary": [json.dumps(result.check_summary)],
            "drift_report": [json.dumps(result.drift_report)],
            "lake_dir": [str(lake_dir)],
        },
        schema=RUNS_SCHEMA,
    )
    row.write_delta(str(target), mode="append")
    return target
