"""Bronze quality check orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandera.polars as pa
import polars as pl
from deltalake import DeltaTable

from amdc_lake.ids import sha256_id
from amdc_lake.paths import bronze_scrapes_path
from amdc_lake.quality.checks import compute_run_drift
from amdc_lake.quality.schemas import bronze_schema


@dataclass
class QualityResult:
    run_id: str
    layer: str
    started_at: str
    finished_at: str
    rows_in: int
    rows_passed: int
    rows_failed: int
    failures: pl.DataFrame
    check_summary: list[dict] = field(default_factory=list)
    drift_report: list[dict] = field(default_factory=list)
    status: str = "pass"


def _read_existing_bronze(lake_dir: Path) -> pl.DataFrame | None:
    table_path = bronze_scrapes_path(lake_dir)
    if not table_path.exists():
        return None
    try:
        table = DeltaTable(str(table_path))
    except Exception:
        return None
    return pl.from_arrow(table.to_pyarrow_table())


def _empty_failure_cases() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "failure_case": pl.Utf8,
            "schema_context": pl.Utf8,
            "column": pl.Utf8,
            "check": pl.Utf8,
            "check_number": pl.Int32,
            "index": pl.Int32,
        }
    )


def _build_failures(df: pl.DataFrame, failure_cases: pl.DataFrame, run_id: str) -> pl.DataFrame:
    """Join failure_cases (one row per failure) back onto df (one row per failed input row)."""
    row_failures = failure_cases.filter(pl.col("index").is_not_null())
    if row_failures.is_empty():
        return df.head(0).with_columns(
            pl.Series("_failure_reasons", [], dtype=pl.List(pl.Utf8)),
            pl.Series("_quality_run_id", [], dtype=pl.Utf8),
        )

    grouped = (
        row_failures.with_columns(
            pl.concat_str([pl.col("column"), pl.lit(": "), pl.col("check")]).alias("_reason")
        )
        .group_by("index")
        .agg(pl.col("_reason").alias("_failure_reasons"))
    )

    df_idx = df.with_row_index(name="_row_index").with_columns(
        pl.col("_row_index").cast(pl.Int32)
    )
    joined = df_idx.join(grouped, left_on="_row_index", right_on="index", how="inner").drop(
        "_row_index"
    )
    return joined.with_columns(pl.lit(run_id).alias("_quality_run_id"))


def _summarize(failure_cases: pl.DataFrame) -> list[dict]:
    if failure_cases.is_empty():
        return []
    return (
        failure_cases.group_by(["column", "check"])
        .agg(pl.len().alias("failed"))
        .sort("failed", descending=True)
        .to_dicts()
    )


def _status(rows_failed: int, drift_report: list[dict]) -> str:
    if rows_failed > 0:
        return "fail"
    if drift_report:
        return "warn"
    return "pass"


def run_bronze_checks(df: pl.DataFrame, lake_dir: Path) -> QualityResult:
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = sha256_id("bronze", started_at, prefix="qrun")

    schema = bronze_schema()
    try:
        schema.validate(df, lazy=True)
        failure_cases = _empty_failure_cases()
    except pa.errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases

    failures = _build_failures(df, failure_cases, run_id)
    check_summary = _summarize(failure_cases)
    drift_report = compute_run_drift(df, _read_existing_bronze(lake_dir))

    rows_in = df.height
    rows_failed = failures.height
    rows_passed = rows_in - rows_failed
    finished_at = datetime.now(timezone.utc).isoformat()

    return QualityResult(
        run_id=run_id,
        layer="bronze",
        started_at=started_at,
        finished_at=finished_at,
        rows_in=rows_in,
        rows_passed=rows_passed,
        rows_failed=rows_failed,
        failures=failures,
        check_summary=check_summary,
        drift_report=drift_report,
        status=_status(rows_failed, drift_report),
    )
