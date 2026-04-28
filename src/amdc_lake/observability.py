"""Append-only pipeline run records mirroring the quality/runs pattern.

`record_run` is a sync context manager that times the wrapped block, captures
status/rows/details, and appends one row to `{lake_dir}/_pipeline/runs`. It
always writes the row, even on exception, then re-raises.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import polars as pl

from amdc_lake.ids import sha256_id
from amdc_lake.paths import ensure_layers, pipeline_runs_path

PIPELINE_RUNS_SCHEMA: dict[str, pl.DataType] = {
    "run_id": pl.Utf8,
    "stage": pl.Utf8,
    "started_at": pl.Utf8,
    "finished_at": pl.Utf8,
    "duration_ms": pl.Int64,
    "status": pl.Utf8,
    "rows_in": pl.Int64,
    "rows_out": pl.Int64,
    "query": pl.Utf8,
    "details": pl.Utf8,
    "error": pl.Utf8,
    "lake_dir": pl.Utf8,
}

_VALID_STATUSES = {"success", "partial", "fail"}


@dataclass
class PipelineRun:
    run_id: str
    stage: str
    started_at: str
    finished_at: str
    duration_ms: int
    status: str
    rows_in: int | None
    rows_out: int | None
    query: str | None
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class RunHandle:
    """Mutable view passed to the `with` block to set rows_out/status/details."""

    stage: str
    rows_in: int | None
    rows_out: int | None = None
    status: str = "success"
    details: dict[str, Any] = field(default_factory=dict)

    def set_rows_out(self, n: int) -> None:
        self.rows_out = n

    def set_status(self, status: str) -> None:
        if status not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {_VALID_STATUSES}, got {status!r}")
        self.status = status

    def update_details(self, **kwargs: Any) -> None:
        self.details.update(kwargs)


def append_pipeline_run(record: PipelineRun, lake_dir: Path) -> Path:
    ensure_layers(lake_dir)
    target = pipeline_runs_path(lake_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = pl.DataFrame(
        {
            "run_id": [record.run_id],
            "stage": [record.stage],
            "started_at": [record.started_at],
            "finished_at": [record.finished_at],
            "duration_ms": [record.duration_ms],
            "status": [record.status],
            "rows_in": [record.rows_in],
            "rows_out": [record.rows_out],
            "query": [record.query],
            "details": [json.dumps(record.details, default=str)],
            "error": [record.error],
            "lake_dir": [str(lake_dir)],
        },
        schema=PIPELINE_RUNS_SCHEMA,
    )
    row.write_delta(str(target), mode="append")
    return target


@contextmanager
def record_run(
    stage: str,
    lake_dir: Path,
    *,
    rows_in: int | None = None,
    query: str | None = None,
    logger: logging.Logger | None = None,
) -> Iterator[RunHandle]:
    log = logger or logging.getLogger("amdc_lake.observability")
    started_at_dt = datetime.now(timezone.utc)
    started_at = started_at_dt.isoformat()
    started_perf = perf_counter()
    handle = RunHandle(stage=stage, rows_in=rows_in)
    log.info(
        "pipeline stage=%s started lake_dir=%s rows_in=%s",
        stage,
        lake_dir,
        rows_in,
    )

    error: str | None = None
    raised: BaseException | None = None
    try:
        yield handle
    except BaseException as exc:
        handle.status = "fail"
        error = repr(exc)[:500]
        raised = exc
    finally:
        finished_at = datetime.now(timezone.utc).isoformat()
        duration_ms = int((perf_counter() - started_perf) * 1000)
        record = PipelineRun(
            run_id=sha256_id(stage, started_at, prefix="prun"),
            stage=stage,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            status=handle.status,
            rows_in=handle.rows_in,
            rows_out=handle.rows_out,
            query=query,
            details=handle.details,
            error=error,
        )
        try:
            append_pipeline_run(record, lake_dir)
        except Exception:
            log.exception("failed to append pipeline run for stage=%s", stage)

        if raised is not None:
            log.error(
                "pipeline stage=%s failed duration_ms=%d error=%s",
                stage,
                duration_ms,
                error,
                exc_info=raised,
            )
        else:
            log.info(
                "pipeline stage=%s finished status=%s duration_ms=%d rows_in=%s rows_out=%s",
                stage,
                handle.status,
                duration_ms,
                handle.rows_in,
                handle.rows_out,
            )

    if raised is not None:
        raise raised
