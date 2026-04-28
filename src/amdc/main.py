"""Typer CLI entrypoint for the Adaptive Market Data Crawler."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer

from amdc.crawler import crawl_all
from amdc.extract import normalize
from amdc.store import write_parquet

app = typer.Typer(add_completion=False, help="Adaptive Market Data Crawler")

log = logging.getLogger(__name__)


@app.command()
def run(
    query: str = typer.Argument(..., help="Search query"),
    data_dir: Path = typer.Option(
        Path("data"),
        "--data-dir",
        help="Output directory for parquet. Resolves to ./data on host or /app/data in Docker (WORKDIR=/app).",
    ),
    lake_dir: Path = typer.Option(
        None,
        "--lake-dir",
        help="If set, append a crawl row to {lake_dir}/_pipeline/runs.",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raw, site_stats = asyncio.run(crawl_all(query))
    records = normalize(raw, query=query)
    path = write_parquet(records, data_dir)

    n_errors = sum(1 for s in site_stats if s.get("error"))
    log.info(
        "crawl summary: query=%r sites=%d errored=%d records=%d output=%s",
        query,
        len(site_stats),
        n_errors,
        len(records),
        path,
    )

    if lake_dir is not None:
        from amdc_lake.observability import record_run

        if len(records) == 0:
            status = "fail"
        elif n_errors > 0:
            status = "partial"
        else:
            status = "success"
        with record_run("crawl", lake_dir, query=query, logger=log) as handle:
            handle.set_status(status)
            handle.set_rows_out(len(records))
            handle.update_details(
                sites=site_stats,
                output_parquet=str(path),
                n_sites=len(site_stats),
                n_errors=n_errors,
            )

    typer.echo(f"Wrote {len(records)} rows -> {path}")


if __name__ == "__main__":
    app()
