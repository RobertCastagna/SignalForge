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


@app.command()
def run(
    query: str = typer.Argument(..., help="Search query"),
    data_dir: Path = typer.Option(
        Path("data"),
        "--data-dir",
        help="Output directory for parquet. Resolves to ./data on host or /app/data in Docker (WORKDIR=/app).",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raw = asyncio.run(crawl_all(query))
    records = normalize(raw, query=query)
    path = write_parquet(records, data_dir)
    typer.echo(f"Wrote {len(records)} rows -> {path}")


if __name__ == "__main__":
    app()
