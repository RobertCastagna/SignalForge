"""Self-updating RAG entrypoint for AMDC.

Embeds the user's query, scores it against existing Silver page embeddings, and
either returns the matching row_ids or kicks off a fresh crawl + Bronze +
Silver build before returning matches.

Usage:
    uv run python run_amdc.py "semiconductor supply chain"
    uv run python run_amdc.py "fed rate decision" --threshold 0.55 --min-articles 5
    uv run python run_amdc.py "semiconductor supply chain" --no-crawl
    uv run python run_amdc.py "semiconductor supply chain" --force-crawl
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np
import polars as pl
import typer
from deltalake import DeltaTable

from amdc.crawler import crawl_all
from amdc.extract import normalize
from amdc.store import write_parquet
from amdc_lake.bronze import backfill_parquet
from amdc_lake.embedder import BgeM3Embedder
from amdc_lake.paths import DEFAULT_LAKE_DIR, silver_pages_path
from amdc_lake.silver import build_silver

app = typer.Typer(add_completion=False, help="AMDC self-updating RAG entrypoint")
log = logging.getLogger(__name__)

_RESULT_COLUMNS = ["row_id", "similarity", "title", "source_url", "crawled_at"]
_EMPTY_RESULT_SCHEMA = {
    "row_id": pl.Utf8,
    "similarity": pl.Float32,
    "title": pl.Utf8,
    "source_url": pl.Utf8,
    "crawled_at": pl.Utf8,
}


def _empty_results() -> pl.DataFrame:
    return pl.DataFrame(schema=_EMPTY_RESULT_SCHEMA)


def _search(
    query: str,
    lake_dir: Path,
    threshold: float,
    top_k: int,
    embedder: BgeM3Embedder,
) -> pl.DataFrame:
    pages_path = silver_pages_path(lake_dir)
    if not pages_path.exists():
        return _empty_results()

    table = DeltaTable(str(pages_path))
    df = pl.from_arrow(
        table.to_pyarrow_table(
            columns=["page_id", "title", "source_url", "crawled_at", "embedding"]
        )
    )
    if df.is_empty():
        return _empty_results()

    embeddings = np.asarray(df["embedding"].to_list(), dtype=np.float32)
    query_vec = np.asarray(embedder.embed([query])[0], dtype=np.float32)
    sims = embeddings @ query_vec

    return (
        df.drop("embedding")
        .with_columns(pl.Series("similarity", sims, dtype=pl.Float32))
        .filter(pl.col("similarity") > threshold)
        .sort("similarity", descending=True)
        .head(top_k)
        .rename({"page_id": "row_id"})
        .select(_RESULT_COLUMNS)
    )


def _trigger_pipeline(query: str, data_dir: Path, lake_dir: Path) -> None:
    log.info("pipeline: starting crawl for query=%r", query)
    raw, site_stats = asyncio.run(crawl_all(query))
    n_errors = sum(1 for s in site_stats if s.get("error"))
    records = normalize(raw, query=query)
    parquet_path = write_parquet(records, data_dir)
    log.info(
        "pipeline: crawl wrote %d records (%d site errors) -> %s",
        len(records),
        n_errors,
        parquet_path,
    )
    bronze_target, bronze_rows = backfill_parquet(
        data_dir, lake_dir, mode="append", validate=True
    )
    log.info("pipeline: bronze append wrote %d row(s) to %s", bronze_rows, bronze_target)
    pages_target, chunks_target, page_rows, chunk_rows = build_silver(lake_dir)
    log.info(
        "pipeline: silver rebuilt — %d page row(s), %d chunk row(s)",
        page_rows,
        chunk_rows,
    )


@app.command()
def run(
    query: str = typer.Argument(..., help="Search query"),
    threshold: float = typer.Option(
        0.5, "--threshold", help="Cosine similarity cutoff (>) for a hit."
    ),
    min_articles: int = typer.Option(
        10, "--min-articles", help="Trigger crawl when fewer than this many hits."
    ),
    top_k: int = typer.Option(20, "--top-k", help="Max rows to display."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    lake_dir: Path = typer.Option(DEFAULT_LAKE_DIR, "--lake-dir"),
    force_crawl: bool = typer.Option(
        False, "--force-crawl", help="Always crawl, even if cache would hit."
    ),
    no_crawl: bool = typer.Option(
        False, "--no-crawl", help="Never crawl, even if under min-articles."
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    if force_crawl and no_crawl:
        raise typer.BadParameter("--force-crawl and --no-crawl are mutually exclusive")

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    embedder = BgeM3Embedder()

    if force_crawl:
        hits = _empty_results()
    else:
        hits = _search(query, lake_dir, threshold, top_k, embedder)

    should_crawl = force_crawl or (len(hits) < min_articles and not no_crawl)
    if should_crawl:
        typer.echo(
            f"cache miss ({len(hits)} < {min_articles} articles above {threshold}); "
            f"crawling..."
        )
        _trigger_pipeline(query, data_dir, lake_dir)
        hits = _search(query, lake_dir, threshold, top_k, embedder)
    elif len(hits) < min_articles:
        typer.echo(
            f"cache below min_articles ({len(hits)} < {min_articles}) but "
            f"--no-crawl set; returning what we have."
        )

    typer.echo(
        f"query={query!r} matches={len(hits)} threshold={threshold} "
        f"min_articles={min_articles}"
    )
    with pl.Config(tbl_rows=top_k, tbl_cols=-1, fmt_str_lengths=80, tbl_width_chars=200):
        print(hits)


if __name__ == "__main__":
    app()
