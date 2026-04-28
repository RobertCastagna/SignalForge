"""Self-updating RAG entrypoint for AMDC.

Embeds the user's query, scores it against Silver chunk embeddings, joins the
best chunk per article back to Silver pages for context, and either returns
matching articles or kicks off a fresh crawl + Bronze + Silver build first.

Usage:
    uv run python run_amdc.py "semiconductor supply chain"
    uv run python run_amdc.py "fed rate decision" --threshold 0.55 --min-articles 5
    uv run python run_amdc.py "semiconductor supply chain" --no-crawl
    uv run python run_amdc.py "semiconductor supply chain" --force-crawl
"""

from __future__ import annotations

# torch 2.2.2 (Intel Mac ceiling — see CLAUDE.md) was compiled against NumPy 1.x
# while we resolve NumPy 2.x as a transitive dep. The mismatch is benign for
# this code path. NumPy writes the multi-line compat banner directly to stderr
# from C (bypassing the warnings module), so we capture stderr around the
# heavy import chain and drop the known noise. The companion torch UserWarning
# does flow through warnings, so a regular filter handles that one.
import io  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=r".*Failed to initialize NumPy.*")

_stderr_orig = sys.stderr
sys.stderr = io.StringIO()
try:
    import asyncio
    import logging
    from dataclasses import dataclass
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
    from amdc_lake.paths import (
        DEFAULT_LAKE_DIR,
        silver_chunks_path,
        silver_pages_path,
    )
    from amdc_lake.silver import build_silver
finally:
    sys.stderr = _stderr_orig
    # Captured stderr is discarded — the only writers during the import chain
    # are NumPy (compat banner) and torch (Failed to initialize NumPy). If an
    # import raises, the traceback prints to the restored stderr after this
    # finally block runs.

app = typer.Typer(add_completion=False, help="AMDC self-updating RAG entrypoint")
log = logging.getLogger(__name__)

_RESULT_COLUMNS = [
    "row_id",
    "similarity",
    "title",
    "source_url",
    "crawled_at",
    "text",
]
_EMPTY_RESULT_SCHEMA = {
    "row_id": pl.Utf8,
    "similarity": pl.Float32,
    "title": pl.Utf8,
    "source_url": pl.Utf8,
    "crawled_at": pl.Utf8,
    "text": pl.Utf8,
}


@dataclass(frozen=True)
class QueryResult:
    query: str
    hits: pl.DataFrame
    initial_matches: int
    crawled: bool
    threshold: float
    min_articles: int
    top_k: int
    status_message: str

    @property
    def final_matches(self) -> int:
        return len(self.hits)


def _empty_results() -> pl.DataFrame:
    return pl.DataFrame(schema=_EMPTY_RESULT_SCHEMA)


def _search(
    query: str,
    lake_dir: Path,
    threshold: float,
    top_k: int,
    embedder: BgeM3Embedder,
) -> pl.DataFrame:
    chunks_path = silver_chunks_path(lake_dir)
    pages_path = silver_pages_path(lake_dir)
    if not chunks_path.exists() or not pages_path.exists():
        log.info("cache: silver is empty (no chunks indexed yet)")
        return _empty_results()

    chunks = pl.from_arrow(
        DeltaTable(str(chunks_path)).to_pyarrow_table(columns=["page_id", "embedding"])
    )
    if chunks.is_empty():
        log.info("cache: silver is empty (no chunks indexed yet)")
        return _empty_results()

    embeddings = np.asarray(chunks["embedding"].to_list(), dtype=np.float32)
    query_vec = np.asarray(embedder.embed([query])[0], dtype=np.float32)
    sims = embeddings @ query_vec

    n_chunks_above = int((sims > threshold).sum())
    max_sim = float(sims.max()) if sims.size else 0.0
    log.info(
        "cache: scanned %d chunks (%d pages), %d chunks above threshold=%.2f (max sim=%.3f)",
        len(chunks),
        chunks["page_id"].n_unique(),
        n_chunks_above,
        threshold,
        max_sim,
    )

    chunk_hits = (
        chunks.drop("embedding")
        .with_columns(pl.Series("similarity", sims, dtype=pl.Float32))
        .filter(pl.col("similarity") > threshold)
    )
    if chunk_hits.is_empty():
        return _empty_results()

    best_per_page = (
        chunk_hits.sort("similarity", descending=True)
        .unique(subset=["page_id"], keep="first")
        .head(top_k)
    )

    pages = pl.from_arrow(
        DeltaTable(str(pages_path)).to_pyarrow_table(
            columns=["page_id", "title", "source_url", "crawled_at", "text"]
        )
    )
    return (
        best_per_page.join(pages, on="page_id", how="inner")
        .sort("similarity", descending=True)
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
    log.info(
        "pipeline: bronze append wrote %d row(s) to %s", bronze_rows, bronze_target
    )
    pages_target, chunks_target, page_rows, chunk_rows = build_silver(lake_dir)
    log.info(
        "pipeline: silver wrote %d new page row(s), %d new chunk row(s)",
        page_rows,
        chunk_rows,
    )


def orchestrate_query(
    query: str,
    *,
    threshold: float = 0.5,
    min_articles: int = 10,
    top_k: int = 20,
    data_dir: Path = Path("data"),
    lake_dir: Path = DEFAULT_LAKE_DIR,
    force_crawl: bool = False,
    no_crawl: bool = False,
    embedder: BgeM3Embedder | None = None,
) -> QueryResult:
    if force_crawl and no_crawl:
        raise ValueError("--force-crawl and --no-crawl are mutually exclusive")

    embedder = embedder or BgeM3Embedder()

    if force_crawl:
        hits = _empty_results()
    else:
        hits = _search(query, lake_dir, threshold, top_k, embedder)

    initial_matches = len(hits)
    should_crawl = force_crawl or (initial_matches < min_articles and not no_crawl)
    if should_crawl:
        status_message = (
            f"cache miss ({initial_matches} < {min_articles} articles above "
            f"{threshold}); crawling..."
        )
        _trigger_pipeline(query, data_dir, lake_dir)
        hits = _search(query, lake_dir, threshold, top_k, embedder)
    elif initial_matches < min_articles:
        status_message = (
            f"cache below min_articles ({initial_matches} < {min_articles}) but "
            f"--no-crawl set; returning what we have."
        )
    else:
        status_message = (
            f"cache hit ({initial_matches} >= {min_articles} articles above "
            f"{threshold}); returning cached matches."
        )

    return QueryResult(
        query=query,
        hits=hits,
        initial_matches=initial_matches,
        crawled=should_crawl,
        threshold=threshold,
        min_articles=min_articles,
        top_k=top_k,
        status_message=status_message,
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

    result = orchestrate_query(
        query,
        threshold=threshold,
        min_articles=min_articles,
        top_k=top_k,
        data_dir=data_dir,
        lake_dir=lake_dir,
        force_crawl=force_crawl,
        no_crawl=no_crawl,
    )
    if result.crawled or result.initial_matches < min_articles:
        typer.echo(result.status_message)

    typer.echo(
        f"query={query!r} matches={result.final_matches} threshold={threshold} "
        f"min_articles={min_articles}"
    )
    with pl.Config(
        tbl_rows=top_k, tbl_cols=-1, fmt_str_lengths=80, tbl_width_chars=200
    ):
        print(result.hits)


if __name__ == "__main__":
    app()
