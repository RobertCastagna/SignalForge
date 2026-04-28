"""Silver Delta transforms and embedding table builds."""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal

import polars as pl
from deltalake import DeltaTable

from amdc_lake.constants import EMBEDDING_DIM, MODEL_NAME
from amdc_lake.ids import sha256_id
from amdc_lake.observability import record_run
from amdc_lake.paths import (
    bronze_scrapes_path,
    ensure_layers,
    silver_chunks_path,
    silver_pages_path,
)

log = logging.getLogger(__name__)


class _TimingEmbedder:
    """Proxy that tracks cumulative seconds spent in embedder.embed."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.elapsed_seconds = 0.0

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def embed(self, texts: list[str], *, batch_size: int = 8) -> list[list[float]]:
        start = perf_counter()
        try:
            return self._inner.embed(texts, batch_size=batch_size)
        finally:
            self.elapsed_seconds += perf_counter() - start

    def reset(self) -> None:
        self.elapsed_seconds = 0.0


PAGE_SCHEMA: dict[str, pl.DataType] = {
    "page_id": pl.Utf8,
    "bronze_id": pl.Utf8,
    "title": pl.Utf8,
    "date_published": pl.Utf8,
    "text": pl.Utf8,
    "text_hash": pl.Utf8,
    "source_url": pl.Utf8,
    "source_domain": pl.Utf8,
    "query": pl.Utf8,
    "crawled_at": pl.Utf8,
    "embedding": pl.List(pl.Float32),
    "embedding_model": pl.Utf8,
    "embedding_dim": pl.Int32,
    "embedded_at": pl.Utf8,
}

CHUNK_SCHEMA: dict[str, pl.DataType] = {
    "chunk_id": pl.Utf8,
    "page_id": pl.Utf8,
    "bronze_id": pl.Utf8,
    "chunk_index": pl.Int32,
    "chunk_text": pl.Utf8,
    "chunk_char_count": pl.Int32,
    "source_url": pl.Utf8,
    "source_domain": pl.Utf8,
    "query": pl.Utf8,
    "crawled_at": pl.Utf8,
    "embedding": pl.List(pl.Float32),
    "embedding_model": pl.Utf8,
    "embedding_dim": pl.Int32,
    "embedded_at": pl.Utf8,
}

_WS_RE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    return _WS_RE.sub(" ", html.unescape(text or "")).strip()


def read_bronze(lake_dir: Path) -> pl.DataFrame:
    table_path = bronze_scrapes_path(lake_dir)
    table = DeltaTable(str(table_path))
    return pl.from_arrow(table.to_pyarrow_table())


def _existing_silver_bronze_ids(pages_target: Path) -> set[str]:
    if not pages_target.exists():
        return set()
    try:
        table = DeltaTable(str(pages_target))
    except Exception:
        return set()
    arrow = table.to_pyarrow_table(columns=["bronze_id"])
    return set(arrow.column("bronze_id").to_pylist())


def build_pages(bronze: pl.DataFrame, embedder, *, batch_size: int) -> pl.DataFrame:
    if bronze.is_empty():
        return pl.DataFrame(schema=PAGE_SCHEMA)

    embedded_at = datetime.now(timezone.utc).isoformat()
    pages = (
        bronze.with_columns(
            pl.col("text").map_elements(clean_text, return_dtype=pl.Utf8).alias("text"),
        )
        .filter(pl.col("text").str.len_chars() > 0)
        .with_columns(
            pl.col("text")
            .map_elements(lambda value: sha256_id(value), return_dtype=pl.Utf8)
            .alias("text_hash"),
        )
        .with_columns(
            pl.struct(["bronze_id", "source_url", "query", "text_hash"])
            .map_elements(
                lambda row: sha256_id(
                    row["bronze_id"],
                    row["source_url"],
                    row["query"],
                    row["text_hash"],
                    prefix="page",
                ),
                return_dtype=pl.Utf8,
            )
            .alias("page_id"),
            pl.lit(MODEL_NAME).alias("embedding_model"),
            pl.lit(EMBEDDING_DIM).cast(pl.Int32).alias("embedding_dim"),
            pl.lit(embedded_at).alias("embedded_at"),
        )
        .unique(subset=["page_id"], keep="first")
    )
    vectors = embedder.embed(pages.get_column("text").to_list(), batch_size=batch_size)
    _validate_vectors(vectors)
    return pages.with_columns(
        pl.Series("embedding", vectors, dtype=pl.List(pl.Float32))
    ).select(list(PAGE_SCHEMA))


def build_chunks(
    pages: pl.DataFrame,
    embedder,
    *,
    batch_size: int,
    chunk_tokens: int,
    chunk_overlap: int,
) -> pl.DataFrame:
    if pages.is_empty():
        return pl.DataFrame(schema=CHUNK_SCHEMA)
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_tokens:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_tokens"
        )

    # Embedder adds special tokens (e.g. [CLS] [SEP]) at embed time; clamp the
    # raw chunk so the encoded sequence fits inside model_max_length without
    # silent truncation losing the tail of every chunk.
    special_overhead = embedder.tokenizer.num_special_tokens_to_add(pair=False)
    effective_max = max(1, embedder.max_length - special_overhead)
    if chunk_tokens > effective_max:
        log.info(
            "silver: clamping chunk_tokens %d -> %d (model_max=%d, special=%d)",
            chunk_tokens,
            effective_max,
            embedder.max_length,
            special_overhead,
        )
        chunk_tokens = effective_max
        if chunk_overlap >= chunk_tokens:
            chunk_overlap = max(0, chunk_tokens // 8)

    embedded_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    for row in pages.iter_rows(named=True):
        chunks = chunk_text(
            row["text"],
            embedder,
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
        )
        for index, text in enumerate(chunks):
            rows.append(
                {
                    "chunk_id": sha256_id(row["page_id"], index, text, prefix="chunk"),
                    "page_id": row["page_id"],
                    "bronze_id": row["bronze_id"],
                    "chunk_index": index,
                    "chunk_text": text,
                    "chunk_char_count": len(text),
                    "source_url": row["source_url"],
                    "source_domain": row["source_domain"],
                    "query": row["query"],
                    "crawled_at": row["crawled_at"],
                    "embedding_model": MODEL_NAME,
                    "embedding_dim": EMBEDDING_DIM,
                    "embedded_at": embedded_at,
                }
            )

    if not rows:
        return pl.DataFrame(schema=CHUNK_SCHEMA)

    chunks_df = pl.DataFrame(rows).with_columns(
        pl.col("chunk_index").cast(pl.Int32),
        pl.col("chunk_char_count").cast(pl.Int32),
        pl.col("embedding_dim").cast(pl.Int32),
    )
    vectors = embedder.embed(
        chunks_df.get_column("chunk_text").to_list(), batch_size=batch_size
    )
    _validate_vectors(vectors)
    return chunks_df.with_columns(
        pl.Series("embedding", vectors, dtype=pl.List(pl.Float32))
    ).select(list(CHUNK_SCHEMA))


def chunk_text(
    text: str, embedder, *, chunk_tokens: int, chunk_overlap: int
) -> list[str]:
    token_ids = embedder.tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    step = chunk_tokens - chunk_overlap
    chunks: list[str] = []
    for start in range(0, len(token_ids), step):
        piece = token_ids[start : start + chunk_tokens]
        decoded = clean_text(embedder.tokenizer.decode(piece, skip_special_tokens=True))
        if decoded:
            chunks.append(decoded)
        if start + chunk_tokens >= len(token_ids):
            break
    return chunks


def write_silver(
    pages: pl.DataFrame,
    chunks: pl.DataFrame,
    lake_dir: Path,
    *,
    mode: Literal["append", "overwrite"] = "append",
) -> tuple[Path, Path]:
    ensure_layers(lake_dir)
    pages_target = silver_pages_path(lake_dir)
    chunks_target = silver_chunks_path(lake_dir)
    pages_target.parent.mkdir(parents=True, exist_ok=True)
    write_options = {"schema_mode": "merge"} if mode == "append" else None
    _align_frame(pages, PAGE_SCHEMA).write_delta(
        str(pages_target), mode=mode, delta_write_options=write_options
    )
    _align_frame(chunks, CHUNK_SCHEMA).write_delta(
        str(chunks_target), mode=mode, delta_write_options=write_options
    )
    return pages_target, chunks_target


def build_silver(
    lake_dir: Path,
    *,
    batch_size: int = 8,
    chunk_tokens: int = 512,
    chunk_overlap: int = 64,
    device: str | None = None,
    rebuild: bool = False,
) -> tuple[Path, Path, int, int]:
    from amdc_lake.embedder import BgeM3Embedder

    bronze = read_bronze(lake_dir)
    log.info("silver: read %d bronze rows", bronze.height)

    pages_target = silver_pages_path(lake_dir)
    chunks_target = silver_chunks_path(lake_dir)
    write_mode: Literal["append", "overwrite"] = "overwrite" if rebuild else "append"

    if not rebuild:
        existing = _existing_silver_bronze_ids(pages_target)
        if existing:
            before = bronze.height
            bronze = bronze.filter(~pl.col("bronze_id").is_in(list(existing)))
            log.info(
                "silver: %d/%d bronze row(s) are new (incremental)",
                bronze.height,
                before,
            )

    if bronze.is_empty():
        log.info("silver: no new bronze rows; skipping embed")
        return pages_target, chunks_target, 0, 0

    pages_rows_in = (
        bronze.with_columns(
            pl.col("text")
            .map_elements(clean_text, return_dtype=pl.Utf8)
            .alias("_silver_text")
        )
        .filter(pl.col("_silver_text").str.len_chars() > 0)
        .height
    )

    timing_embedder = _TimingEmbedder(BgeM3Embedder(device=device))

    with record_run(
        "silver_pages", lake_dir, rows_in=pages_rows_in, logger=log
    ) as page_handle:
        timing_embedder.reset()
        pages = build_pages(bronze, timing_embedder, batch_size=batch_size)
        page_handle.set_rows_out(pages.height)
        page_handle.update_details(
            embed_seconds=round(timing_embedder.elapsed_seconds, 3),
            batch_size=batch_size,
            model=MODEL_NAME,
        )

    with record_run(
        "silver_chunks", lake_dir, rows_in=pages.height, logger=log
    ) as chunk_handle:
        timing_embedder.reset()
        chunks = build_chunks(
            pages,
            timing_embedder,
            batch_size=batch_size,
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
        )
        chunk_handle.set_rows_out(chunks.height)
        chunk_handle.update_details(
            embed_seconds=round(timing_embedder.elapsed_seconds, 3),
            batch_size=batch_size,
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
            model=MODEL_NAME,
        )

    pages_target, chunks_target = write_silver(pages, chunks, lake_dir, mode=write_mode)
    return pages_target, chunks_target, pages.height, chunks.height


def _validate_vectors(vectors: list[list[float]]) -> None:
    bad = [
        index for index, vector in enumerate(vectors) if len(vector) != EMBEDDING_DIM
    ]
    if bad:
        raise ValueError(
            f"expected {EMBEDDING_DIM}-dimensional embeddings; bad rows: {bad[:5]}"
        )


def _align_frame(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    for name, dtype in schema.items():
        if name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(name))
    casts = [
        pl.col(name).cast(dtype, strict=False).alias(name)
        for name, dtype in schema.items()
    ]
    return df.with_columns(casts).select(list(schema))
