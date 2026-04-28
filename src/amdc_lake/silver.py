"""Silver Delta transforms and embedding table builds."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from deltalake import DeltaTable

from amdc_lake.constants import EMBEDDING_DIM, MODEL_NAME
from amdc_lake.ids import sha256_id
from amdc_lake.paths import (
    bronze_scrapes_path,
    ensure_layers,
    silver_chunks_path,
    silver_pages_path,
)

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
    return _WS_RE.sub(" ", text or "").strip()


def read_bronze(lake_dir: Path) -> pl.DataFrame:
    table_path = bronze_scrapes_path(lake_dir)
    table = DeltaTable(str(table_path))
    return pl.from_arrow(table.to_pyarrow_table())


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
            pl.col("text").map_elements(lambda value: sha256_id(value), return_dtype=pl.Utf8).alias("text_hash"),
        )
        .with_columns(
            pl.struct(["bronze_id", "source_url", "query", "text_hash"]).map_elements(
                lambda row: sha256_id(
                    row["bronze_id"],
                    row["source_url"],
                    row["query"],
                    row["text_hash"],
                    prefix="page",
                ),
                return_dtype=pl.Utf8,
            ).alias("page_id"),
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
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_tokens")

    embedded_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    for row in pages.iter_rows(named=True):
        chunks = chunk_text(row["text"], embedder, chunk_tokens=chunk_tokens, chunk_overlap=chunk_overlap)
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
    vectors = embedder.embed(chunks_df.get_column("chunk_text").to_list(), batch_size=batch_size)
    _validate_vectors(vectors)
    return chunks_df.with_columns(
        pl.Series("embedding", vectors, dtype=pl.List(pl.Float32))
    ).select(list(CHUNK_SCHEMA))


def chunk_text(text: str, embedder, *, chunk_tokens: int, chunk_overlap: int) -> list[str]:
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


def write_silver(pages: pl.DataFrame, chunks: pl.DataFrame, lake_dir: Path) -> tuple[Path, Path]:
    ensure_layers(lake_dir)
    pages_target = silver_pages_path(lake_dir)
    chunks_target = silver_chunks_path(lake_dir)
    pages_target.parent.mkdir(parents=True, exist_ok=True)
    _align_frame(pages, PAGE_SCHEMA).write_delta(str(pages_target), mode="overwrite")
    _align_frame(chunks, CHUNK_SCHEMA).write_delta(str(chunks_target), mode="overwrite")
    return pages_target, chunks_target


def build_silver(
    lake_dir: Path,
    *,
    batch_size: int = 8,
    chunk_tokens: int = 512,
    chunk_overlap: int = 64,
    device: str | None = None,
) -> tuple[Path, Path, int, int]:
    from amdc_lake.embedder import BgeM3Embedder

    bronze = read_bronze(lake_dir)
    embedder = BgeM3Embedder(device=device)
    pages = build_pages(bronze, embedder, batch_size=batch_size)
    chunks = build_chunks(
        pages,
        embedder,
        batch_size=batch_size,
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
    )
    pages_target, chunks_target = write_silver(pages, chunks, lake_dir)
    return pages_target, chunks_target, pages.height, chunks.height


def _validate_vectors(vectors: list[list[float]]) -> None:
    bad = [index for index, vector in enumerate(vectors) if len(vector) != EMBEDDING_DIM]
    if bad:
        raise ValueError(f"expected {EMBEDDING_DIM}-dimensional embeddings; bad rows: {bad[:5]}")


def _align_frame(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    for name, dtype in schema.items():
        if name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(name))
    casts = [pl.col(name).cast(dtype, strict=False).alias(name) for name, dtype in schema.items()]
    return df.with_columns(casts).select(list(schema))
