from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from deltalake import DeltaTable

from amdc_lake.bronze import backfill_parquet, load_parquet_dir
from amdc_lake.constants import EMBEDDING_DIM, MODEL_NAME
from amdc_lake.ids import sha256_id
from amdc_lake.paths import (
    bronze_scrapes_path,
    ensure_layers,
    silver_chunks_path,
    silver_pages_path,
)
from amdc_lake.silver import (
    _validate_vectors,
    build_chunks,
    build_pages,
    chunk_text,
    clean_text,
    write_silver,
)


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(range(1, len(text.split()) + 1))

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok{token_id}" for token_id in token_ids)


class FakeEmbedder:
    tokenizer = FakeTokenizer()

    def embed(self, texts: list[str], *, batch_size: int = 8) -> list[list[float]]:
        return [
            [float(index)] + [0.0] * (EMBEDDING_DIM - 1)
            for index, _text in enumerate(texts)
        ]


def test_sha256_id_is_stable_and_prefixed() -> None:
    first = sha256_id("url", "query", "title", prefix="bronze")
    second = sha256_id("url", "query", "title", prefix="bronze")

    assert first == second
    assert first.startswith("bronze_")


def test_backfill_parquet_writes_deduped_bronze_delta(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    lake_dir = tmp_path / "lakehouse"
    input_dir.mkdir()
    rows = [
        {
            "title": "Fed decision",
            "date_published": "2026-04-28",
            "text": "Markets watch the Federal Reserve decision.",
            "source_url": "https://example.com/fed",
            "source_domain": "example.com",
            "relevance_score": 1.0,
            "query": "Fed rate decision",
            "crawled_at": "2026-04-28T00:00:00+00:00",
        },
        {
            "title": "Fed decision",
            "date_published": "2026-04-28",
            "text": "Markets watch the Federal Reserve decision.",
            "source_url": "https://example.com/fed",
            "source_domain": "example.com",
            "relevance_score": 1.0,
            "query": "Fed rate decision",
            "crawled_at": "2026-04-28T00:00:00+00:00",
        },
    ]
    pl.DataFrame(rows).write_parquet(input_dir / "market_data_20260428T000000Z.parquet")

    target, row_count = backfill_parquet(input_dir, lake_dir)
    bronze = pl.from_arrow(DeltaTable(str(target)).to_pyarrow_table())

    assert target == bronze_scrapes_path(lake_dir)
    assert row_count == 1
    assert bronze.height == 1
    assert bronze.select("bronze_id").item().startswith("bronze_")
    assert bronze.select("source_file").item().endswith("market_data_20260428T000000Z.parquet")


def test_load_parquet_dir_returns_empty_bronze_schema(tmp_path: Path) -> None:
    bronze = load_parquet_dir(tmp_path)

    assert bronze.is_empty()
    assert "bronze_id" in bronze.columns
    assert "ingested_at" in bronze.columns


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("  alpha\n\n beta\tgamma  ") == "alpha beta gamma"
    assert clean_text(None) == ""


def test_build_pages_skips_blank_text_and_adds_384_dim_embeddings() -> None:
    bronze = pl.DataFrame(
        [
            {
                "bronze_id": "bronze_1",
                "title": "Useful",
                "date_published": None,
                "text": "  useful\ntext  ",
                "source_url": "https://example.com/useful",
                "source_domain": "example.com",
                "query": "markets",
                "crawled_at": "2026-04-28T00:00:00+00:00",
            },
            {
                "bronze_id": "bronze_2",
                "title": "Blank",
                "date_published": None,
                "text": "   ",
                "source_url": "https://example.com/blank",
                "source_domain": "example.com",
                "query": "markets",
                "crawled_at": "2026-04-28T00:00:00+00:00",
            },
        ]
    )

    pages = build_pages(bronze, FakeEmbedder(), batch_size=2)

    assert pages.height == 1
    assert pages.select("text").item() == "useful text"
    assert pages.select("embedding_model").item() == MODEL_NAME
    assert pages.select("embedding_dim").item() == EMBEDDING_DIM
    assert pages.select(pl.col("embedding").list.len()).item() == EMBEDDING_DIM


def test_chunk_text_uses_token_windows_with_overlap() -> None:
    chunks = chunk_text(
        "one two three four five",
        FakeEmbedder(),
        chunk_tokens=3,
        chunk_overlap=1,
    )

    assert chunks == ["tok1 tok2 tok3", "tok3 tok4 tok5"]


def test_build_chunks_validates_chunk_settings_and_embeds_rows() -> None:
    pages = pl.DataFrame(
        [
            {
                "page_id": "page_1",
                "bronze_id": "bronze_1",
                "text": "one two three four five",
                "source_url": "https://example.com/page",
                "source_domain": "example.com",
                "query": "markets",
                "crawled_at": "2026-04-28T00:00:00+00:00",
            }
        ]
    )

    chunks = build_chunks(
        pages,
        FakeEmbedder(),
        batch_size=2,
        chunk_tokens=3,
        chunk_overlap=1,
    )

    assert chunks.height == 2
    assert chunks.select("chunk_index").to_series().to_list() == [0, 1]
    assert chunks.select("embedding_model").to_series().to_list() == [MODEL_NAME, MODEL_NAME]
    assert chunks.select(pl.col("embedding").list.len()).to_series().to_list() == [
        EMBEDDING_DIM,
        EMBEDDING_DIM,
    ]

    with pytest.raises(ValueError, match="chunk_tokens"):
        build_chunks(pages, FakeEmbedder(), batch_size=1, chunk_tokens=0, chunk_overlap=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        build_chunks(pages, FakeEmbedder(), batch_size=1, chunk_tokens=3, chunk_overlap=3)


def test_write_silver_round_trips_delta_vector_columns(tmp_path: Path) -> None:
    lake_dir = tmp_path / "lakehouse"
    vector = [0.0] * EMBEDDING_DIM
    pages = pl.DataFrame(
        [
            {
                "page_id": "page_test",
                "bronze_id": "bronze_test",
                "title": "Smoke",
                "date_published": None,
                "text": "sample text",
                "text_hash": "hash",
                "source_url": "https://example.com/smoke",
                "source_domain": "example.com",
                "query": "Fed rate decision",
                "crawled_at": "2026-04-28T00:00:00+00:00",
                "embedding": vector,
                "embedding_model": MODEL_NAME,
                "embedding_dim": EMBEDDING_DIM,
                "embedded_at": "2026-04-28T00:00:00+00:00",
            }
        ]
    )
    chunks = pl.DataFrame(
        [
            {
                "chunk_id": "chunk_test",
                "page_id": "page_test",
                "bronze_id": "bronze_test",
                "chunk_index": 0,
                "chunk_text": "sample text",
                "chunk_char_count": 11,
                "source_url": "https://example.com/smoke",
                "source_domain": "example.com",
                "query": "Fed rate decision",
                "crawled_at": "2026-04-28T00:00:00+00:00",
                "embedding": vector,
                "embedding_model": MODEL_NAME,
                "embedding_dim": EMBEDDING_DIM,
                "embedded_at": "2026-04-28T00:00:00+00:00",
            }
        ]
    )

    pages_target, chunks_target = write_silver(pages, chunks, lake_dir)
    pages_out = pl.from_arrow(DeltaTable(str(pages_target)).to_pyarrow_table())
    chunks_out = pl.from_arrow(DeltaTable(str(chunks_target)).to_pyarrow_table())

    assert pages_target == silver_pages_path(lake_dir)
    assert chunks_target == silver_chunks_path(lake_dir)
    assert pages_out.select("embedding_model").item() == MODEL_NAME
    assert chunks_out.select("embedding_dim").item() == EMBEDDING_DIM
    assert chunks_out.select(pl.col("embedding").list.len()).item() == EMBEDDING_DIM


def test_ensure_layers_creates_bronze_silver_and_gold(tmp_path: Path) -> None:
    ensure_layers(tmp_path)

    assert (tmp_path / "bronze").is_dir()
    assert (tmp_path / "silver").is_dir()
    assert (tmp_path / "gold").is_dir()


def test_validate_vectors_rejects_wrong_embedding_dim() -> None:
    with pytest.raises(ValueError, match=f"expected {EMBEDDING_DIM}-dimensional"):
        _validate_vectors([[0.0]])


def test_real_bge_small_embedder_outputs_configured_dimension() -> None:
    pytest.importorskip("torch")

    from amdc_lake.embedder import BgeM3Embedder

    embedder = BgeM3Embedder(device="cpu")
    vectors = embedder.embed(["Federal Reserve policy moved markets."], batch_size=1)

    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIM
    assert all(isinstance(value, float) for value in vectors[0])
