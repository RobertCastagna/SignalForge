"""Command-line entrypoint for the AMDC Delta Lake pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer

from amdc_lake.bronze import backfill_parquet
from amdc_lake.paths import DEFAULT_LAKE_DIR, ensure_layers

app = typer.Typer(add_completion=False, help="AMDC Delta Lake pipeline")


@app.command("init")
def init_lake(
    lake_dir: Path = typer.Option(DEFAULT_LAKE_DIR, "--lake-dir", help="Delta Lake root directory."),
) -> None:
    ensure_layers(lake_dir)
    typer.echo(f"Initialized lakehouse layers under {lake_dir}")


@app.command("bronze-backfill")
def bronze_backfill(
    input_dir: Path = typer.Option(Path("data"), "--input-dir", help="Directory containing market_data_*.parquet files."),
    lake_dir: Path = typer.Option(DEFAULT_LAKE_DIR, "--lake-dir", help="Delta Lake root directory."),
    mode: Literal["append", "overwrite"] = typer.Option("overwrite", "--mode", help="Delta write mode."),
) -> None:
    target, rows = backfill_parquet(input_dir, lake_dir, mode=mode)
    typer.echo(f"Wrote {rows} bronze rows -> {target}")


@app.command("silver-build")
def silver_build(
    lake_dir: Path = typer.Option(DEFAULT_LAKE_DIR, "--lake-dir", help="Delta Lake root directory."),
    batch_size: int = typer.Option(8, "--batch-size", min=1, help="Embedding batch size."),
    chunk_tokens: int = typer.Option(512, "--chunk-tokens", min=1, help="Tokenizer tokens per chunk."),
    chunk_overlap: int = typer.Option(64, "--chunk-overlap", min=0, help="Overlapping tokenizer tokens per chunk."),
    device: str | None = typer.Option(None, "--device", help="Torch device override, such as cpu or cuda."),
) -> None:
    from amdc_lake.silver import build_silver

    pages_target, chunks_target, page_rows, chunk_rows = build_silver(
        lake_dir,
        batch_size=batch_size,
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
        device=device,
    )
    typer.echo(f"Wrote {page_rows} silver page rows -> {pages_target}")
    typer.echo(f"Wrote {chunk_rows} silver chunk rows -> {chunks_target}")


if __name__ == "__main__":
    app()
