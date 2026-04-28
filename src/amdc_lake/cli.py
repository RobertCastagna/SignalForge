"""Command-line entrypoint for the AMDC Delta Lake pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import typer

from amdc_lake.bronze import backfill_parquet
from amdc_lake.paths import DEFAULT_LAKE_DIR, bronze_scrapes_path, ensure_layers

app = typer.Typer(add_completion=False, help="AMDC Delta Lake pipeline")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command("init")
def init_lake(
    lake_dir: Path = typer.Option(
        DEFAULT_LAKE_DIR, "--lake-dir", help="Delta Lake root directory."
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    _configure_logging(log_level)
    ensure_layers(lake_dir)
    typer.echo(f"Initialized lakehouse layers under {lake_dir}")


@app.command("bronze-backfill")
def bronze_backfill(
    input_dir: Path = typer.Option(
        Path("data"),
        "--input-dir",
        help="Directory containing market_data_*.parquet files.",
    ),
    lake_dir: Path = typer.Option(
        DEFAULT_LAKE_DIR, "--lake-dir", help="Delta Lake root directory."
    ),
    mode: Literal["append", "overwrite"] = typer.Option(
        "overwrite", "--mode", help="Delta write mode."
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Run Bronze quality checks before writing.",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    _configure_logging(log_level)
    target, rows = backfill_parquet(input_dir, lake_dir, mode=mode, validate=validate)
    typer.echo(f"Wrote {rows} bronze rows -> {target}")


@app.command("silver-build")
def silver_build(
    lake_dir: Path = typer.Option(
        DEFAULT_LAKE_DIR, "--lake-dir", help="Delta Lake root directory."
    ),
    batch_size: int = typer.Option(
        8, "--batch-size", min=1, help="Embedding batch size."
    ),
    chunk_tokens: int = typer.Option(
        512, "--chunk-tokens", min=1, help="Tokenizer tokens per chunk."
    ),
    chunk_overlap: int = typer.Option(
        64, "--chunk-overlap", min=0, help="Overlapping tokenizer tokens per chunk."
    ),
    device: str | None = typer.Option(
        None, "--device", help="Torch device override, such as cpu or cuda."
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Force a full overwrite rebuild instead of incremental append.",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    _configure_logging(log_level)
    from amdc_lake.silver import build_silver

    pages_target, chunks_target, page_rows, chunk_rows = build_silver(
        lake_dir,
        batch_size=batch_size,
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
        device=device,
        rebuild=rebuild,
    )
    typer.echo(f"Wrote {page_rows} silver page rows -> {pages_target}")
    typer.echo(f"Wrote {chunk_rows} silver chunk rows -> {chunks_target}")


@app.command("quality-check")
def quality_check(
    lake_dir: Path = typer.Option(
        DEFAULT_LAKE_DIR, "--lake-dir", help="Delta Lake root directory."
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run Bronze quality checks against the existing Bronze table."""
    _configure_logging(log_level)
    import polars as pl
    from deltalake import DeltaTable

    from amdc_lake.quality.metrics import append_run
    from amdc_lake.quality.quarantine import write_quarantine
    from amdc_lake.quality.runner import run_bronze_checks

    table_path = bronze_scrapes_path(lake_dir)
    if not table_path.exists():
        typer.echo(f"No Bronze table at {table_path}")
        raise typer.Exit(code=1)

    df = pl.from_arrow(DeltaTable(str(table_path)).to_pyarrow_table())
    result = run_bronze_checks(df, lake_dir)
    write_quarantine(result.failures, lake_dir)
    append_run(result, lake_dir)

    typer.echo(f"status        : {result.status}")
    typer.echo(
        f"rows in/pass/fail: {result.rows_in} / {result.rows_passed} / {result.rows_failed}"
    )
    if result.check_summary:
        typer.echo("failing checks:")
        for entry in result.check_summary[:10]:
            typer.echo(
                f"  {entry['column']}: {entry['check']} ({entry['failed']} rows)"
            )
    if result.drift_report:
        typer.echo("drift findings:")
        for finding in result.drift_report:
            typer.echo(
                f"  [{finding['domain']}] {finding['metric']}: {finding['note']}"
            )

    if result.status == "fail":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
