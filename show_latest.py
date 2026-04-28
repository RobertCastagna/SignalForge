"""Print the most recent crawl output as a polars DataFrame.

Usage:
    uv run python show_latest.py
    uv run python show_latest.py --data-dir ./data --rows 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


def latest_parquet(data_dir: Path) -> Path | None:
    files = sorted(data_dir.glob("market_data_*.parquet"))
    return files[-1] if files else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Show the most recent AMDC run.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--rows", type=int, default=10, help="Rows to preview (0 = all).")
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Don't truncate the text column when printing.",
    )
    args = parser.parse_args()

    path = latest_parquet(args.data_dir)
    if path is None:
        print(f"No market_data_*.parquet files found in {args.data_dir.resolve()}", file=sys.stderr)
        return 1

    df = pl.read_parquet(path)
    print(f"File:       {path}")
    print(f"Rows:       {df.height}")
    print(f"Columns:    {df.columns}")
    if df.height:
        by_domain = df.group_by("source_domain").agg(pl.len().alias("rows"))
        print("\nRows by source_domain:")
        print(by_domain)

    if df.height and "text" in df.columns:
        sample_text = df.select(pl.col("text").head(1)).item()
        print("\nSample text (row 0):")
        print(sample_text)

    preview = df if args.rows == 0 else df.head(args.rows)
    if not args.full_text:
        preview = preview.with_columns(
            pl.col("text").str.slice(0, 200).str.replace_all(r"\s+", " ").alias("text")
        )

    print("\nPreview:")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=200, tbl_width_chars=200):
        print(preview)

    return 0


if __name__ == "__main__":
    sys.exit(main())
