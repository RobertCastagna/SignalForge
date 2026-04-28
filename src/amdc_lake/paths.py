"""Lakehouse path conventions."""
from __future__ import annotations

from pathlib import Path

DEFAULT_LAKE_DIR = Path("data/lakehouse")

BRONZE_LAYER = "bronze"
SILVER_LAYER = "silver"
GOLD_LAYER = "gold"
QUALITY_LAYER = "_quality"
PIPELINE_LAYER = "_pipeline"

BRONZE_SCRAPES_TABLE = "scrapes"
BRONZE_SCRAPES_QUARANTINE_TABLE = "scrapes_quarantine"
SILVER_PAGES_TABLE = "pages"
SILVER_CHUNKS_TABLE = "chunks"
QUALITY_RUNS_TABLE = "runs"
PIPELINE_RUNS_TABLE = "runs"


def bronze_scrapes_path(lake_dir: Path) -> Path:
    return lake_dir / BRONZE_LAYER / BRONZE_SCRAPES_TABLE


def bronze_scrapes_quarantine_path(lake_dir: Path) -> Path:
    return lake_dir / BRONZE_LAYER / BRONZE_SCRAPES_QUARANTINE_TABLE


def silver_pages_path(lake_dir: Path) -> Path:
    return lake_dir / SILVER_LAYER / SILVER_PAGES_TABLE


def silver_chunks_path(lake_dir: Path) -> Path:
    return lake_dir / SILVER_LAYER / SILVER_CHUNKS_TABLE


def quality_runs_path(lake_dir: Path) -> Path:
    return lake_dir / QUALITY_LAYER / QUALITY_RUNS_TABLE


def pipeline_runs_path(lake_dir: Path) -> Path:
    return lake_dir / PIPELINE_LAYER / PIPELINE_RUNS_TABLE


def ensure_layers(lake_dir: Path) -> None:
    for layer in (BRONZE_LAYER, SILVER_LAYER, GOLD_LAYER, QUALITY_LAYER, PIPELINE_LAYER):
        (lake_dir / layer).mkdir(parents=True, exist_ok=True)

