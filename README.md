# Adaptive Market Data Crawler (AMDC)

Dockerized Python CLI that runs `crawl4ai`'s `AdaptiveCrawler` against three
predefined market-data sites for a given search query and writes the results
to a parquet file for downstream `polars` analysis. It also includes a Delta
Lake pipeline that ports those local parquet outputs into Bronze and Silver
layers with BAAI/bge-small-en-v1.5 embeddings.

## Stack

- Python 3.12 in `python:3.12-slim-bookworm`
- [`uv`](https://github.com/astral-sh/uv) for dependency resolution
- `crawl4ai` (Playwright/Chromium under the hood) for adaptive crawling
- `polars` + `pyarrow` for the parquet write/read path
- `delta-rs` for Delta Lake writes
- Hugging Face `transformers` + PyTorch for BAAI/bge-small-en-v1.5 embeddings
- `typer` for the CLI

## Build

```bash
docker build -t market-crawler .
```

The first build is slow because Chromium is downloaded by `playwright install`.
The image also pre-downloads `BAAI/bge-small-en-v1.5`, so the first Silver build does not
need to fetch model weights at runtime.

## Run

```bash
mkdir -p data
docker run --rm -v "$(pwd)/data:/app/data" market-crawler "semiconductor supply chain"
```

On success, the container prints:

```
Wrote N rows -> /app/data/market_data_<UTC timestamp>.parquet
```

That file lives on the host at `./data/market_data_<UTC timestamp>.parquet`
because of the volume mount.

## Read the output with polars

From the host (anywhere `polars` is available):

```bash
uv run --with polars --with pyarrow python -c "
import glob, polars as pl
path = sorted(glob.glob('data/market_data_*.parquet'))[-1]
print(pl.read_parquet(path).head())
"
```

Schema: `title`, `date_published`, `text`, `source_url`, `source_domain`,
`relevance_score`, `query`, `crawled_at`.

## Delta Lake pipeline

The lakehouse lives under `data/lakehouse` by default:

```text
data/lakehouse/
  bronze/scrapes/
  silver/pages/
  silver/chunks/
  gold/
```

Initialize the layer directories:

```bash
uv run amdc-lake init --lake-dir ./data/lakehouse
```

Backfill the existing local parquet files into Bronze Delta:

```bash
uv run amdc-lake bronze-backfill --input-dir ./data --lake-dir ./data/lakehouse
```

Build Silver page and chunk embedding tables from Bronze:

```bash
uv run amdc-lake silver-build \
  --lake-dir ./data/lakehouse \
  --batch-size 8 \
  --chunk-tokens 512 \
  --chunk-overlap 64
```

Silver writes two Delta tables:

- `silver/pages`: one cleaned page-level row per scrape with a 384-float
  BAAI/bge-small-en-v1.5 embedding.
- `silver/chunks`: retrieval-sized text chunks with their own 384-float
  BAAI/bge-small-en-v1.5 embeddings.

Gold is intentionally empty until downstream analytics requirements are added.

With Docker, run the lake CLI by overriding the entrypoint:

```bash
docker run --rm -v "$(pwd)/data:/app/data" --entrypoint amdc-lake market-crawler init --lake-dir /app/data/lakehouse
docker run --rm -v "$(pwd)/data:/app/data" --entrypoint amdc-lake market-crawler bronze-backfill --input-dir /app/data --lake-dir /app/data/lakehouse
docker run --rm -v "$(pwd)/data:/app/data" --entrypoint amdc-lake market-crawler silver-build --lake-dir /app/data/lakehouse
```

## Configuration

Targets and tunables live in `src/amdc/config.py`:

- `SITES` — list of three start URLs (Yahoo Finance, Reuters Markets,
  MarketWatch by default; swap freely).
- `RATE_LIMIT_SECONDS` — politeness delay between sites.
- `ADAPTIVE` — `confidence_threshold`, `max_pages`, `top_k_links`, `strategy`
  passed to `AdaptiveConfig`.

## Local development without Docker

```bash
uv sync
uv run playwright install chromium
uv run amdc "semiconductor supply chain" --data-dir ./data
uv run amdc-lake bronze-backfill --input-dir ./data --lake-dir ./data/lakehouse
```

## Tests

The pipeline tests are designed to run inside the Docker image and include one
real `BAAI/bge-small-en-v1.5` embedding smoke test:

```bash
docker build -t market-crawler .
docker run --rm --entrypoint pytest market-crawler tests
```

For an already-running test container:

```bash
docker exec signalforge-test pytest tests
```

## Error handling

If one of the three sites blocks the crawler or fails, the error is logged
and the run continues with the remaining sites. The output file always
includes whatever was successfully extracted.
