# Adaptive Market Data Crawler (AMDC)

Dockerized Python CLI that runs `crawl4ai`'s `BestFirstCrawlingStrategy` (with
BM25 content filtering and keyword scoring) against three predefined market-data
sites — CNBC, Investing.com, and Finviz — for a given search query, then writes
results to parquet for downstream `polars` analysis. A companion Delta Lake
pipeline (`amdc-lake`) promotes those parquet outputs through Bronze (with
Pandera data-quality validation) and Silver (page- and chunk-level tables with
384-dim BAAI/bge-small-en-v1.5 embeddings). Every stage records a durable run
row to `_pipeline/runs` so timing, row counts, and per-site / per-batch detail
are inspectable after the fact.

## Stack

- Python 3.12 in `python:3.12-slim-bookworm`
- [`uv`](https://github.com/astral-sh/uv) for dependency resolution
- `crawl4ai` (Playwright/Chromium under the hood) for `BestFirstCrawlingStrategy`
- `pandera[polars]` for Bronze data-quality validation
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

## Observing pipeline runs

Every Bronze and Silver build appends one row per stage to `{lake_dir}/_pipeline/runs`
(crawl runs append a row only when `amdc` is called with `--lake-dir`). Each row
captures `started_at`, `finished_at`, `duration_ms`, `rows_in`, `rows_out`,
`status` (`success` / `partial` / `fail`), and a `details` JSON blob with
stage-specific context (per-site stats for crawl, embed seconds and batch size
for silver, source files and quarantine count for bronze).

```python
import polars as pl
pl.read_delta("data/lakehouse/_pipeline/runs") \
  .sort("started_at", descending=True) \
  .select(["stage", "status", "duration_ms", "rows_in", "rows_out"]) \
  .head(20)
```

`_pipeline/runs` (stage execution) and `_quality/runs` (Bronze data validation)
are complementary: the bronze row's `details.quality_run_id` cross-links to the
matching `_quality/runs` row.

## Configuration

Crawl targets and tunables live in `src/amdc/config.py`:

- `SITES` — three start URLs with per-site `url_patterns` and an
  `include_external` flag. Defaults are `cnbc.com/markets/`,
  `investing.com/news/stock-market-news`, and `finviz.com/news.ashx`. Swap
  freely; the Pandera Bronze schema derives its allowed-domain set from this
  list, so adding a site here is enough.
- `DEEP_CRAWL_MAX_DEPTH` (4) and `DEEP_CRAWL_MAX_PAGES` (60) — link-hops and
  page ceiling per site for the `BestFirstCrawlingStrategy`.
- `MIN_RAW_MARKDOWN_CHARS` (800) and `MIN_FIT_MARKDOWN_CHARS` (200) — keep a
  page when *either* the raw body or the BM25-filtered body clears its floor.
- `BM25_THRESHOLD` (0.3) — content filter cutoff for the BM25 pass.
- `CONCURRENCY_PER_SITE` (10) and `PARALLEL_SITES` (True) — intra- and
  cross-site fetch parallelism.
- `TEXT_CHAR_CAP` (8000) — per-record body length cap.
- `PAGE_TIMEOUT_MS` (20_000), `STEALTH` (True), `SIMULATE_USER` (False) —
  browser tuning.

Lakehouse paths live in `src/amdc_lake/paths.py`. Tables under `bronze/`,
`silver/`, `gold/`, `_quality/`, and `_pipeline/` are created by
`amdc-lake init`.

## Data quality

Bronze writes route through a Pandera `DataFrameSchema`
(`src/amdc_lake/quality/schemas.py`) that enforces:

- `bronze_id` SHA256 shape and uniqueness
- `source_url` HTTP(S) prefix
- `source_domain` ∈ allowed set (derived from `SITES`)
- text length bounds, BM25 score floor, and a custom URL-junk-ratio check

Failures are appended to `bronze/scrapes_quarantine` (with `_failure_reasons`
and a `_quality_run_id` cross-link) rather than written to Bronze. Each
validation run also appends to `_quality/runs` with summaries and a
trailing-window drift report (per-domain row count, mean text length, unique
URL count). Invoke standalone:

```bash
uv run amdc-lake quality-check --lake-dir ./data/lakehouse
```

`bronze-backfill` runs the same checks by default; pass `--no-validate` to
skip.

## Local development without Docker

```bash
uv sync
uv run playwright install chromium

# Crawl. --lake-dir is optional; when set, the run is recorded to _pipeline/runs.
uv run amdc "semiconductor supply chain" \
  --data-dir ./data --lake-dir ./data/lakehouse

# Lakehouse pipeline
uv run amdc-lake init           --lake-dir ./data/lakehouse
uv run amdc-lake bronze-backfill --input-dir ./data --lake-dir ./data/lakehouse
uv run amdc-lake silver-build    --lake-dir ./data/lakehouse
```

Every `amdc-lake` command takes `--log-level` (default `INFO`) for stdout
log verbosity.

## Tests

The full suite runs locally without Docker:

```bash
uv run pytest tests
```

55 unit tests cover the crawler, extract, store, lakehouse pipeline (Bronze
and Silver), CLI surfaces, data-quality framework, and the observability
module. Coverage is ~92% across the `src/` tree; run it yourself with:

```bash
uv run --with coverage --with pytest python -m coverage run --source=src \
  -m pytest tests --deselect \
  tests/test_lakehouse_pipeline.py::test_real_bge_small_embedder_outputs_configured_dimension
uv run --with coverage python -m coverage report
```

The deselected test loads the real `BAAI/bge-small-en-v1.5` weights and
needs torch + the model cache; it runs by default but can be skipped on
machines without torch installed.

To run the same tests inside the Docker image:

```bash
docker build -t market-crawler .
docker run --rm --entrypoint pytest market-crawler tests
```

## Error handling

If one of the three sites blocks the crawler or fails, the error is logged
and the run continues with the remaining sites. The output parquet always
includes whatever was successfully extracted; when `amdc` is invoked with
`--lake-dir`, the corresponding `_pipeline/runs` row is recorded with
`status="partial"` (≥1 site errored, ≥1 succeeded) or `status="fail"`
(zero rows written) so the failure is durable, not just in stdout.
