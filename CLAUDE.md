# CLAUDE.md — SignalForge / AMDC

Guidance for Claude (and humans) working in this repository.

## Project overview

**Adaptive Market Data Crawler (AMDC)** — scrapes financial news from CNBC, Investing.com, and Finviz, writes raw parquet, then promotes it through a Bronze → Silver → Gold Delta Lake with BAAI/bge-small-en-v1.5 embeddings (384-dim). Gold is intentionally empty; downstream analytics live there later.

Two installable packages:
- `amdc` — the crawler (`src/amdc/`).
- `amdc_lake` — the lakehouse pipeline (`src/amdc_lake/`).

## Tech stack

- Python **3.12** (see `pyproject.toml` `requires-python`).
- `uv` for env + lockfile (`uv.lock` is committed).
- `crawl4ai` (Playwright/Chromium) for scraping.
- `polars` + `pyarrow` for columnar IO; `deltalake>=0.25` for table format.
- `transformers` + `torch` for embeddings (BAAI/bge-small-en-v1.5).
- `typer` for both CLIs.

## Layout

```
src/amdc/         crawler package
  main.py         Typer CLI: `amdc <query>`
  config.py       SITES list + crawl tunables (see "Hot spots")
  crawler.py      crawl_all(query) — BestFirstCrawlingStrategy orchestration
  extract.py      normalize raw pages → flat records
  store.py        write_parquet(records, data_dir) → Path

src/amdc_lake/    lakehouse pipeline
  cli.py          Typer CLI: `amdc-lake init|bronze-backfill|silver-build`
  paths.py        Lakehouse layer + table path conventions
  bronze.py       Raw parquet → Bronze Delta (dedup + composite ID)
  silver.py       Bronze → page-level + chunk-level Silver (with embeddings)
  embedder.py     BGE-small wrapper around HF AutoModel/AutoTokenizer
  ids.py          sha256_id(*parts, prefix) helper

scripts/          probe_sites.py, validate_sites.py — site reachability probes
data/             gitignored output (parquet + lakehouse/)
show_latest.py    quick preview of the most recent crawl parquet
Dockerfile        multi-stage uv + Playwright + BGE-small weights pre-baked
```

## Common commands

Local dev (no Docker):

```bash
uv sync
uv run playwright install chromium

uv run amdc "semiconductor supply chain" --data-dir ./data
uv run amdc-lake init           --lake-dir ./data/lakehouse
uv run amdc-lake bronze-backfill --input-dir ./data --lake-dir ./data/lakehouse
uv run amdc-lake silver-build    --lake-dir ./data/lakehouse --batch-size 8 \
                                 --chunk-tokens 512 --chunk-overlap 64

uv run python show_latest.py    # preview latest parquet
```

Docker (recommended for reproducibility — see `README.md` for the full set):

```bash
docker build -t market-crawler .
docker run --rm -p 8501:8501 -v "$(pwd)/data:/app/data" market-crawler
docker run --rm -v "$(pwd)/data:/app/data" market-crawler amdc "<query>"
docker run --rm -v "$(pwd)/data:/app/data" market-crawler amdc-lake \
  silver-build --lake-dir /app/data/lakehouse

docker run --rm market-crawler pytest tests
```

## Data conventions

- Raw filename: `data/market_data_<YYYYMMDDTHHMMSSZ>.parquet` (UTC).
- All IDs are SHA256 composites built via `amdc_lake.ids.sha256_id`:
  - `bronze_id` = hash(source_url, query, crawled_at, title, text)
  - `page_id`   = hash(bronze_id, source_url, query, text_hash)
  - `chunk_id`  = hash(page_id, chunk_index, chunk_text)
- Embedding columns: `List[Float32]`, dim **384**, model `BAAI/bge-small-en-v1.5`.
- Chunking is **token-based** (sliding window over tokenizer output), not char-based; defaults 512 tokens / 64 overlap.

## Configuration hot spots

Almost all crawler tuning lives in **`src/amdc/config.py`**:

| Constant | Purpose |
|---|---|
| `SITES` | List of crawl targets (URL, link patterns, allow-externals). Swap domains here. |
| `DEEP_CRAWL_MAX_DEPTH` (4) | Link-hops per site. |
| `DEEP_CRAWL_MAX_PAGES` (30) | Page ceiling per site. |
| `CONCURRENCY_PER_SITE` (10) | Parallel fetches inside one site's deep crawl. |
| `PARALLEL_SITES` (True) | Cross-site concurrency via `asyncio.gather`. |
| `MIN_FIT_MARKDOWN_CHARS` (500) | Drop shell/empty pages post-filter. |
| `BM25_THRESHOLD` (1.0) | Content relevance cutoff. |
| `TEXT_CHAR_CAP` (8000) | Per-record text length cap. |
| `PAGE_TIMEOUT_MS` (12_000) | Per-page browser timeout. |

Lakehouse paths and layer names: `src/amdc_lake/paths.py`.

## Embedding constraints

- BGE-small weights are pre-downloaded at Docker build time. Don't re-download at runtime in images.
- `BgeM3Embedder` is the historical class name; it currently loads `BAAI/bge-small-en-v1.5` and auto-selects CUDA if present, else CPU.
- **Intel Mac (x86_64 darwin) caveat**: `pyproject.toml` excludes `torch` from the Linux CPU index on darwin x86_64; the local install uses whatever wheel pip can resolve. PyTorch wheels for Intel Mac top out at **2.2.2**, which forces `transformers<5` and rules out the Qwen3.5+ family. Don't bump these past the documented ceiling without testing on the actual hardware.

## Git workflow

### Branching strategy

- `main` is the protected, releasable branch. **Never push directly to `main`** — local or CI.
- `dev` is the long-lived integration branch. Day-to-day work lands here first.
- Feature work happens on short-lived branches cut **from `dev`**:
  `<type>/<short-kebab-description>` — e.g. `feat/silver-incremental-build`, `fix/cnbc-date-parser`, `chore/bump-deltalake`.
  Squash-merge back into `dev` and delete the branch.
- `dev` → `main` is **always via a Pull Request opened in the GitHub UI**, with passing integration tests. Never bypass with a local `git push origin dev:main`.

```
feat/<x>  ──▶  dev  ──(PR + integration tests in GitHub UI)──▶  main
```

### Working from a plan (Claude / agents)

When a session produces an implementation plan (plan mode, `/plan`, or any explicit design step) and is about to start writing code, **cut a feature branch from `dev` before the first edit**. Do not start coding on `dev` or `main` once a plan exists.

Steps before the first edit:
1. `git fetch origin && git checkout dev && git pull --ff-only`
2. `git checkout -b <type>/<short-kebab-summary>` — name derived from the plan, e.g. `feat/bronze-data-quality-layer`, `fix/cnbc-date-parser`.
3. Then implement.

Override: if the user explicitly says "commit to dev" (or similar), respect that and skip the branch.

### Commit messages — Conventional Commits

```
<type>(<optional-scope>): <imperative summary, ≤72 chars>

<optional body — wrap at 72 cols, explain *why*, not *what*>

<optional footer — e.g. "Closes #12", "BREAKING CHANGE: ...">
```

Allowed types:

| Type | Use for |
|---|---|
| `feat` | new user-facing capability |
| `fix` | bug fix (canonical — **do not use `bug:`**) |
| `docs` | documentation only |
| `style` | formatting/whitespace, no behavior change |
| `refactor` | restructure without changing behavior |
| `perf` | performance improvement |
| `test` | adding or fixing tests |
| `build` | build system, dependencies, Dockerfile |
| `ci` | CI configuration |
| `chore` | maintenance, tooling, housekeeping |
| `revert` | reverts a prior commit |

Scopes for this repo: `amdc`, `amdc-lake`, `crawler`, `silver`, `bronze`, `embedder`, `docker`, `deps`.

Examples:
- `feat(silver): add incremental chunk embedding by bronze_id`
- `fix(crawler): handle Finviz redirect to login page`
- `build(docker): pre-download bge-small weights at image build time`
- `docs: document lakehouse layer schemas in CLAUDE.md`

## Things to avoid

- Committing `data/`, `.venv/`, `.uv-cache/`, model weights, or anything in `__pycache__/`. The `.gitignore` covers these — don't `git add -f` past it.
- Pushing to `main` directly. Promote via PR in the GitHub UI.
- Bumping `torch` / `transformers` past the Intel-Mac ceiling without testing on that hardware.
- Adding new crawl targets without updating `SITES` *and* the corresponding URL patterns in `src/amdc/config.py`.
- Changing embedding model or dim without bumping `embedding_model` / `embedding_dim` in Silver and rebuilding the table.
