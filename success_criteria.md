# SignalForge / AMDC — Success Criteria

A snapshot of what v1 of the Adaptive Market Data Crawler is, what it
must do to be considered "done," what it intentionally does not do, and
why the chosen tech stack leaves the door open to scale without a
rewrite.

## Purpose

AMDC is a personal research tool that scrapes financial news from a
small set of public sources (CNBC, Investing.com, Finviz), promotes the
raw pages through a Bronze → Silver Delta lakehouse with 384-dim
BAAI/bge-small-en-v1.5 embeddings, and exposes the result through a
Streamlit cosine-similarity search.

Success for v1 means: a single user can ask a question in the UI and
get back relevant, recently-crawled article chunks with full provenance
back to the source URL — reproducibly, on a laptop, without manual
plumbing between stages.

## What v1 must do (success criteria, met today)

- [x] **End-to-end happy path works.** `amdc <query>` → `amdc-lake
      bronze-backfill` → `amdc-lake silver-build` → Streamlit search
      returns ranked chunks. The `run_amdc.py` orchestrator and the
      Streamlit auto-refresh path stitch these together for the UI user.
- [x] **Identity is deterministic.** `bronze_id`, `page_id`, and
      `chunk_id` are SHA256 composites built via `amdc_lake.ids.sha256_id`
      from stable inputs. Re-running any stage on the same input
      produces the same IDs.
- [x] **Bronze writes are idempotent.** Append mode skips `bronze_id`s
      already present in the Delta table; a re-run never duplicates
      rows.
- [x] **Bronze data quality is gated.** Pandera validates rows on
      ingest; failures land in `bronze/scrapes_quarantine` and a
      structured row is appended to `_quality/runs`.
- [x] **Silver embeddings are incremental.** `silver-build` anti-joins
      on `bronze_id` and only embeds new pages/chunks; `--rebuild` is
      available as an explicit escape hatch.
- [x] **Pipeline runs are auditable.** Every stage records start/end,
      duration, row counts, status, and error to `_pipeline/runs` via
      `observability.record_run`.
- [x] **Reproducible build and environment.** `uv.lock` pins all Python
      deps; the Dockerfile pre-bakes Chromium and the BGE-small weights
      so the runtime image is hermetic.
- [x] **CI keeps the codebase honest.** `.github/workflows/ci.yml` runs
      ruff lint + format check + pytest on every push/PR.

## Known limitations (accepted for v1)

| Area | Limitation | Why it is acceptable for v1 |
|---|---|---|
| Crawl reliability | No retries, no per-domain rate limit, no global crawl timeout. Transient network errors silently drop pages. | Single user, low-frequency manual crawls against three public sources. The cost of an occasional missed page is "search again," not data corruption. |
| Data quality scope | Pandera checks gate Bronze only; Silver/Gold have no equivalent. Embedding norm and dimension are not validated post-generation. | The embedder is pinned to one model with a known output shape. A bad batch would be visible in search quality immediately. |
| Operational maturity | No scheduled crawls, no alerting, stdout-only logging, single-container deployment, no auth. | The deployment target is one user, one laptop. Ops investment would be premature optimization. |
| Model constraints | Intel-Mac PyTorch ceiling (torch ≤ 2.2.2, transformers < 5) caps embedder choice; no CUDA OOM recovery. | Documented in `CLAUDE.md` and in agent memory. BGE-small is sufficient for the corpus size and query style. |
| Test coverage | ~15 lakehouse tests cover IDs/Bronze/Silver and one real embedding round-trip. Crawler logic is exercised end-to-end via the UI rather than in unit tests. | The crawler is the most external-dependency-heavy component; integration-style validation through the UI is the most honest signal that it works. |

These are not bugs; they are scope boundaries.

## Why the stack is built for scale

Single-user-on-a-laptop is a deployment choice, not a stack choice.
Every load-bearing component was chosen from the production
data-engineering shelf, so the path to "shared service" is additive
rather than a rewrite.

| Layer | Choice | Why it scales |
|---|---|---|
| Storage format | **Delta Lake** (`deltalake>=0.25`) | ACID transactions, schema evolution, time travel, Z-ordering. Same format Databricks uses at PB scale. `paths.py` already abstracts the lake root — swapping `./data/lakehouse` for `s3://...` requires no code change. |
| Compute on tables | **Polars + PyArrow** | Columnar, multi-threaded, zero-copy with Arrow. The same dataframe code runs on a single parquet or partitioned multi-GB tables. Polars' lazy API is the natural step for query pruning. |
| Embedding model | **BAAI/bge-small-en-v1.5** | Small enough for CPU today, but `embedder.py` is a thin wrapper over HF `AutoModel`. Switching to a larger BGE variant or a hosted embedding API is a single-class change; `embedding_dim` and `embedding_model` are already columns on the Silver table. |
| Crawl engine | **crawl4ai + Playwright** | Async, with concurrency knobs and best-first/BM25 strategies that mirror what a distributed crawler would use. Horizontal scaling is "more workers writing into the same Bronze table" — `bronze_id` dedup handles overlap automatically. |
| Identity / dedup | **SHA256 composite IDs** (`ids.py`) | Deterministic, content-addressed, partition-friendly. The same discipline production lakehouses use for idempotent writes. |
| Pipeline orchestration | **Typer CLI commands** (`amdc`, `amdc-lake`) | Each stage is an independent, idempotent CLI call — exactly the contract Airflow / Prefect / Argo expect. Wrapping today's commands in DAG tasks is glue, not a port. |
| Observability substrate | **`_pipeline/runs` + `_quality/runs` Delta tables** | Structured, append-only audit logs already exist. A Grafana or Looker dashboard, or an alerting rule, becomes "query a Delta table" — no new data plumbing. |
| Packaging | **`uv` + locked `pyproject.toml`, two-package layout** | Reproducible builds; clean module boundaries. The crawler and the lakehouse are already separable services and can be deployed on different schedules and hardware without code surgery. |
| Containerization | **Docker with model + Chromium pre-baked** | Image is heavy (~3–4 GB) but immutable. The same image runs as a Streamlit pod, a Kubernetes CronJob, or a batch worker — only the entrypoint changes. |
| Language | **Python 3.12, typed throughout** | Aligns with the current production ecosystem. The only ceiling (Intel-Mac torch pin) is local-dev only, not a deployment constraint. |

Nothing in the stack would have to be replaced to support many users,
scheduled crawls, or much larger corpora. The work would be
**operational** — orchestrator, object storage, monitoring, auth — layered
on top of the same primitives.

## Forward path (post-v1)

Eight additive improvements on top of the v1 substrate, in priority
order. None of them require rethinking the architecture.

1. **Retries + per-domain rate limits in the crawler** — wrap the
   per-page fetch in tenacity-style backoff and add an
   `asyncio.Semaphore` keyed by `source_domain`. Highest ROI; directly
   reduces silent data loss.
2. **Crawl-level timeout** — `asyncio.wait_for` around `crawl_all` so a
   single hung site cannot hang the whole run.
3. **Silver-side data quality checks** — mirror the Bronze Pandera
   pattern: assert embedding norm > 0, no NaNs, dim matches model
   constant. Quarantine the same way Bronze does.
4. **Partition Bronze + Silver by `crawled_at` date** — cheap to add
   while tables are small; unlocks pruning when they aren't.
5. **Scheduled crawls** — a GitHub Actions cron or a small Prefect flow
   running `amdc` + `bronze-backfill` + `silver-build` on a schedule,
   piping results into `_pipeline/runs`.
6. **Surface `_pipeline/runs` errors in the UI** — a banner in the Search
   tab when the latest pipeline run errored. Cheaper than real alerting,
   90% of the value for a single-user tool.
7. **Define a Gold layer** — even one table (e.g. daily per-domain
   article counts) gives downstream analytics a stable contract instead
   of every consumer re-aggregating Silver.
8. **Crawler unit tests** — cover `extract.py` parsing edge cases (date
   regex fallbacks, missing titles, truncation). Pure functions, easy to
   test deterministically.
