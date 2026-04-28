"""Custom data quality check primitives for the Bronze layer."""

from __future__ import annotations

import re
from typing import Callable, Sequence

import polars as pl

URL_RE = re.compile(r"https?://\S+|www\.\S+")

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


def url_junk_ratio(text: str | None) -> float:
    """Fraction of `text` characters covered by URL-structured substrings."""
    if not text:
        return 0.0
    matched = sum(len(m) for m in URL_RE.findall(text))
    return matched / len(text)


def text_passes_junk(text: str | None) -> bool:
    return url_junk_ratio(text) <= 0.5


def compute_run_drift(
    new_df: pl.DataFrame,
    historical: pl.DataFrame | None,
    *,
    window: int = 3,
    drop_threshold: float = 0.5,
) -> list[dict]:
    """Compare per-source_domain stats in new_df against trailing `window` runs."""
    if new_df.is_empty():
        return []

    current = new_df.group_by("source_domain").agg(
        pl.len().alias("rows"),
        pl.col("text").str.len_chars().mean().alias("mean_text_len"),
        pl.col("source_url").n_unique().alias("n_unique_url"),
    )

    if historical is None or historical.is_empty():
        return [
            {
                "domain": row["source_domain"],
                "metric": "baseline",
                "current": float(row["rows"]),
                "baseline": None,
                "ratio": None,
                "severity": "warn",
                "note": "no prior runs to compare",
            }
            for row in current.iter_rows(named=True)
        ]

    hist_agg = (
        historical.group_by(["source_domain", "ingested_at"])
        .agg(
            pl.len().alias("rows"),
            pl.col("text").str.len_chars().mean().alias("mean_text_len"),
            pl.col("source_url").n_unique().alias("n_unique_url"),
        )
        .sort(["source_domain", "ingested_at"], descending=[False, True])
    )

    findings: list[dict] = []
    for cur_row in current.iter_rows(named=True):
        domain = cur_row["source_domain"]
        domain_hist = hist_agg.filter(pl.col("source_domain") == domain).head(window)
        if domain_hist.is_empty():
            findings.append(
                {
                    "domain": domain,
                    "metric": "baseline",
                    "current": float(cur_row["rows"]),
                    "baseline": None,
                    "ratio": None,
                    "severity": "warn",
                    "note": "no prior runs for this domain",
                }
            )
            continue

        for metric in ("rows", "mean_text_len", "n_unique_url"):
            current_val = cur_row[metric]
            baseline_val = domain_hist.get_column(metric).mean()
            if current_val is None or baseline_val is None or baseline_val == 0:
                continue
            ratio = float(current_val) / float(baseline_val)
            if ratio < drop_threshold:
                findings.append(
                    {
                        "domain": domain,
                        "metric": metric,
                        "current": float(current_val),
                        "baseline": float(baseline_val),
                        "ratio": ratio,
                        "severity": "warn",
                        "note": f"{metric} dropped to {ratio:.0%} of trailing-{window} mean",
                    }
                )

    recent_runs = (
        historical.select("ingested_at")
        .unique()
        .sort("ingested_at", descending=True)
        .head(window)
    )
    recent_ts = recent_runs.get_column("ingested_at").to_list()
    if len(recent_ts) >= window:
        domains_per_run = [
            set(
                historical.filter(pl.col("ingested_at") == ts)
                .get_column("source_domain")
                .unique()
                .to_list()
            )
            for ts in recent_ts
        ]
        always_present = (
            set.intersection(*domains_per_run) if domains_per_run else set()
        )
        current_domains = set(current.get_column("source_domain").to_list())
        for missing in sorted(always_present - current_domains):
            findings.append(
                {
                    "domain": missing,
                    "metric": "presence",
                    "current": 0.0,
                    "baseline": None,
                    "ratio": 0.0,
                    "severity": "warn",
                    "note": f"absent from current run; present in last {window} runs",
                }
            )

    return findings


def compute_null_counts(df: pl.DataFrame) -> dict[str, int]:
    if df.is_empty():
        return {col: 0 for col in df.columns}
    nulls = df.null_count().row(0, named=True)
    return {col: int(value) for col, value in nulls.items()}


def find_title_duplicate_clusters(
    df: pl.DataFrame,
    embed_fn: EmbedFn,
    *,
    threshold: float = 0.95,
    min_title_chars: int = 8,
) -> list[dict]:
    """Cluster rows whose title embeddings have cosine similarity >= threshold.

    Embeddings are assumed L2-normalized (BGE output is normalized in `BgeM3Embedder`),
    so cosine similarity reduces to a dot product.
    """
    if df.is_empty() or "title" not in df.columns or "bronze_id" not in df.columns:
        return []

    candidates = df.select(["bronze_id", "title"]).filter(
        pl.col("title").is_not_null()
        & (pl.col("title").str.len_chars() >= min_title_chars)
    )
    if candidates.height < 2:
        return []

    bronze_ids = candidates.get_column("bronze_id").to_list()
    titles = candidates.get_column("title").to_list()
    vectors = embed_fn(titles)
    if not vectors:
        return []

    n = len(vectors)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pair_sims: dict[tuple[int, int], float] = {}
    for i in range(n):
        vi = vectors[i]
        for j in range(i + 1, n):
            vj = vectors[j]
            sim = sum(a * b for a, b in zip(vi, vj))
            if sim >= threshold:
                union(i, j)
                pair_sims[(i, j)] = sim

    if not pair_sims:
        return []

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    clusters: list[dict] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        max_sim = max(
            sim for (i, j), sim in pair_sims.items() if i in members and j in members
        )
        clusters.append(
            {
                "size": len(members),
                "max_similarity": float(max_sim),
                "bronze_ids": [bronze_ids[m] for m in members],
                "titles": [titles[m] for m in members],
            }
        )
    clusters.sort(key=lambda c: (-c["size"], -c["max_similarity"]))
    return clusters
