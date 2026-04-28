"""Custom data quality check primitives for the Bronze layer."""

from __future__ import annotations

import re

import polars as pl

URL_RE = re.compile(r"\[[^\]]*\]\([^)]+\)|https?://\S+|www\.\S+")

_ERROR_PAGE_RE = re.compile(
    r"(?i)("
    r"oops[,!]?\s*something\s+went\s+wrong"
    r"|temporarily\s+down\s+for\s+maintenance"
    r"|access\s+denied"
    r"|page\s+not\s+found"
    r"|\b404\s+not\s+found\b"
    r"|sign\s+in\s+to\s+(?:read|continue)"
    r"|subscribe\s+to\s+(?:read|continue)"
    r")"
)


def url_junk_ratio(text: str | None) -> float:
    """Fraction of `text` characters covered by URL or markdown-link substrings."""
    if not text:
        return 0.0
    matched = sum(len(m) for m in URL_RE.findall(text))
    return matched / len(text)


def text_passes_junk(text: str | None) -> bool:
    return url_junk_ratio(text) <= 0.5


def text_is_not_error_page(text: str | None) -> bool:
    return text is None or _ERROR_PAGE_RE.search(text) is None


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
