from __future__ import annotations

from pathlib import Path

import polars as pl
from deltalake import DeltaTable

from amdc_lake.bronze import BRONZE_SCHEMA
from amdc_lake.paths import bronze_scrapes_quarantine_path
from amdc_lake.quality.checks import (
    compute_null_counts,
    compute_run_drift,
    text_is_not_error_page,
    find_title_duplicate_clusters,
    text_passes_junk,
    url_junk_ratio,
)
from amdc_lake.quality.quarantine import write_quarantine
from amdc_lake.quality.runner import run_bronze_checks


def _bronze_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=BRONZE_SCHEMA)


def _good_row(**overrides) -> dict:
    base = {
        "bronze_id": "bronze_" + ("a" * 64),
        "title": "Headline",
        "date_published": "2026-04-28",
        "text": "Markets watch the Federal Reserve decision today and tomorrow.",
        "source_url": "https://cnbc.com/fed",
        "source_domain": "cnbc.com",
        "relevance_score": 1.0,
        "query": "Fed rate decision",
        "crawled_at": "2026-04-28T00:00:00+00:00",
        "source_file": "f.parquet",
        "ingested_at": "2026-04-28T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def test_url_junk_ratio_handles_empty_and_pure_url() -> None:
    assert url_junk_ratio(None) == 0.0
    assert url_junk_ratio("") == 0.0
    assert url_junk_ratio("https://example.com") > 0.9
    assert text_passes_junk("real article body with some words") is True
    assert text_passes_junk("https://a.com https://b.com https://c.com") is False


def test_url_junk_ratio_counts_markdown_link_residue() -> None:
    nav = "[Skip Navigation](https://cnbc.com/x) [Halftime Report](https://cnbc.com/y)"
    assert url_junk_ratio(nav) > 0.9
    assert text_passes_junk(nav) is False


def test_text_is_not_error_page_flags_known_sentinels() -> None:
    assert text_is_not_error_page(None) is True
    assert text_is_not_error_page("real article body with some words") is True
    assert text_is_not_error_page("Oops, something went wrong") is False
    assert (
        text_is_not_error_page("We are temporarily down for maintenance, please retry")
        is False
    )
    assert text_is_not_error_page("Sign in to read more of this article") is False
    assert text_is_not_error_page("Subscribe to continue reading") is False
    assert text_is_not_error_page("404 Not Found - the resource is missing") is False


def test_compute_run_drift_returns_baseline_when_no_history() -> None:
    new_df = _bronze_frame([_good_row()])

    findings = compute_run_drift(new_df, historical=None)

    assert len(findings) == 1
    assert findings[0]["domain"] == "cnbc.com"
    assert findings[0]["metric"] == "baseline"
    assert findings[0]["note"] == "no prior runs to compare"


def test_compute_run_drift_flags_row_count_drop() -> None:
    rows = []
    counter = 0
    for ts in (
        "2026-04-21T00:00:00+00:00",
        "2026-04-22T00:00:00+00:00",
        "2026-04-23T00:00:00+00:00",
    ):
        for _ in range(5):
            rows.append(
                _good_row(
                    bronze_id=f"bronze_{counter:064x}",
                    source_url=f"https://cnbc.com/{counter}",
                    ingested_at=ts,
                )
            )
            counter += 1
    historical = _bronze_frame(rows)
    new_df = _bronze_frame([_good_row()])

    findings = compute_run_drift(new_df, historical)

    rows_findings = [f for f in findings if f["metric"] == "rows"]
    assert rows_findings, f"expected a rows-drop finding, got {findings}"
    assert rows_findings[0]["ratio"] < 0.5


def test_compute_run_drift_detects_missing_domain_present_in_history() -> None:
    historical = _bronze_frame(
        [
            _good_row(
                bronze_id=f"bronze_{i:064x}",
                source_domain=domain,
                source_url=f"https://{domain}/{i}",
                ingested_at=ts,
            )
            for i, ts in enumerate(
                [
                    "2026-04-21T00:00:00+00:00",
                    "2026-04-22T00:00:00+00:00",
                    "2026-04-23T00:00:00+00:00",
                ]
            )
            for domain in ("cnbc.com", "finviz.com")
        ]
    )
    new_df = _bronze_frame([_good_row()])

    findings = compute_run_drift(new_df, historical)

    presence = [f for f in findings if f["metric"] == "presence"]
    assert any(f["domain"] == "finviz.com" for f in presence)


def test_run_bronze_checks_passes_for_clean_row(tmp_path: Path) -> None:
    df = _bronze_frame([_good_row()])

    result = run_bronze_checks(df, tmp_path)

    assert result.rows_in == 1
    assert result.rows_failed == 0
    assert result.rows_passed == 1
    assert result.status in {"pass", "warn"}
    assert result.run_id.startswith("qrun_")


def test_write_quarantine_appends_failures_with_reasons(tmp_path: Path) -> None:
    df = _bronze_frame([_good_row(text="Short.")])
    result = run_bronze_checks(df, tmp_path)

    target = write_quarantine(result.failures, tmp_path)

    assert target == bronze_scrapes_quarantine_path(tmp_path)
    quarantined = pl.from_arrow(DeltaTable(str(target)).to_pyarrow_table())
    assert quarantined.height == 1
    assert quarantined.select("_quality_run_id").item() == result.run_id
    reasons = quarantined.get_column("_failure_reasons").to_list()[0]
    assert any("text" in r for r in reasons)


def test_write_quarantine_returns_none_for_empty_failures(tmp_path: Path) -> None:
    empty = _bronze_frame([]).with_columns(
        pl.lit(None, dtype=pl.List(pl.Utf8)).alias("_failure_reasons"),
        pl.lit(None, dtype=pl.Utf8).alias("_quality_run_id"),
    )
    assert write_quarantine(empty, tmp_path) is None


def test_run_bronze_checks_rejects_text_below_min_length(tmp_path: Path) -> None:
    df = _bronze_frame([_good_row(text="Short.")])

    result = run_bronze_checks(df, tmp_path)

    assert result.rows_failed == 1
    assert result.status == "fail"
    assert any(c["column"] == "text" for c in result.check_summary)


def test_run_bronze_checks_rejects_error_page_text(tmp_path: Path) -> None:
    df = _bronze_frame(
        [_good_row(text="Oops, something went wrong " + "padding " * 10)]
    )

    result = run_bronze_checks(df, tmp_path)

    assert result.rows_failed == 1
    assert result.status == "fail"
    assert any(
        c["column"] == "text" and "sentinel" in c["check"] for c in result.check_summary
    )


def test_compute_null_counts_reports_per_column() -> None:
    df = _bronze_frame(
        [
            _good_row(bronze_id=f"bronze_{0:064x}", date_published=None, title=None),
            _good_row(bronze_id=f"bronze_{1:064x}", date_published="2026-04-28"),
        ]
    )

    counts = compute_null_counts(df)

    assert counts["title"] == 1
    assert counts["date_published"] == 1
    assert counts["text"] == 0
    assert counts["bronze_id"] == 0


def test_find_title_duplicate_clusters_groups_high_cosine() -> None:
    df = _bronze_frame(
        [
            _good_row(bronze_id=f"bronze_{0:064x}", title="Fed raises rates today"),
            _good_row(bronze_id=f"bronze_{1:064x}", title="Fed raises rates today!"),
            _good_row(bronze_id=f"bronze_{2:064x}", title="Apple unveils new chip"),
        ]
    )

    def stub_embed(titles):
        mapping = {
            "Fed raises rates today": [1.0, 0.0, 0.0],
            "Fed raises rates today!": [0.99, 0.141, 0.0],
            "Apple unveils new chip": [0.0, 1.0, 0.0],
        }
        return [mapping[t] for t in titles]

    clusters = find_title_duplicate_clusters(df, stub_embed, threshold=0.95)

    assert len(clusters) == 1
    assert clusters[0]["size"] == 2
    assert set(clusters[0]["bronze_ids"]) == {
        f"bronze_{0:064x}",
        f"bronze_{1:064x}",
    }
    assert clusters[0]["max_similarity"] >= 0.95


def test_find_title_duplicate_clusters_returns_empty_when_below_threshold() -> None:
    df = _bronze_frame(
        [
            _good_row(bronze_id=f"bronze_{0:064x}", title="Fed raises rates today"),
            _good_row(bronze_id=f"bronze_{1:064x}", title="Apple unveils new chip"),
        ]
    )

    def stub_embed(titles):
        mapping = {
            "Fed raises rates today": [1.0, 0.0, 0.0],
            "Apple unveils new chip": [0.0, 1.0, 0.0],
        }
        return [mapping[t] for t in titles]

    assert find_title_duplicate_clusters(df, stub_embed, threshold=0.95) == []


def test_run_bronze_checks_records_nulls_and_duplicates(tmp_path: Path) -> None:
    df = _bronze_frame(
        [
            _good_row(
                bronze_id=f"bronze_{0:064x}",
                title="Fed raises rates today",
                date_published=None,
            ),
            _good_row(
                bronze_id=f"bronze_{1:064x}",
                title="Fed raises rates today!",
            ),
        ]
    )

    def stub_embed(titles):
        mapping = {
            "Fed raises rates today": [1.0, 0.0, 0.0],
            "Fed raises rates today!": [0.99, 0.141, 0.0],
        }
        return [mapping[t] for t in titles]

    result = run_bronze_checks(
        df, tmp_path, embed_fn=stub_embed, duplicate_threshold=0.95
    )

    assert result.null_counts["date_published"] == 1
    assert result.null_counts["text"] == 0
    assert len(result.duplicate_clusters) == 1
    assert result.duplicate_clusters[0]["size"] == 2
    assert result.status == "warn"
