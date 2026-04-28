from __future__ import annotations

from pathlib import Path
import sys

import polars as pl
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_amdc as run_module


def _hits(count: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "row_id": [f"page-{i}" for i in range(count)],
            "similarity": [0.75] * count,
            "title": [f"Title {i}" for i in range(count)],
            "source_url": [f"https://example.com/{i}" for i in range(count)],
            "crawled_at": ["2026-04-28T00:00:00+00:00"] * count,
            "text": ["Body"] * count,
        },
        schema=run_module._EMPTY_RESULT_SCHEMA,
    )


def test_orchestrate_query_cache_hit_does_not_crawl(monkeypatch) -> None:
    calls: list[str] = []

    def fake_search(query, lake_dir, threshold, top_k, embedder):
        calls.append(query)
        return _hits(2)

    def fail_pipeline(query, data_dir, lake_dir):
        raise AssertionError("pipeline should not run")

    monkeypatch.setattr(run_module, "_search", fake_search)
    monkeypatch.setattr(run_module, "_trigger_pipeline", fail_pipeline)

    result = run_module.orchestrate_query(
        "markets",
        min_articles=2,
        embedder=object(),
    )

    assert calls == ["markets"]
    assert result.crawled is False
    assert result.initial_matches == 2
    assert result.final_matches == 2
    assert "cache hit" in result.status_message


def test_orchestrate_query_cache_miss_crawls_and_requeries(monkeypatch) -> None:
    search_results = iter([_hits(0), _hits(3)])
    pipeline_queries: list[str] = []

    def fake_search(query, lake_dir, threshold, top_k, embedder):
        return next(search_results)

    def fake_pipeline(query, data_dir, lake_dir):
        pipeline_queries.append(query)

    monkeypatch.setattr(run_module, "_search", fake_search)
    monkeypatch.setattr(run_module, "_trigger_pipeline", fake_pipeline)

    result = run_module.orchestrate_query(
        "semis",
        min_articles=2,
        embedder=object(),
    )

    assert pipeline_queries == ["semis"]
    assert result.crawled is True
    assert result.initial_matches == 0
    assert result.final_matches == 3
    assert "crawling" in result.status_message


def test_orchestrate_query_no_crawl_returns_thin_cache(monkeypatch) -> None:
    def fake_search(query, lake_dir, threshold, top_k, embedder):
        return _hits(1)

    def fail_pipeline(query, data_dir, lake_dir):
        raise AssertionError("pipeline should not run")

    monkeypatch.setattr(run_module, "_search", fake_search)
    monkeypatch.setattr(run_module, "_trigger_pipeline", fail_pipeline)

    result = run_module.orchestrate_query(
        "rates",
        min_articles=10,
        no_crawl=True,
        embedder=object(),
    )

    assert result.crawled is False
    assert result.initial_matches == 1
    assert result.final_matches == 1
    assert "--no-crawl set" in result.status_message


def test_cli_uses_orchestrator(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_orchestrate(query, **kwargs):
        captured["query"] = query
        captured.update(kwargs)
        return run_module.QueryResult(
            query=query,
            hits=_hits(1),
            initial_matches=1,
            crawled=False,
            threshold=kwargs["threshold"],
            min_articles=kwargs["min_articles"],
            top_k=kwargs["top_k"],
            status_message="cache hit",
        )

    monkeypatch.setattr(run_module, "orchestrate_query", fake_orchestrate)

    result = CliRunner().invoke(
        run_module.app,
        [
            "markets",
            "--threshold",
            "0.6",
            "--min-articles",
            "1",
            "--top-k",
            "5",
            "--data-dir",
            "custom-data",
            "--lake-dir",
            "custom-lake",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["query"] == "markets"
    assert captured["threshold"] == 0.6
    assert captured["min_articles"] == 1
    assert captured["top_k"] == 5
    assert captured["data_dir"] == Path("custom-data")
    assert captured["lake_dir"] == Path("custom-lake")
    assert "matches=1" in result.output


def test_streamlit_app_import_has_no_runtime_side_effects() -> None:
    import streamlit_app

    assert streamlit_app.DEFAULT_MIN_ARTICLES == 10


def test_format_quality_runs_sorts_and_humanizes_json() -> None:
    import streamlit_app

    raw = pl.DataFrame(
        {
            "run_id": ["old", "new"],
            "layer": ["bronze", "bronze"],
            "started_at": [
                "2026-04-28T10:00:00+00:00",
                "2026-04-28T12:00:00+00:00",
            ],
            "finished_at": [
                "2026-04-28T10:00:02+00:00",
                "2026-04-28T12:00:02+00:00",
            ],
            "rows_in": [10, 12],
            "rows_passed": [9, 10],
            "rows_failed": [1, 2],
            "status": ["pass", "warn"],
            "check_summary": [
                "[]",
                '[{"column":"text","check":"min_length","failed":2}]',
            ],
            "drift_report": [
                "[]",
                '[{"domain":"finviz.com","metric":"rows","note":"rows dropped"}]',
            ],
            "null_counts": [
                '{"title":0,"date_published":0}',
                '{"title":1,"date_published":2}',
            ],
            "duplicate_clusters": [
                "[]",
                '[{"size":3,"max_similarity":0.987}]',
            ],
            "lake_dir": ["data/lakehouse", "data/lakehouse"],
        }
    )

    formatted = streamlit_app._format_quality_runs(raw)

    assert formatted.get_column("Run ID").to_list() == ["new", "old"]
    newest = formatted.row(0, named=True)
    assert newest["Status"] == "WARN"
    assert newest["Checks Failed"] == "text: 2 failed (min_length)"
    assert "finviz.com: rows" in newest["Drift Findings"]
    assert newest["Null Columns"] == "date_published: 2; title: 1"
    assert newest["Duplicate Clusters"] == (
        "1 cluster(s); max size 3; max similarity 0.987"
    )


def test_quality_json_helpers_tolerate_empty_and_malformed_values() -> None:
    import streamlit_app

    assert streamlit_app._parse_json_value("", []) == []
    assert streamlit_app._parse_json_value("not-json", {}) == {}
    assert streamlit_app._summarize_checks([]) == "None"
    assert streamlit_app._summarize_drift([]) == "None"
    assert streamlit_app._summarize_nulls({"title": 0}) == "None"
    assert streamlit_app._summarize_duplicates([]) == "None"


def test_read_quality_runs_missing_table_returns_empty(tmp_path: Path) -> None:
    import streamlit_app

    assert streamlit_app._read_quality_runs(tmp_path).is_empty()
