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
