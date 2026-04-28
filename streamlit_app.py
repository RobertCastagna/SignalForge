"""Light Streamlit UI for the AMDC self-updating RAG orchestrator."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import polars as pl
import streamlit as st

from amdc_lake.paths import DEFAULT_LAKE_DIR, quality_runs_path
from run_amdc import BgeM3Embedder, orchestrate_query

DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_ARTICLES = 10
DEFAULT_TOP_K = 20
_RAW_DQ_COLUMNS = [
    "Raw Check Summary",
    "Raw Drift Report",
    "Raw Null Counts",
    "Raw Duplicate Clusters",
]


class _ListLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


@contextmanager
def _capture_info_logs() -> Iterator[_ListLogHandler]:
    handler = _ListLogHandler()
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(min(previous_level, logging.INFO))
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


@st.cache_resource(show_spinner="Loading embedding model...")
def _get_embedder() -> BgeM3Embedder:
    return BgeM3Embedder()


def _render_results(result) -> None:
    st.caption(
        f"Matches: {result.final_matches} | Threshold: {result.threshold:.2f} | "
        f"Minimum articles: {result.min_articles}"
    )
    st.dataframe(
        result.hits,
        hide_index=True,
        use_container_width=True,
        column_config={
            "similarity": st.column_config.NumberColumn(
                "Similarity",
                format="%.3f",
            ),
            "source_url": st.column_config.LinkColumn("Source"),
            "text": st.column_config.TextColumn("Text", width="large"),
        },
    )


def _parse_json_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _summarize_checks(checks: Any) -> str:
    if not isinstance(checks, list) or not checks:
        return "None"

    parts = []
    for item in checks[:3]:
        if not isinstance(item, dict):
            continue
        column = item.get("column") or "row"
        check = item.get("check") or "check"
        failed = item.get("failed")
        count = f"{failed} failed" if failed is not None else "failed"
        parts.append(f"{column}: {count} ({check})")

    if not parts:
        return f"{len(checks)} finding(s)"
    if len(checks) > len(parts):
        parts.append(f"+{len(checks) - len(parts)} more")
    return "; ".join(parts)


def _summarize_drift(findings: Any) -> str:
    if not isinstance(findings, list) or not findings:
        return "None"

    parts = []
    for item in findings[:3]:
        if not isinstance(item, dict):
            continue
        domain = item.get("domain") or "domain"
        metric = item.get("metric") or "metric"
        note = item.get("note")
        parts.append(f"{domain}: {metric}" + (f" ({note})" if note else ""))

    if not parts:
        return f"{len(findings)} finding(s)"
    if len(findings) > len(parts):
        parts.append(f"+{len(findings) - len(parts)} more")
    return "; ".join(parts)


def _summarize_nulls(null_counts: Any) -> str:
    if not isinstance(null_counts, dict) or not null_counts:
        return "None"

    nonzero = [
        f"{column}: {count}"
        for column, count in sorted(null_counts.items())
        if isinstance(count, (int, float)) and count > 0
    ]
    return "; ".join(nonzero[:5]) + (
        f"; +{len(nonzero) - 5} more" if len(nonzero) > 5 else ""
    ) if nonzero else "None"


def _summarize_duplicates(clusters: Any) -> str:
    if not isinstance(clusters, list) or not clusters:
        return "None"

    sizes = [
        int(item.get("size", 0))
        for item in clusters
        if isinstance(item, dict) and item.get("size") is not None
    ]
    similarities = [
        float(item.get("max_similarity", 0.0))
        for item in clusters
        if isinstance(item, dict) and item.get("max_similarity") is not None
    ]
    max_size = max(sizes) if sizes else 0
    max_similarity = max(similarities) if similarities else 0.0
    return (
        f"{len(clusters)} cluster(s); max size {max_size}; "
        f"max similarity {max_similarity:.3f}"
    )


def _empty_quality_display() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "Run Started": pl.Utf8,
            "Run Finished": pl.Utf8,
            "Layer": pl.Utf8,
            "Status": pl.Utf8,
            "Rows In": pl.Int64,
            "Rows Passed": pl.Int64,
            "Rows Failed": pl.Int64,
            "Checks Failed": pl.Utf8,
            "Drift Findings": pl.Utf8,
            "Null Columns": pl.Utf8,
            "Duplicate Clusters": pl.Utf8,
            "Run ID": pl.Utf8,
            "Lake Dir": pl.Utf8,
            "Raw Check Summary": pl.Utf8,
            "Raw Drift Report": pl.Utf8,
            "Raw Null Counts": pl.Utf8,
            "Raw Duplicate Clusters": pl.Utf8,
        }
    )


def _format_quality_runs(runs: pl.DataFrame) -> pl.DataFrame:
    if runs.is_empty():
        return _empty_quality_display()

    rows = []
    for row in runs.sort("started_at", descending=True).iter_rows(named=True):
        checks = _parse_json_value(row.get("check_summary"), [])
        drift = _parse_json_value(row.get("drift_report"), [])
        nulls = _parse_json_value(row.get("null_counts"), {})
        duplicates = _parse_json_value(row.get("duplicate_clusters"), [])
        rows.append(
            {
                "Run Started": row.get("started_at"),
                "Run Finished": row.get("finished_at"),
                "Layer": row.get("layer"),
                "Status": (row.get("status") or "").upper(),
                "Rows In": row.get("rows_in"),
                "Rows Passed": row.get("rows_passed"),
                "Rows Failed": row.get("rows_failed"),
                "Checks Failed": _summarize_checks(checks),
                "Drift Findings": _summarize_drift(drift),
                "Null Columns": _summarize_nulls(nulls),
                "Duplicate Clusters": _summarize_duplicates(duplicates),
                "Run ID": row.get("run_id"),
                "Lake Dir": row.get("lake_dir"),
                "Raw Check Summary": row.get("check_summary"),
                "Raw Drift Report": row.get("drift_report"),
                "Raw Null Counts": row.get("null_counts"),
                "Raw Duplicate Clusters": row.get("duplicate_clusters"),
            }
        )
    return pl.DataFrame(rows)


def _read_quality_runs(lake_dir: Path = DEFAULT_LAKE_DIR) -> pl.DataFrame:
    target = quality_runs_path(lake_dir)
    if not target.exists():
        return pl.DataFrame()
    try:
        return pl.read_delta(str(target))
    except Exception:
        logging.getLogger(__name__).exception("failed to read quality runs from %s", target)
        return pl.DataFrame()


def _process_flow_dot() -> str:
    return """
digraph {
    graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.45", ranksep="0.55"];
    node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#64748b", fontname="Helvetica", fontsize=11, margin="0.12,0.08"];
    edge [color="#64748b", arrowsize=0.7, fontname="Helvetica", fontsize=10];

    query [label="User Query"];
    search [label="Silver Search"];
    cache [label="Cache Check"];
    crawl [label="Optional Crawl"];
    bronze [label="Bronze Quality"];
    silver [label="Silver Embeddings"];
    results [label="Results"];

    query -> search -> cache;
    cache -> results [label="enough matches"];
    cache -> crawl [label="thin cache"];
    crawl -> bronze -> silver -> results;
}
"""


def _render_header() -> None:
    st.title("Adaptive Market Data Crawler")
    st.write(
        "Search market articles in the AMDC lakehouse. When cached matches are "
        "thin, the orchestrator can crawl fresh sources, run Bronze data "
        "quality, rebuild Silver embeddings, and return updated results."
    )
    st.graphviz_chart(_process_flow_dot(), use_container_width=True)


def _render_search_tab() -> None:
    with st.form("amdc_search"):
        query = st.text_input("Search", max_chars=200)
        no_crawl = st.checkbox("No crawl", value=False)
        threshold = st.slider(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_THRESHOLD,
            step=0.01,
        )
        submitted = st.form_submit_button("Search")

    if not submitted:
        return

    query = query.strip()
    if not query:
        st.warning("Enter a search query.")
        return

    with st.spinner("Running AMDC query..."):
        with _capture_info_logs() as logs:
            result = orchestrate_query(
                query,
                threshold=threshold,
                min_articles=DEFAULT_MIN_ARTICLES,
                top_k=DEFAULT_TOP_K,
                no_crawl=no_crawl,
                embedder=_get_embedder(),
            )

    st.info(result.status_message)

    if result.crawled:
        with st.expander("Crawler logs", expanded=False):
            st.code("\n".join(logs.messages) or "No logs captured.", language="text")

    _render_results(result)


def _render_quality_tab() -> None:
    st.subheader("Data Quality Runs")
    formatted = _format_quality_runs(_read_quality_runs())
    if formatted.is_empty():
        st.info("No data quality runs found at data/lakehouse/_quality/runs.")
        return

    statuses = formatted.get_column("Status").unique().sort().to_list()
    with st.sidebar:
        st.header("Data Quality")
        selected_statuses = st.multiselect("Status", statuses, default=statuses)
        max_rows = st.selectbox("Max rows", [10, 25, 50, 100], index=1)
        show_raw = st.checkbox("Show raw JSON detail columns", value=False)

    display = formatted.filter(pl.col("Status").is_in(selected_statuses)).head(max_rows)
    if not show_raw:
        display = display.drop(_RAW_DQ_COLUMNS)

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rows In": st.column_config.NumberColumn("Rows In", format="%d"),
            "Rows Passed": st.column_config.NumberColumn("Rows Passed", format="%d"),
            "Rows Failed": st.column_config.NumberColumn("Rows Failed", format="%d"),
            "Run ID": st.column_config.TextColumn("Run ID", width="medium"),
        },
    )


def main() -> None:
    st.set_page_config(page_title="Adaptive Market Data Crawler", layout="wide")
    _render_header()

    search_tab, dq_tab = st.tabs(["Search", "Data Quality"])
    with search_tab:
        _render_search_tab()
    with dq_tab:
        _render_quality_tab()


if __name__ == "__main__":
    main()
