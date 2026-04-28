"""Light Streamlit UI for the AMDC self-updating RAG orchestrator."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import streamlit as st

from run_amdc import BgeM3Embedder, orchestrate_query

DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_ARTICLES = 10
DEFAULT_TOP_K = 20


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


def main() -> None:
    st.set_page_config(page_title="AMDC Search", layout="wide")
    st.title("AMDC Search")
    st.write(
        "Search the AMDC lakehouse for market articles. When cached matches are "
        "thin, the orchestrator can crawl fresh sources, append Bronze data, "
        "rebuild Silver embeddings, and return updated results."
    )

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


if __name__ == "__main__":
    main()
