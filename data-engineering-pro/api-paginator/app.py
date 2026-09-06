from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from paginator import MockAPI, paginate

st.set_page_config(page_title="API Pagination Extractor", page_icon="📡", layout="wide")
st.title("📡 API Pagination Extractor")
st.caption("One extractor, four pagination styles — offset, page, cursor, link — with 429 retry/backoff built in.")

with st.sidebar:
    st.header("Mock API")
    n_records = st.slider("Records behind the API", 100, 5000, 950, step=50)
    page_size = st.slider("Page size", 10, 500, 100, step=10)
    flaky = st.slider("Send a 429 every Nth call (0=never)", 0, 10, 7)
    strategies = st.multiselect("Strategies to run", ["offset", "page", "cursor", "link"],
                                default=["offset", "page", "cursor", "link"])

if st.button("Extract", type="primary") and strategies:
    rows = []
    per_page_series = {}
    for strat in strategies:
        api = MockAPI(n_records, flaky_429_every=flaky)
        items, s = paginate(api.fetch, strat, page_size=page_size)
        ids = [i["id"] for i in items]
        complete = len(items) == n_records and ids == sorted(set(ids))
        rows.append({"Strategy": strat, "Items": s.items, "Requests": s.requests,
                     "Retries": s.retries, "429s": s.rate_limited,
                     "Complete & deduped": "✅" if complete else "❌"})
        per_page_series[strat] = s.pages

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5.5, 3))
        ax.bar(df["Strategy"], df["Requests"], color="#457b9d", label="requests")
        ax.bar(df["Strategy"], df["Retries"], color="#e76f51", label="retries")
        ax.set_ylabel("HTTP calls")
        ax.set_title("Requests and retries per strategy")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(5.5, 3))
        for strat, pages in per_page_series.items():
            ax2.plot(range(1, len(pages) + 1), pages, marker="o", markersize=3, label=strat)
        ax2.set_xlabel("Page number")
        ax2.set_ylabel("Items returned")
        ax2.set_title("Items per page (last page is partial)")
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)

    if all(r["Complete & deduped"] == "✅" for r in rows):
        st.success(f"All strategies extracted {n_records:,} records completely, no duplicates, "
                   "429s absorbed by backoff.")
else:
    st.info("Pick strategies and click **Extract**.")
