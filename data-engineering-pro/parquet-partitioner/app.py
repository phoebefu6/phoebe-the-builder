from __future__ import annotations

import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.dataset as ds
import streamlit as st

from partitioner import (compare_compressions, convert_to_partitioned_parquet,
                         infer_partition_cols, make_sample_events, timed_query)

st.set_page_config(page_title="Parquet Partitioner", page_icon="🗂️", layout="wide")
st.title("🗂️ CSV/JSON → Partitioned Parquet")
st.caption("Turn a pile of CSVs into a partitioned, compressed data lake layout — and prove the query speedup.")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("CSV or JSON file", type=["csv", "json"])
    n_rows = st.slider("...or sample events", 10_000, 500_000, 200_000, step=10_000)
    compression = st.selectbox("Compression", ["snappy", "gzip", "zstd"])

if st.button("Convert", type="primary"):
    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_json(uploaded)
        csv_bytes = uploaded.size
    else:
        df = make_sample_events(n_rows)
        csv_bytes = None

    work = Path(tempfile.mkdtemp())
    inferred = infer_partition_cols(df)
    report = convert_to_partitioned_parquet(df, str(work / "dataset"), compression=compression,
                                            csv_bytes=csv_bytes)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{report.rows:,}")
    c2.metric("CSV size", f"{report.csv_bytes / 1e6:.1f} MB")
    c3.metric("Parquet size", f"{report.parquet_bytes / 1e6:.1f} MB", f"-{report.compression_ratio}x")
    c4.metric("Partitions", report.n_partitions)
    st.info(f"Partitioned by: `{', '.join(report.partition_cols) or 'none inferred'}`")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Size by codec")
        sizes = compare_compressions(df, str(work))
        fig, ax = plt.subplots(figsize=(5, 3))
        names = list(sizes) + ["csv"]
        vals = [sizes[k] / 1e6 for k in sizes] + [report.csv_bytes / 1e6]
        ax.bar(names, vals, color=["#2a9d8f"] * len(sizes) + ["#adb5bd"])
        ax.set_ylabel("MB on disk")
        fig.tight_layout()
        st.pyplot(fig)
    with col2:
        st.subheader("Partition pruning")
        if report.partition_cols and "__year" in report.partition_cols[0]:
            month_col = report.partition_cols[1]
            rows_f, t_filtered = timed_query(str(work / "dataset"),
                                             ds.field(month_col) == 3)
            start = time.perf_counter()
            full = ds.dataset(str(work / "dataset"), format="parquet", partitioning="hive").to_table()
            t_full = time.perf_counter() - start
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.bar(["full scan", "one month\n(pruned)"], [t_full * 1000, t_filtered * 1000],
                    color=["#adb5bd", "#e76f51"])
            ax2.set_ylabel("Query time (ms)")
            fig2.tight_layout()
            st.pyplot(fig2)
            st.success(f"One-month query read {rows_f:,} of {full.num_rows:,} rows in "
                       f"{t_filtered * 1000:.0f} ms vs {t_full * 1000:.0f} ms full scan.")
        else:
            st.info("No date partition inferred — pruning demo needs a date-like column.")
else:
    st.info("Upload a file or use sample events, then click **Convert**.")
