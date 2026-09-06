from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from incremental import WatermarkStore, make_initial_source, run_incremental_load, simulate_source_day

st.set_page_config(page_title="Incremental / CDC Loader", page_icon="⏩", layout="wide")
st.title("⏩ Incremental / CDC Loader")
st.caption("Watermark + upsert instead of nightly full reloads — watch the scanned-rows savings compound.")

with st.sidebar:
    st.header("Simulation")
    n_rows = st.slider("Initial source rows", 100, 5000, 500, step=100)
    n_days = st.slider("Days to simulate", 2, 10, 5)
    inserts = st.slider("Inserts per day", 0, 200, 40)
    updates = st.slider("Updates per day", 0, 200, 25)
    deletes = st.slider("Soft deletes per day", 0, 50, 5)

if st.button("Run simulation", type="primary"):
    store = WatermarkStore(str(Path(tempfile.mkdtemp()) / "wm.json"))
    source = make_initial_source(n_rows)
    target = source.iloc[0:0].copy()
    history = []
    for d in range(n_days):
        day = f"2026-07-{d + 1:02d}"
        if d > 0:
            source = simulate_source_day(source, day, inserts, updates, deletes, seed=42 + d)
        target, stats = run_incremental_load(source, target, key="id", watermark_col="updated_at",
                                             store=store, table="customers", cycle=d + 1,
                                             soft_delete_col="deleted")
        history.append(stats)

    df = pd.DataFrame([{
        "Cycle": s.cycle, "Source rows": s.source_rows, "Extracted": s.extracted,
        "Inserted": s.inserted, "Updated": s.updated, "Deleted": s.deleted,
        "% scanned": s.scanned_pct} for s in history])

    c1, c2, c3 = st.columns(3)
    full_reload = df["Source rows"].sum()
    incremental = df["Extracted"].sum()
    c1.metric("Rows a full reload would scan", f"{full_reload:,}")
    c2.metric("Rows incremental actually scanned", f"{incremental:,}")
    c3.metric("Work avoided", f"{100 * (1 - incremental / full_reload):.1f}%")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(df["Cycle"] - 0.2, df["Source rows"], 0.4, label="full reload", color="#adb5bd")
        ax.bar(df["Cycle"] + 0.2, df["Extracted"], 0.4, label="incremental", color="#2a9d8f")
        ax.set_xlabel("Load cycle")
        ax.set_ylabel("Rows scanned")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    live = target[~target["deleted"].fillna(False).astype(bool)]
    st.success(f"Target table: {len(live):,} live rows after {n_days} cycles "
               f"(cycle 1 = initial full load, then increments only).")
else:
    st.info("Set the simulation in the sidebar and click **Run simulation**.")
