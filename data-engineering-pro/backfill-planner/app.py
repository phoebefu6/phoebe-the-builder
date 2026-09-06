from __future__ import annotations

import random
import tempfile
import time
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from planner import BackfillState, plan_chunks, run_backfill

st.set_page_config(page_title="Backfill Planner", page_icon="🧱", layout="wide")
st.title("🧱 Backfill Planner")
st.caption("Date-range chunking + persisted state + retry/resume — backfills stop being scary shell scripts.")

with st.sidebar:
    st.header("Plan")
    start = st.date_input("Start", date(2026, 1, 1))
    end = st.date_input("End (exclusive)", date(2026, 7, 1))
    gran = st.selectbox("Granularity", ["daily", "weekly", "monthly"], index=1)
    fail_rate = st.slider("Simulated failure rate", 0.0, 0.5, 0.15, 0.05)
    max_parallel = st.slider("Max parallel", 1, 16, 4)

if st.button("Plan + run backfill", type="primary"):
    chunks = plan_chunks(start, end, gran)
    state = BackfillState(str(Path(tempfile.mkdtemp()) / "backfill.json"))
    state.init_plan(chunks)

    rng = random.Random(7)

    def job(chunk):
        time.sleep(0.002)
        if rng.random() < fail_rate:
            raise RuntimeError("simulated upstream timeout")

    summary = run_backfill(state, job, max_parallel=max_parallel)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks", summary["total"])
    c2.metric("Succeeded", summary["success"])
    c3.metric("Dead (3 strikes)", len(summary["dead_chunks"]))
    c4.metric("Complete", f"{summary['pct_complete']}%")

    df = pd.DataFrame([{"Chunk": c.chunk_id, "Start": c.start, "End": c.end,
                        "Status": c.status, "Attempts": c.attempts,
                        "Error": c.error or ""} for c in state.chunks])
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with col2:
        colors = {"success": "#2a9d8f", "failed": "#d62828", "pending": "#adb5bd", "running": "#e9c46a"}
        fig, ax = plt.subplots(figsize=(5, max(2.5, len(state.chunks) * 0.16)))
        for i, c in enumerate(state.chunks):
            ax.barh(i, (date.fromisoformat(c.end) - date.fromisoformat(c.start)).days,
                    left=date.fromisoformat(c.start).toordinal(), color=colors[c.status])
        ax.set_yticks(range(len(state.chunks)))
        ax.set_yticklabels([c.chunk_id for c in state.chunks], fontsize=6)
        ax.set_xticks([])
        ax.invert_yaxis()
        ax.set_title("Chunk map (green=done, red=dead)")
        fig.tight_layout()
        st.pyplot(fig)

    if summary["dead_chunks"]:
        st.error(f"Dead chunks after 3 attempts: {', '.join(summary['dead_chunks'])} — "
                 "fix the upstream issue, then re-run; successes are never repeated.")
    else:
        st.success("Backfill complete — every chunk succeeded (retries included).")
else:
    st.info("Set the range and click **Plan + run backfill**.")
