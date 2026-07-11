from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from tracker_parser import REPO_URL, cumulative_timeline, parse_tracker, portfolio_stats

st.set_page_config(page_title="Portfolio Dashboard", page_icon="🏗️", layout="wide")
st.title("🏗️ Phoebe the Builder — Portfolio Dashboard")
st.caption(f"All 60 builds of Portfolio 1 in one view. Source: TRACKER.md · [repo]({REPO_URL})")

TRACKER_CANDIDATES = [Path(__file__).resolve().parents[2] / "TRACKER.md", Path("TRACKER.md")]
tracker_path = next((p for p in TRACKER_CANDIDATES if p.exists()), None)

uploaded = st.sidebar.file_uploader("TRACKER.md (optional override)", type=["md"])
if uploaded:
    text = uploaded.read().decode("utf-8")
elif tracker_path:
    text = tracker_path.read_text()
    st.sidebar.success(f"Loaded {tracker_path}")
else:
    st.error("No TRACKER.md found — upload one in the sidebar.")
    st.stop()

builds = [b for b in parse_tracker(text) if b.day <= 60]
stats = portfolio_stats(builds)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Builds shipped", f"{stats['completed']}/{stats['total']}")
c2.metric("Calendar days", stats["calendar_days"])
c3.metric("Builds per day", stats["builds_per_calendar_day"])
c4.metric("Active build days", stats["active_days"])
c5.metric("Busiest day", f"{stats['busiest_day'][1]} builds")

col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("Burn-up: cumulative builds")
    tl = pd.DataFrame(cumulative_timeline(builds))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(tl["date"], tl["cumulative"], color="#457b9d", linewidth=2.5, label="actual")
    ax.plot(tl["date"], tl["ideal"], color="#adb5bd", linestyle="--", linewidth=1, label="1/day pace")
    ax.fill_between(tl["date"], tl["cumulative"], alpha=0.15, color="#457b9d")
    ax.set_ylabel("Builds completed")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
with col2:
    st.subheader("By product line")
    by_line = pd.DataFrame(
        [{"Product line": k, "Done": v["done"], "Total": v["total"]} for k, v in stats["by_line"].items()]
    )
    fig2, ax2 = plt.subplots(figsize=(5, 3.2))
    ax2.barh(by_line["Product line"], by_line["Done"], color="#2a9d8f")
    ax2.set_xlabel("Builds")
    ax2.invert_yaxis()
    fig2.tight_layout()
    st.pyplot(fig2)

st.subheader("All builds")
query = st.text_input("Filter by name, slug, or product line", "")
df = pd.DataFrame(
    {
        "Day": [b.day for b in builds],
        "Project": [b.title for b in builds],
        "Product line": [b.product_line for b in builds],
        "Completed": [str(b.completed) if b.done else "—" for b in builds],
        "GitHub": [b.url for b in builds],
    }
)
if query.strip():
    mask = df.apply(lambda r: query.lower() in " ".join(str(v) for v in r.values).lower(), axis=1)
    df = df[mask]
st.dataframe(df, use_container_width=True, hide_index=True,
             column_config={"GitHub": st.column_config.LinkColumn("GitHub", display_text="open →")})
