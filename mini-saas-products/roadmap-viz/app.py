from __future__ import annotations

import pandas as pd
import streamlit as st

from roadmap_viz import STATUS_COLORS, RoadmapItem, render_roadmap

st.set_page_config(page_title="Product Roadmap Visualizer", page_icon="🗺️", layout="wide")
st.title("🗺️ Product Roadmap Visualizer")
st.caption(
    "Roadmaps are PowerPoints that go stale. This one is generated from data — edit the table, "
    "regenerate the timeline, always current."
)

with st.sidebar:
    st.subheader("Settings")
    today = st.text_input("Today marker (YYYY-MM-DD)", "2026-04-01")
    title = st.text_input("Title", "Product Roadmap 2026")
    st.write("**Status colors:** " + " · ".join(f"{k}" for k in STATUS_COLORS))

default_df = pd.DataFrame([
    {"lane": "Platform", "name": "Auth revamp", "start": "2026-01-01", "end": "2026-02-15", "status": "done"},
    {"lane": "Platform", "name": "Mobile app", "start": "2026-02-15", "end": "2026-05-30", "status": "in-progress"},
    {"lane": "Platform", "name": "SSO / SAML", "start": "2026-06-01", "end": "2026-07-15", "status": "planned"},
    {"lane": "Growth", "name": "Onboarding flow", "start": "2026-01-15", "end": "2026-03-01", "status": "done"},
    {"lane": "Growth", "name": "Referral program", "start": "2026-03-15", "end": "2026-05-01", "status": "at-risk"},
    {"lane": "Data", "name": "Data warehouse", "start": "2026-02-01", "end": "2026-04-15", "status": "in-progress"},
    {"lane": "Data", "name": "Self-serve analytics", "start": "2026-04-15", "end": "2026-08-01", "status": "planned"},
])

st.subheader("Roadmap items")
edited = st.data_editor(
    default_df, num_rows="dynamic", use_container_width=True, hide_index=True,
    column_config={"status": st.column_config.SelectboxColumn("status", options=list(STATUS_COLORS.keys()))},
)

if st.button("Render roadmap", type="primary"):
    items = []
    for _, r in edited.iterrows():
        if not str(r["name"]).strip():
            continue
        try:
            items.append(RoadmapItem(str(r["name"]), str(r["lane"]), str(r["start"]),
                                     str(r["end"]), str(r["status"])))
        except Exception:
            st.warning(f"Skipped '{r['name']}' — check date format (YYYY-MM-DD).")
    if not items:
        st.warning("Add at least one valid item.")
        st.stop()

    try:
        fig = render_roadmap(items, today=today, title=title)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Render failed: {e}")

    late = [it for it in items if it.status == "at-risk"]
    if late:
        st.error("⚠️ At-risk items: " + ", ".join(it.name for it in late))
