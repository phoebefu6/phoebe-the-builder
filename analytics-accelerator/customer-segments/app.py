"""Customer Segmentation Tool — Streamlit UI.

Upload a customer CSV (or use the built-in sample), pick features, and let
KMeans surface natural segments — with auto-k, silhouette quality, named
profiles, and a 2-D scatter.
"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from segments import sample_customers, segment_customers, select_numeric_features

st.set_page_config(
    page_title="Customer Segmentation Tool", page_icon="👥", layout="wide"
)

st.title("👥 Customer Segmentation Tool")
st.caption(
    "Find your natural customer groups with KMeans — auto-picks the number of "
    "segments, names each one, and shows how distinct they really are."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Customer CSV", type=["csv"])
    use_sample = st.button("Use sample customer base")

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = sample_customers()
else:
    st.info(
        "Upload a CSV (one row per customer, numeric behavioral columns), "
        "or click **Use sample customer base** to try it."
    )
    st.stop()

st.subheader("Raw data")
st.dataframe(df.head(20), use_container_width=True)

numeric_cols = select_numeric_features(df)
if len(numeric_cols) < 2:
    st.error("Need at least 2 numeric behavioral columns to segment.")
    st.stop()

with st.sidebar:
    st.header("Segmentation")
    features = st.multiselect(
        "Features to cluster on", numeric_cols, default=numeric_cols
    )
    auto = st.checkbox("Auto-pick number of segments", value=True)
    k = None if auto else st.slider("Number of segments (k)", 2, 8, 3)

if len(features) < 2:
    st.warning("Pick at least 2 features.")
    st.stop()

res = segment_customers(df, n_clusters=k, feature_cols=features)

c1, c2, c3 = st.columns(3)
c1.metric("Segments found", res.k)
c2.metric("Silhouette score", f"{res.silhouette:.3f}")
c3.metric("Customers", len(df))
st.caption(
    "Silhouette ranges -1 to 1; higher means tighter, better-separated clusters. "
    "Above ~0.5 is a strong segmentation."
)

# --- Segment profiles ------------------------------------------------------
st.subheader("Segment profiles")
prof_df = pd.DataFrame(
    [
        {"Segment": p.name, "Customers": p.size, "Share %": p.share_pct, **p.means}
        for p in res.profiles
    ]
)
st.dataframe(prof_df, use_container_width=True)

# --- Auto-k curve ----------------------------------------------------------
if res.auto_k_scores:
    st.subheader("How many segments? (silhouette by k)")
    ks = sorted(res.auto_k_scores)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(ks, [res.auto_k_scores[i] for i in ks], marker="o", color="#4F46E5")
    ax.axvline(res.k, color="#DC2626", linestyle="--", label=f"chosen k={res.k}")
    ax.set_xlabel("k (segments)")
    ax.set_ylabel("silhouette")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# --- 2-D scatter -----------------------------------------------------------
st.subheader("Segment map")
ax_x = st.selectbox("X axis", features, index=0)
ax_y = st.selectbox("Y axis", features, index=min(1, len(features) - 1))

fig2, ax2 = plt.subplots(figsize=(8, 5))
palette = [
    "#4F46E5", "#16A34A", "#DC2626", "#F59E0B",
    "#0EA5E9", "#9333EA", "#65A30D", "#DB2777",
]
for p in res.profiles:
    mask = res.labels == p.label
    ax2.scatter(
        df[ax_x][mask],
        df[ax_y][mask],
        s=22,
        alpha=0.7,
        color=palette[p.label % len(palette)],
        label=p.name,
    )
ax2.set_xlabel(ax_x)
ax2.set_ylabel(ax_y)
ax2.legend(fontsize=8)
st.pyplot(fig2)
plt.close(fig2)

# --- Download labeled data -------------------------------------------------
labeled = df.copy()
name_by_label = {p.label: p.name for p in res.profiles}
labeled["segment"] = res.labels
labeled["segment_name"] = [name_by_label[lbl] for lbl in res.labels]
buf = io.StringIO()
labeled.to_csv(buf, index=False)
st.download_button(
    "⬇️ Download labeled customers CSV",
    buf.getvalue(),
    file_name="customers_segmented.csv",
    mime="text/csv",
)
