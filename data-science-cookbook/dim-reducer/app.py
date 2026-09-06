from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dimreduce import pca_project, sample_dataframe, scree, tsne_project

st.set_page_config(page_title="PCA / t-SNE Explorer", layout="wide")
st.title("PCA / t-SNE Explorer")
st.caption('"Too many features to see" — project high-dimensional data to 2D and look at the structure.')

with st.sidebar:
    uploaded = st.file_uploader("Upload CSV (or use sample)", type=["csv"])
    method = st.radio("Method", ["PCA", "t-SNE"])

df = pd.read_csv(uploaded) if uploaded is not None else sample_dataframe()
label_options = ["(none)"] + [c for c in df.columns if df[c].dtype == object or df[c].nunique() <= 12]
label_col = st.selectbox("Color by", label_options)
label_col = None if label_col == "(none)" else label_col

if method == "PCA":
    proj = pca_project(df, label_col)
else:
    perp = st.sidebar.slider("t-SNE perplexity", 5, 50, 30)
    proj = tsne_project(df, label_col, perplexity=perp)

st.subheader(f"{proj.method} projection")
fig, ax = plt.subplots(figsize=(7.5, 5.5))
labels = proj.labels
uniq = list(dict.fromkeys(labels))
cmap = plt.get_cmap("tab10")
for i, lab in enumerate(uniq):
    mask = labels == lab
    ax.scatter(proj.coords[mask, 0], proj.coords[mask, 1], s=18, alpha=0.7,
               color=cmap(i % 10), label=str(lab))
if label_col:
    ax.legend(fontsize=8, title=label_col)
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_title(f"{proj.method} — {df.select_dtypes('number').shape[1]}D → 2D", fontsize=12, weight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
st.pyplot(fig)

if proj.method == "PCA":
    st.metric("Variance kept in 2D", f"{sum(proj.explained_variance):.1%}")
    st.subheader("Scree plot")
    sc = scree(df, label_col)
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.bar(range(1, len(sc) + 1), sc, color="#3b6fd6")
    ax2.plot(range(1, len(sc) + 1), np.cumsum(sc), color="#c0553b", marker="o", label="cumulative")
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Explained variance")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig2)

with st.expander("Data preview"):
    st.dataframe(df.head(), use_container_width=True)
