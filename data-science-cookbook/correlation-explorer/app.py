from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from correxplore import correlation_matrix, high_correlations, sample_dataframe, suggest_drops, vif

st.set_page_config(page_title="Correlation & Multicollinearity Explorer", layout="wide")
st.title("Correlation & Multicollinearity Explorer")
st.caption('"Which features relate?" — a correlation heatmap plus VIF to catch redundant, collinear features.')

with st.sidebar:
    uploaded = st.file_uploader("Upload CSV (or use sample)", type=["csv"])
    method = st.selectbox("Correlation method", ["pearson", "spearman"])
    threshold = st.slider("High-correlation threshold", 0.5, 0.99, 0.8)

df = pd.read_csv(uploaded) if uploaded is not None else sample_dataframe()

corr = correlation_matrix(df, method)

st.subheader("Correlation matrix")
fig, ax = plt.subplots(figsize=(6.5, 5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns, fontsize=8)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                fontsize=7, color="white" if abs(corr.values[i, j]) > 0.5 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
st.pyplot(fig)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Highly-correlated pairs")
    pairs = high_correlations(df, threshold, method)
    if pairs:
        st.dataframe(pd.DataFrame([{"feature A": p.a, "feature B": p.b, "corr": round(p.corr, 3)} for p in pairs]),
                     use_container_width=True)
    else:
        st.write("None above threshold.")

with c2:
    st.subheader("Variance Inflation Factor")
    v = vif(df)
    st.dataframe(v, use_container_width=True)
    st.caption("VIF > 5 elevated, > 10 serious multicollinearity.")

drops = suggest_drops(df)
if drops:
    st.warning("Suggested features to drop (greedy, by VIF): " + ", ".join(f"`{d}`" for d in drops))
else:
    st.success("No serious multicollinearity — no drops suggested.")

with st.expander("Data preview"):
    st.dataframe(df.head(), use_container_width=True)
