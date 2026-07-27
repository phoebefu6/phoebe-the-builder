from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from crosstab import crosstab_chi2, interpret_v, key_cells, narrate, sample_dataframe

st.set_page_config(page_title="Crosstab & Chi-Square Tool", layout="wide")
st.title("Crosstab & Chi-Square Tool")
st.caption('"Compare groups in survey data" — contingency table, chi-square test, effect size, and which cells drive it.')

with st.sidebar:
    uploaded = st.file_uploader("Upload CSV (or use sample)", type=["csv"])
    alpha = st.slider("Significance level α", 0.01, 0.10, 0.05)

df = pd.read_csv(uploaded) if uploaded is not None else sample_dataframe()
cat_cols = [c for c in df.columns if df[c].dtype == object or df[c].nunique() <= 12]

if len(cat_cols) < 2:
    st.error("Need at least two categorical columns.")
    st.stop()

c1, c2 = st.columns(2)
row = c1.selectbox("Row variable", cat_cols, index=0)
col = c2.selectbox("Column variable", cat_cols, index=min(1, len(cat_cols) - 1))

if row == col:
    st.warning("Pick two different variables.")
    st.stop()

result = crosstab_chi2(df, row, col, alpha=alpha)

m1, m2, m3 = st.columns(3)
m1.metric("chi²", f"{result.chi2:.1f}")
m2.metric("p-value", f"{result.p_value:.4f}")
m3.metric("Cramér's V", f"{result.cramers_v} ({interpret_v(result.cramers_v)})")

(st.success if result.significant else st.info)(narrate(result, row, col))

st.subheader("Observed counts")
st.dataframe(result.table, use_container_width=True)

st.subheader("Standardized residuals (where the action is)")
fig, ax = plt.subplots(figsize=(6.5, 4.2))
res = result.residuals
im = ax.imshow(res.values, cmap="RdBu_r", vmin=-3, vmax=3)
ax.set_xticks(range(len(res.columns)))
ax.set_xticklabels(res.columns, rotation=30, ha="right")
ax.set_yticks(range(len(res.index)))
ax.set_yticklabels(res.index)
for i in range(len(res.index)):
    for j in range(len(res.columns)):
        ax.text(j, i, f"{res.values[i, j]:+.1f}", ha="center", va="center",
                color="white" if abs(res.values[i, j]) > 1.5 else "black")
ax.set_title("Residuals: red = more than expected, blue = fewer", fontsize=11, weight="bold")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
st.pyplot(fig)

cells = key_cells(result)
if cells:
    st.caption("Driving cells (|residual| ≥ 2): " +
               "; ".join(f"{r}×{c}: {v:+.1f}σ ({d})" for r, c, v, d in cells))
