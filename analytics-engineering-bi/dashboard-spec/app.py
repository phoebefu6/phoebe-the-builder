from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dashspec import classify_columns, recommend_dashboard, sample_dataframe, spec_to_dict

st.set_page_config(page_title="Dashboard Spec Generator", layout="wide")
st.title("Dashboard Spec Generator")
st.caption("Vague dashboard asks? Point it at data — it recommends the right charts and renders them.")

with st.sidebar:
    st.subheader("Data")
    uploaded = st.file_uploader("Upload a CSV (or use sample)", type=["csv"])
    max_panels = st.slider("Max panels", 3, 8, 6)

df = pd.read_csv(uploaded) if uploaded is not None else sample_dataframe()
roles = classify_columns(df)

st.subheader("Detected column roles")
st.dataframe(pd.DataFrame([{"column": r.name, "role": r.role} for r in roles]), use_container_width=True)

panels = recommend_dashboard(df, max_panels=max_panels)


def _render(panel, df):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    try:
        if panel.chart == "kpi":
            return None
        if panel.chart == "line":
            g = df.groupby(panel.x)[panel.y].sum()
            ax.plot(range(len(g)), g.values, color="#3b6fd6")
            step = max(1, len(g) // 6)
            ax.set_xticks(range(0, len(g), step))
            ax.set_xticklabels([str(i)[5:] for i in g.index[::step]], rotation=45, fontsize=7)
        elif panel.chart == "bar":
            g = df.groupby(panel.x)[panel.y].sum().sort_values(ascending=False)
            ax.bar(g.index.astype(str), g.values, color="#3b6fd6")
            plt.xticks(rotation=30, ha="right", fontsize=8)
        elif panel.chart == "grouped_bar":
            piv = df.pivot_table(index=panel.x, columns=panel.series, values=panel.y, aggfunc="sum", fill_value=0)
            piv.plot(kind="bar", ax=ax, width=0.8)
            ax.legend(fontsize=6, title=panel.series)
            plt.xticks(rotation=30, ha="right", fontsize=8)
        elif panel.chart == "scatter":
            ax.scatter(df[panel.x], df[panel.y], s=12, alpha=0.6, color="#3b6fd6")
            ax.set_xlabel(panel.x, fontsize=8)
            ax.set_ylabel(panel.y, fontsize=8)
        elif panel.chart == "histogram":
            ax.hist(df[panel.x], bins=20, color="#3b6fd6")
        ax.set_title(panel.title, fontsize=10, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        return fig
    except Exception:
        plt.close(fig)
        return None


st.subheader("Recommended dashboard")
cols = st.columns(2)
i = 0
for panel in panels:
    with cols[i % 2]:
        if panel.chart == "kpi":
            val = df[panel.y].sum()
            st.metric(panel.title, f"{val:,.0f}")
            st.caption(panel.rationale)
        else:
            fig = _render(panel, df)
            if fig:
                st.pyplot(fig)
                st.caption(panel.rationale)
    i += 1

st.subheader("Dashboard spec (JSON)")
spec = spec_to_dict(panels)
st.code(json.dumps(spec, indent=2), language="json")
st.download_button("Download spec.json", data=json.dumps(spec, indent=2), file_name="dashboard_spec.json", mime="application/json")
