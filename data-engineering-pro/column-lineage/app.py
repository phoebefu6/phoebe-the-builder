from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from lineage import SAMPLE_MODELS, parse_models

st.set_page_config(page_title="Column-Level Lineage Parser", page_icon="🧬", layout="wide")
st.title("🧬 Column-Level Lineage Parser")
st.caption("Paste your SQL models. See which upstream column feeds each downstream column - "
           "and the exact blast radius before you change anything.")

with st.sidebar:
    st.header("SQL models")
    st.write("One `CREATE TABLE/VIEW ... AS SELECT ...` per box. Edit or add your own.")
    scripts = {}
    default = SAMPLE_MODELS
    n = st.number_input("Number of models", 1, 8, len(default))
    names = list(default)
    for i in range(int(n)):
        key = names[i] if i < len(names) else f"model_{i+1}"
        scripts[key] = st.text_area(key, value=default.get(key, ""), height=140, key=f"sql_{i}")

lin = parse_models({k: v for k, v in scripts.items() if v.strip()})

if not lin.edges:
    st.info("Add at least one SQL model in the sidebar to see lineage.")
    st.stop()

df = pd.DataFrame(
    [{"Source column": e.source, "→": "→", "Target column": e.target} for e in lin.edges]
)

left, right = st.columns([1, 1])
with left:
    st.subheader("Column dependencies")
    st.dataframe(df, use_container_width=True, hide_index=True)
    if lin.warnings:
        for w in dict.fromkeys(lin.warnings):
            st.warning(w)

with right:
    st.subheader("Impact analysis")
    all_cols = sorted({e.source for e in lin.edges} | {e.target for e in lin.edges})
    picked = st.selectbox("If I change this column…", all_cols)
    direct = lin.downstream_of(picked)
    blast = lin.impact(picked)
    st.metric("Directly affected columns", len(direct))
    st.metric("Total blast radius (transitive)", len(blast))
    if blast:
        st.error("Changing **{}** ripples to:\n\n".format(picked)
                 + "\n".join(f"- `{c}`" for c in blast))
    else:
        st.success(f"`{picked}` is a leaf - nothing downstream depends on it.")

# ---- Lineage graph -------------------------------------------------------
st.subheader("Lineage graph")
nodes = sorted({e.source for e in lin.edges} | {e.target for e in lin.edges})
depth: dict[str, int] = {}


def _depth(node: str, guard: set[str]) -> int:
    if node in depth:
        return depth[node]
    parents = [e.source for e in lin.edges if e.target == node]
    d = 0 if not parents else 1 + max(
        (_depth(p, guard | {node}) for p in parents if p not in guard), default=0
    )
    depth[node] = d
    return d


for nd in nodes:
    _depth(nd, set())

layers: dict[int, list[str]] = {}
for nd in nodes:
    layers.setdefault(depth[nd], []).append(nd)

pos = {}
for d, layer in layers.items():
    for j, nd in enumerate(sorted(layer)):
        pos[nd] = (d, -j)

fig, ax = plt.subplots(figsize=(11, max(4, 0.6 * max(len(v) for v in layers.values()))))
for e in lin.edges:
    x1, y1 = pos[e.source]
    x2, y2 = pos[e.target]
    hot = e.source == picked or e.target in blast
    ax.annotate("", xy=(x2 - 0.02, y2), xytext=(x1 + 0.02, y1),
                arrowprops=dict(arrowstyle="->", color="#e63946" if hot else "#c9ccd1",
                                lw=1.8 if hot else 0.9))
for nd, (x, y) in pos.items():
    hot = nd == picked or nd in blast
    ax.text(x, y, nd, ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="#e63946" if nd == picked else ("#ffd6d9" if hot else "#eef1f5"),
                      ec="#457b9d", lw=1))
ax.axis("off")
ax.set_title("Column lineage (red = blast radius of selected column)")
fig.tight_layout()
st.pyplot(fig)
