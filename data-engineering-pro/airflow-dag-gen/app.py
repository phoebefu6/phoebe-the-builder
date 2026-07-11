from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from dag_gen import generate_dag, load_yaml, topo_levels

st.set_page_config(page_title="Airflow DAG Generator", page_icon="🌀", layout="wide")
st.title("🌀 Airflow DAG Generator")
st.caption("YAML in → validated Airflow DAG code + dependency graph out. No more boilerplate.")

SAMPLE = """dag_id: nightly_sales_pipeline
description: Extract sales, transform, load to warehouse, refresh dashboard
schedule: "0 2 * * *"
start_date: 2026-07-01
owner: phoebe
retries: 2
tags: [sales, nightly]
tasks:
  - id: extract_orders
    type: bash
    command: python extract.py --source orders
  - id: extract_customers
    type: bash
    command: python extract.py --source customers
  - id: transform
    type: python
    callable: transform_sales
    depends_on: [extract_orders, extract_customers]
  - id: load_warehouse
    type: sql
    conn_id: warehouse
    sql: CALL load_sales_mart();
    depends_on: [transform]
  - id: refresh_dashboard
    type: bash
    command: curl -X POST $DASHBOARD_REFRESH_URL
    depends_on: [load_warehouse]
"""

yaml_text = st.text_area("DAG config (YAML)", SAMPLE, height=380)

if st.button("Generate DAG", type="primary"):
    try:
        cfg = load_yaml(yaml_text)
    except Exception as exc:
        st.error(f"YAML parse error: {exc}")
        st.stop()
    code, errors = generate_dag(cfg)
    if errors:
        st.error("Validation failed:")
        for e in errors:
            st.warning(f"• {e}")
    else:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Generated DAG")
            st.code(code, language="python")
            st.download_button("⬇️ Download DAG file", code, file_name=f"{cfg['dag_id']}.py")
        with col2:
            st.subheader("Dependency graph")
            tasks = cfg["tasks"]
            levels = topo_levels(tasks)
            by_level = {}
            for tid, lv in levels.items():
                by_level.setdefault(lv, []).append(tid)
            pos = {}
            for lv, ids in sorted(by_level.items()):
                for i, tid in enumerate(sorted(ids)):
                    pos[tid] = (lv, -(i - (len(ids) - 1) / 2))
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            for t in tasks:
                for dep in t.get("depends_on", []):
                    x0, y0 = pos[dep]
                    x1, y1 = pos[t["id"]]
                    ax.annotate("", xy=(x1 - 0.12, y1), xytext=(x0 + 0.12, y0),
                                arrowprops={"arrowstyle": "->", "color": "#6b6b80"})
            for tid, (x, y) in pos.items():
                ax.scatter([x], [y], s=1400, color="#457b9d", zorder=3)
                ax.text(x, y, tid.replace("_", "\n"), ha="center", va="center",
                        fontsize=6.5, color="white", zorder=4)
            ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)
        st.success(f"DAG `{cfg['dag_id']}` valid — {len(cfg['tasks'])} tasks, drop the file into your dags/ folder.")
