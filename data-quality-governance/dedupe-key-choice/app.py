"""Streamlit front end: paste customer names, see the key every scheme makes and which rows it joins.

Tab one keys your own rows under all six schemes and counts the customers each scheme leaves. Tab two shows the
study's scoreboard and figure. Verified headless with `streamlit.testing.v1.AppTest`, widgets BY KEY.
"""

from __future__ import annotations

from itertools import combinations

import explore as E
import streamlit as st

st.set_page_config(page_title="Dedupe key choice", layout="wide")
st.title("Same customer, three rows: six match keys against one customer table")
tab_try, tab_study = st.tabs(["Key your names", "The study"])

with tab_try:
    raw = st.text_area(
        "One customer name per line",
        "Jonathan Smith\nSmith, Jonathan\nJonathon Smith\nJohn Smith\nJosé Núñez\nJose Nunez",
        key="rows",
    )
    names = [line.strip() for line in raw.split("\n") if line.strip()]
    if not names:
        st.info("Paste at least one name.")
    else:
        table = [{"name": s} for s in names]
        counts = {}
        for key, f in E.ROW_KEYS.items():
            ks = [f(s) for s in names]
            for row, k in zip(table, ks):
                row[key] = k
            counts[key] = len(set(ks))
        for key, (sim, t) in E.PAIR_KEYS.items():
            edges = [(i, j) for i, j in combinations(range(len(names)), 2) if sim(names[i], names[j]) >= t]
            lab = E._components(len(names), edges)
            for row, c in zip(table, lab):
                row[key] = f"cluster {c}"
            counts[key] = len(set(lab))
        st.dataframe(table, width="stretch")
        lo, hi = min(counts.values()), max(counts.values())
        if lo == hi:
            st.success(f"Every key leaves {lo} customers: the keys agree on these rows.")
        else:
            st.warning(
                f"The keys leave anywhere from {lo} to {hi} customers for {len(names)} rows. The key choice is "
                "deciding how many customers you have; label a sample and count the splits and merges."
            )

with tab_study:
    r = E.study()
    st.table(
        [
            {
                "key": n,
                "false splits": f"{len(p['false_splits'])}/{r['ceiling']['true_pairs']}",
                "false merges": len(p["false_merges"]),
                "customers": f"{p['customers']} (truth {r['true_customers']})",
                "chained pairs": p["chained_pairs"],
            }
            for n, p in r["per"].items()
        ]
    )
    st.image("audit.png")
