"""Streamlit front end: paste full names, see how each of the six splitters takes them apart.

Tab one splits your own names and flags the rows where the splitters disagree. Tab two re-weights the customer mix
(a declared assumption, not data) and shows what share of customers each splitter would greet by the wrong name.
Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import explore as E
import streamlit as st

st.set_page_config(page_title="Name splitter audit", layout="wide")
st.title("First name came out as van: six name splitters against twelve naming conventions")
tab_try, tab_mix = st.tabs(["Split names", "Your customer mix"])

with tab_try:
    raw = st.text_area(
        "One full name per line",
        "van den Broek, Pieter\nDr. Robert Hale Jr.\nAna María Pérez López\nWang Fang",
        key="names",
    )
    names = [n.strip() for n in raw.split("\n") if n.strip()]
    if not names:
        st.info("Paste at least one name.")
    else:
        rows, split_votes = [], 0
        for n in names:
            row = {"full name": n}
            outs = {k: f(n) for k, f in E.SPLITTERS.items()}
            row.update({k: f"{g} | {fam}" for k, (g, fam) in outs.items()})
            rows.append(row)
            split_votes += len(set(outs.values())) > 1
        st.dataframe(rows, width="stretch")
        if split_votes:
            st.warning(
                f"The splitters disagree on {split_votes} of {len(names)} names. Ask the person: capture given "
                "and family names in separate fields."
            )
        else:
            st.success("All six splitters agree on these names (agreement is not proof they are right).")

with tab_mix:
    st.caption(
        "Share of your customers in each naming convention. Declared, not measured: set it from your data."
    )
    share = {}
    for g, v in E.SHARE.items():
        share[g] = st.number_input(g, 0.0, 1.0, float(v), step=0.005, format="%.3f", key=f"share_{g}")
    if sum(share.values()) <= 0:
        st.error("The shares must add up to more than zero.")
    else:
        r = E.study(share)
        st.table(
            [
                {
                    "splitter": n,
                    "both right": f"{p['both_right']}/{r['n']}",
                    "customers greeted wrong": f"{p['greeted_wrong']:.2%}",
                }
                for n, p in r["per"].items()
            ]
        )
        worst = max(r["per"], key=lambda n: r["per"][n]["greeted_wrong"])
        st.warning(
            f"Worst under this mix: {worst}, greeting {r['per'][worst]['greeted_wrong']:.2%} by the wrong name."
        )
    st.image("audit.png")
