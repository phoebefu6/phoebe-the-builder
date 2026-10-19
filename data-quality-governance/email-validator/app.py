"""Streamlit front end: paste addresses, see what each of the ten validators and the RFC reference say.

Tab one checks your own addresses. Tab two re-weights the sign-up mix (a declared assumption, not data) and shows
how many real sign-ups each validator would turn away. Verified headless with `streamlit.testing.v1.AppTest`,
widgets addressed BY KEY.
"""

from __future__ import annotations

import explore as E
import streamlit as st

st.set_page_config(page_title="Email validator audit", layout="wide")
st.title("The valid address was rejected: ten email validators against the RFC")
tab_try, tab_mix = st.tabs(["Check addresses", "Your sign-up mix"])

with tab_try:
    raw = st.text_area(
        "One address per line",
        "o'brien@example.ie\njosé@example.com\nlee@studio.photography\njohn..doe@example.com",
        key="addrs",
    )
    addrs = [a for a in raw.split("\n") if a.strip()]
    if not addrs:
        st.info("Paste at least one address.")
    else:
        rows = []
        for a in addrs:
            tier, why = E.reference(a)
            row = {"address": a, "RFC reference": f"{tier} ({why})"}
            row.update({n: "accept" if f(a) else "reject" for n, f in E.VALIDATORS.items()})
            rows.append(row)
        st.dataframe(rows, width="stretch")
        wrong = sum(
            1
            for r in rows
            for n in E.VALIDATORS
            if (r["RFC reference"].startswith("ok") and r[n] == "reject")
            or (r["RFC reference"].startswith("bad") and r[n] == "accept")
        )
        if wrong:
            st.warning(f"{wrong} validator verdicts disagree with the RFC on these addresses.")
        else:
            st.success("Every validator agrees with the RFC on these addresses.")

with tab_mix:
    st.caption(
        "Share of real sign-ups in each deliverable group. Declared, not measured: set it from your own data."
    )
    share = {}
    for g, v in E.SHARE.items():
        share[g] = st.number_input(g, 0.0, 1.0, float(v), step=0.001, format="%.4f", key=f"share_{g}")
    if sum(share.values()) <= 0:
        st.error("The shares must add up to more than zero.")
    else:
        r = E.study(share)
        table = [
            {
                "validator": n,
                "deliverable rejected": len(p["false_reject"]),
                "invalid accepted": len(p["false_accept"]),
                "sign-ups lost": f"{p['signups_lost']:.2%}",
            }
            for n, p in r["per"].items()
        ]
        st.table(table)
        worst = max(r["per"], key=lambda n: r["per"][n]["signups_lost"])
        st.warning(
            f"Worst under this mix: {worst}, turning away {r['per'][worst]['signups_lost']:.2%} of sign-ups."
        )
    st.image("audit.png")
