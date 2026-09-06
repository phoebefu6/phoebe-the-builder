"""Streamlit front end for the percentage audit.

Paste the counts behind a percentage column. The app never shows one column of
numbers - it shows every method's column, and names the ones that disagree.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import streamlit as st
from percentages import (
    CORPUS,
    METHOD_KIND,
    Row,
    Table,
    Verdict,
    audit,
    audit_corpus,
    no_method_is_clean,
    quota_violations,
    representable_step,
)

st.set_page_config(page_title="Percent Recomputer", page_icon="%", layout="wide")

BADGE = {
    Verdict.CONSISTENT: ("#e7f2e8", "#2f6b39", "consistent - every method returns the same column"),
    Verdict.RESIDUAL: ("#fbeeda", "#8a5410", "residual - the methods agree to within one unit and differ on which row holds it"),
    Verdict.CONTESTED: ("#f9e3e0", "#a5291c", "contested - the methods differ by more than a residual, or a paradox fires"),
    Verdict.UNDEFINED: ("#eceae6", "#4a4a4a", "undefined - there is no share to display"),
}
SEV_ICON = {"blocking": "🔴", "silent": "🟠", "advisory": "🔵"}

st.title("The percentages sum to 101%")
st.caption(
    "A percentage column is an apportionment - the same problem as seats in a parliament. "
    "Balinski and Young proved no method both stays inside the quota and avoids the Alabama "
    "paradox, so the honest output names which failure you took."
)

with st.sidebar:
    st.header("How to read this")
    st.markdown(
        "- **consistent** is the only verdict where the method choice does not matter\n"
        "- **contested** means two defensible columns exist and they differ by more than a rounding residual\n"
        "- a **quota violation** is a row awarded outside the floor and ceiling of its exact share"
    )
    st.divider()
    st.subheader("The scoreboard")
    board = no_method_is_clean()
    st.dataframe(
        pd.DataFrame(
            [{"method": m, "fails to sum": v[0], "quota": v[1], "alabama": v[2]}
             for m, v in board.items()]
        ),
        hide_index=True, use_container_width=True,
    )
    st.caption("over the 13 bundled tables that have a share at all - no method has an empty row")

tab_one, tab_corpus = st.tabs(["One table", "The bundled corpus"])

with tab_one:
    col_pick, col_units = st.columns([2, 1])
    with col_pick:
        preset = st.selectbox("start from a bundled table", [t.name for t in CORPUS], index=9)
    src = next(t for t in CORPUS if t.name == preset)
    with col_units:
        if src.kind == "seats":
            budget = st.number_input("seats to allocate", min_value=1, max_value=500,
                                     value=src.units, step=1)
            decimals = 0
        else:
            decimals = st.selectbox("decimal places", [0, 1, 2],
                                    index=[0, 1, 2].index(src.decimals))
            budget = 100 * 10 ** decimals

    text = st.text_area(
        "one `label, value` per line",
        value="\n".join(f"{r.label}, {r.value:g}" for r in src.rows),
        height=180,
    )
    rows: List[Row] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.rsplit(",", 1)
        if len(parts) != 2:
            continue
        try:
            rows.append(Row(parts[0].strip(), float(parts[1])))
        except ValueError:
            st.warning(f"skipped unparseable line: {line!r}")

    if rows:
        table = Table(preset, tuple(rows), int(budget), src.kind, decimals, src.group_of)
        a = audit(table)
        bg, fg, label = BADGE[a.verdict]
        st.markdown(
            f"<div style='background:{bg};color:{fg};padding:0.6rem 0.9rem;border-radius:6px;"
            f"font-weight:600'>{a.verdict.value.upper()} &mdash; {label}</div>",
            unsafe_allow_html=True,
        )

        if a.allocations:
            unit = "%" if table.kind == "percent" else " seats"
            frame: Dict[str, List] = {"row": list(table.labels),
                                      "exact share": [f"{float(q) / (10 ** decimals):.4f}"
                                                      for q in table.quotas()]}
            for al in a.allocations:
                col = [f"{p:g}" for p in (al.percents(table) if table.kind == "percent" else al.units)]
                mark = "" if al.sums_to(table) else "  (does not sum)"
                frame[al.method + mark] = col
            st.dataframe(pd.DataFrame(frame), hide_index=True, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("methods that sum", f"{sum(1 for x in a.allocations if x.sums_to(table))}"
                                          f" of {len(a.allocations)}")
            c2.metric("rows in dispute", len(a.disagreeing_rows()))
            c3.metric("widest gap", f"{a.max_row_gap()} units",
                      help="one unit is 0.1 of a point at 1 dp")
            step = representable_step(table)
            c4.metric("finest real step", f"{step:.3g} pts" if step else "n/a",
                      help="what the denominator can actually express")

            violations = []
            for al in a.allocations:
                if not al.sums_to(table):
                    continue
                for lbl, awarded, q in quota_violations(table, al):
                    violations.append({"method": al.method, "row": lbl, "awarded": awarded,
                                       "exact share": round(q, 3), "off by": round(awarded - q, 3)})
            if violations:
                st.subheader("Quota violations")
                st.dataframe(pd.DataFrame(violations), hide_index=True, use_container_width=True)

        st.subheader("Findings")
        for f in a.findings:
            st.markdown(f"{SEV_ICON[f.severity]} **{f.code}**"
                        + (f" · `{f.method}`" if f.method else "")
                        + f"  \n{f.message}")

with tab_corpus:
    rep = audit_corpus()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("consistent", rep.verdicts["consistent"])
    c2.metric("residual", rep.verdicts["residual"])
    c3.metric("contested", rep.verdicts["contested"])
    c4.metric("undefined", rep.verdicts["undefined"])
    st.dataframe(
        pd.DataFrame([
            {
                "table": x.table.name,
                "kind": x.table.kind,
                "rows": len(x.table.rows),
                "budget": x.table.units,
                "verdict": x.verdict.value,
                "widest gap": x.max_row_gap(),
                "rows in dispute": ", ".join(x.disagreeing_rows()) or "-",
                "findings": len(x.findings),
                "note": x.table.note,
            }
            for x in rep.audits
        ]),
        hide_index=True, use_container_width=True,
    )
    st.subheader("How often each mechanism fires")
    st.bar_chart(pd.DataFrame({"fires": rep.finding_counts}), height=320)
    st.caption("Method kinds: " + ", ".join(f"{m} ({k})" for m, k in METHOD_KIND.items()))
