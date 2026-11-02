"""Streamlit front end: paste one customer's records from several systems, see the golden record each rule-set keeps.

Tab one resolves your own cluster (CSV: source, updated_at, then any of name, dob, email, phone, address). There is
no truth to score against here, so the app shows where the rule-sets DISAGREE: those are the fields a steward must
check. Tab two shows the study. Verified headless with `streamlit.testing.v1.AppTest`, widgets BY KEY.
"""

from __future__ import annotations

import csv
import io
from typing import List

import explore as E
import streamlit as st

st.set_page_config(page_title="golden record", layout="wide")
st.title("Which customer record is real? Six survivorship rule-sets, side by side")
tab_try, tab_study = st.tabs(["Resolve your cluster", "The study"])

DEFAULT = (
    "source,updated_at,name,dob,email,phone,address\n"
    "crm,82,Robert Petrov,1960-01-14,rpetrov570@post.test,+44 7700 998849,\"57 Lantern Way, G18 4XY\"\n"
    "web,807,Rob Petrov,,rpetrov570@post.test,+44 7700 930022,\"134 Mill Street, C6 7XY\"\n"
    "billing,558,ROBERT PETROV,1960-01-14,accounts@petrov-ltd.test,+44 7700 998849,\"57 Lantern Way, G18 4XY\"\n"
)


def parse(raw: str) -> List[E.Record]:
    recs: List[E.Record] = []
    for row in csv.DictReader(io.StringIO(raw.strip())):
        src = (row.get("source") or "").strip().lower()
        if src not in E.PRIORITY:
            raise ValueError(f"unknown source {src!r}; use one of {', '.join(E.SOURCES)}")
        day = int(row.get("updated_at") or 0)
        rec: E.Record = {"source": src, "updated_at": day, "_ts": {f: day for f in E.FIELDS}}
        rec.update({f: (row.get(f) or "").strip() or None for f in E.FIELDS})
        recs.append(rec)
    return recs


with tab_try:
    raw = st.text_area("One customer's records (CSV with header)", DEFAULT, height=150, key="cluster")
    try:
        recs = parse(raw)
    except ValueError as e:
        st.error(str(e))
        recs = []
    if len(recs) < 2:
        if recs:
            st.info("Paste at least two records - one record has nothing to resolve.")
    else:
        gs = {k: E.golden(recs, r) for k, r in E.RULES.items()}
        st.table([{"rule-set": k, **{f: g[f] or "-" for f in E.FIELDS}} for k, g in gs.items()])
        contested = [f for f in E.FIELDS if len({g[f] for g in gs.values()}) > 1]
        if contested:
            st.warning(f"The rule-sets disagree on: {', '.join(contested)}. Choosing a rule-set decides these for "
                       "you; check them against a fresh source.")
        else:
            st.success("All six rule-sets keep the same golden record.")
        st.caption("per_field reads each field's own capture day; here that is the row's updated_at, since a pasted "
                   "row carries no field lineage - which is exactly the gap section 3 of the study measures.")

with tab_study:
    st.image("audit.png")
    st.code(open("evidence.txt").read())
