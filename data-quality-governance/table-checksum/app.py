"""Streamlit front end: paste two tables as CSV, see what each of the five fingerprints says about them.

Tab one compares your own two tables. Columns are typed from the data (all whole numbers -> int, all numbers ->
money to the cent, else text); a cell of \\N is NULL, an empty cell is an empty string. Tab two shows the study.
Verified headless with `streamlit.testing.v1.AppTest`, widgets BY KEY.
"""

from __future__ import annotations

import csv
import io
from typing import Dict, List

import explore as E
import streamlit as st

st.set_page_config(page_title="table checksum", layout="wide")
st.title("Are these two tables the same? Five fingerprints, side by side")
tab_try, tab_study = st.tabs(["Compare your tables", "The study"])

A_DEFAULT = "event_id,customer,note,amount\n1,Ada Lim,gift,12.50\n2,Ben Okafor,\\N,7.00\n3,Chloe Tan,,0.30\n"
B_DEFAULT = "amount,event_id,customer,note\n0.30000000000000004,3,Chloe Tan,\n7,2,Ben Okafor,\\N\n12.5,1,Ada Lim,gift\n"


def is_num(v: str, whole: bool) -> bool:
    try:
        x = float(v)
    except ValueError:
        return False
    return x.is_integer() and "." not in v if whole else True


def parse(raw: str) -> E.Table:
    rows = list(csv.DictReader(io.StringIO(raw.strip())))
    cols = list(rows[0].keys()) if rows else []
    return cols, [dict(r) for r in rows]


def infer(tables: List[E.Table]) -> Dict[str, str]:
    cols = sorted(set().union(*[set(t[0]) for t in tables]))
    vals = {c: [r.get(c) for t in tables for r in t[1] if r.get(c) not in (None, "", E.NULL)] for c in cols}
    return {c: "int" if v and all(is_num(x, True) for x in v) else ("money" if v and all(is_num(x, False) for x in v)
            else "str") for c, v in vals.items()}


with tab_try:
    c1, c2 = st.columns(2)
    raw_a = c1.text_area("Table A (CSV with header)", A_DEFAULT, height=180, key="a")
    raw_b = c2.text_area("Table B (CSV with header)", B_DEFAULT, height=180, key="b")
    ta, tb = parse(raw_a), parse(raw_b)
    if not ta[1] or not tb[1]:
        st.info("Paste at least one row into each table.")
    elif set(ta[0]) != set(tb[0]):
        st.error(f"Different columns: {sorted(set(ta[0]) ^ set(tb[0]))}. No fingerprint can call these the same.")
    else:
        E.SCHEMA = infer([ta, tb])
        res = [{"fingerprint": k, "A": f(ta)[:20], "B": f(tb)[:20], "verdict": "match" if f(ta) == f(tb) else "differ"}
               for k, f in E.SCHEMES.items()]
        st.caption("Column types read from the data: " + ", ".join(f"{c}={t}" for c, t in E.SCHEMA.items()))
        st.table(res)
        canon = res[-1]["verdict"]
        others = [r["fingerprint"] for r in res[:-1] if r["verdict"] != canon]
        if canon == "match":
            st.success(f"Same data under the declared types. Disagreeing fingerprints: {', '.join(others) or 'none'}.")
        else:
            st.warning(f"The tables differ under the declared types. Fingerprints that call them equal: "
                       f"{', '.join(r['fingerprint'] for r in res if r['verdict'] == 'match') or 'none'}.")

with tab_study:
    st.image("audit.png")
    st.code(open("evidence.txt").read())
