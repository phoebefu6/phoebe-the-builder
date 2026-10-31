"""Streamlit front end: paste an export's quasi-identifier columns, see how many rows are unique and what each
generalisation buys.

Tab one measures your own rows: classes, smallest class, unique rows, rows below k, at the postcode and
date-of-birth granularity you pick. Tab two shows the study's ladder and figure. Verified headless with
`streamlit.testing.v1.AppTest`, widgets BY KEY.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from datetime import date

import explore as E
import streamlit as st

st.set_page_config(page_title="k-anonymity check", layout="wide")
st.title("The anonymised export was re-identified: how few columns it takes")
tab_try, tab_study = st.tabs(["Check your rows", "The study"])


@st.cache_data
def default_rows() -> str:
    ex = E.sample(E.population(), E.EXPORT_FRAC)[:300]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["postcode", "dob", "sex"])
    for _, pc, dob, sex, _ in ex:
        w.writerow([pc, dob.isoformat(), sex])
    return buf.getvalue()


with tab_try:
    raw = st.text_area("CSV with columns postcode, dob (YYYY-MM-DD), sex", default_rows(), height=180, key="rows")
    c1, c2, c3 = st.columns(3)
    pl = c1.selectbox("postcode", list(E.POSTCODE_LEVELS), key="pl")
    dl = c2.selectbox("date of birth", list(E.DOB_LEVELS), key="dl")
    sl = c3.selectbox("sex", list(E.SEX_LEVELS), key="sl")
    k = st.slider("k", 2, 20, E.K, key="k")
    rows = []
    for rec in csv.DictReader(io.StringIO(raw)):
        try:
            rows.append((0, rec["postcode"].strip(), date.fromisoformat(rec["dob"].strip()), rec["sex"].strip(), ""))
        except (KeyError, ValueError, AttributeError):
            continue
    if not rows:
        st.info("Paste at least one row with postcode, dob and sex.")
    else:
        keys = [E.key(r, (pl, dl, sl)) for r in rows]
        c = Counter(keys)
        unique = sum(1 for kk in keys if c[kk] == 1)
        below = sum(1 for kk in keys if c[kk] < k)
        st.metric("rows", len(rows))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("classes", len(c))
        m2.metric("smallest class", min(c.values()))
        m3.metric("unique rows", unique)
        m4.metric(f"rows below k = {k}", below)
        if unique:
            st.warning(
                f"{unique} of {len(rows)} rows are unique on this key. Anyone who knows a person is in this file can "
                f"pick them out; whether a stranger can depends on the population, which is not in the file."
            )
        elif below:
            st.warning(f"No row is unique, but {below} rows sit in classes smaller than {k}; suppress them or generalise further.")
        else:
            st.success(f"Every row shares its key with at least {k - 1} others in this file (k >= {k}).")

with tab_study:
    r = E.study()
    st.caption(
        f"Fictional town of {r['pop_n']:,}; a {r['export_n']:,}-row export; every number in evidence.txt. "
        f"Unique in the town is what a stranger needs; unique in the export is what an insider needs."
    )
    st.table(
        [
            {
                "generalisation": " / ".join(p["levels"]),
                "classes": p["classes"],
                "smallest class": p["k_min"],
                "unique in export": p["unique_export"],
                "unique in town": p["unique_pop"],
                "below k=5 in export": p["below_k"],
                "below k=5 in town": p["below_k_pop"],
            }
            for p in r["ladder"]
            if p["levels"][2] == "kept" or p["below_k"] == 0
        ]
    )
    st.image("audit.png")
