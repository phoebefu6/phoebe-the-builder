"""Streamlit front end: paste phone numbers with a country field, see every normaliser's key and every reading.

Tab one normalises your own rows and flags strings that are valid in more than one country. Tab two re-runs the
study under a different default region. Verified headless with `streamlit.testing.v1.AppTest`, widgets BY KEY.
"""

from __future__ import annotations

import explore as E
import streamlit as st

st.set_page_config(page_title="Phone normaliser audit", layout="wide")
st.title("Two formats, one number, two customers: six phone normalisers against one CRM table")
tab_try, tab_hq = st.tabs(["Normalise numbers", "Your default region"])

with tab_try:
    raw = st.text_area(
        "One row per line: phone, country field (the field may be blank)",
        "612 345 678, FR\n06 12 34 56 78, FR\n+44 20 7946 0958, GB\n9123 4567, SG",
        key="rows",
    )
    rows = []
    for line in raw.split("\n"):
        if line.strip():
            phone, _, country = line.partition(",")
            rows.append((phone.strip(), country.strip().upper()))
    if not rows:
        st.info("Paste at least one phone number.")
    else:
        table, ambiguous = [], 0
        for phone, country in rows:
            n_read = len(E.readings(phone))
            ambiguous += n_read > 1
            row = {"phone": phone, "country field": country or "(blank)", "valid readings": n_read}
            row.update({k: f(phone, country, E.HQ) or "refused" for k, f in E.NORMALISERS.items()})
            table.append(row)
        st.dataframe(table, width="stretch")
        if ambiguous:
            st.warning(
                f"{ambiguous} of {len(rows)} numbers are valid in more than one country. Store them in E.164 "
                "with a + at capture time, or ask the customer which country the number is in."
            )
        else:
            st.success(
                "Every number carries its own country code (unambiguous is not the same as in service)."
            )

with tab_hq:
    hq = st.text_input("Default region an import script assumes (ISO code)", E.HQ, key="hq").strip().upper()
    if hq not in E.REGIONS:
        st.error(f"{hq!r} is not one of the {len(E.REGIONS)} regions libphonenumber knows.")
    else:
        r = E.study(hq=hq, sweep_n=0)
        st.table(
            [
                {
                    "normaliser": n,
                    "E.164 right": "key only" if p["right"] is None else f"{p['right']}/{r['n']}",
                    "refused": p["refused"],
                    "false splits": len(p["false_splits"]),
                    "false merges": len(p["false_merges"]),
                    "customers": f"{p['customers']} (truth {r['true_customers']})",
                }
                for n, p in r["per"].items()
            ]
        )
    st.image("audit.png")
