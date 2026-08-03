"""Streamlit front end for the DSAR extractor."""

from __future__ import annotations

import io
import json
from datetime import date

import pandas as pd
import streamlit as st

from dsar import (
    DIRECT,
    MENTION,
    RETAIN,
    SPEC_BY_NAME,
    SUBJECT_EMAIL,
    SUBJECT_MAP,
    build_corpus,
    coverage,
    disclosure_pack,
    erasure_plan,
    extract,
    naive_extract,
    naive_fk_sweep,
    resolve_identity,
    weak_link_cost,
)

st.set_page_config(page_title="DSAR Extractor", layout="wide")
st.title("DSAR Extractor")
st.caption(
    "One subject access request. Resolve the person across 11 tables, disclose what is theirs, "
    "withhold what is someone else's, and say which rows you are not allowed to delete."
)

corpus = build_corpus()

with st.sidebar:
    st.header("The request")
    seed_email = st.text_input("Identifier on the request form", SUBJECT_EMAIL)
    as_of = st.date_input("Assess retention as of", date(2026, 8, 3))
    st.divider()
    st.header("Warehouse")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "table": spec.name,
                    "rows": len(corpus[spec.name]),
                    "scope": spec.category,
                    "erasure": spec.erasure,
                    "retention": spec.retention.years if spec.retention else None,
                }
                for spec in SUBJECT_MAP
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

ident = resolve_identity(corpus, seed_email)
hits = extract(corpus, ident)
cov = coverage(corpus, hits, seed_email)
plan = erasure_plan(hits, corpus, today=as_of)

if not hits:
    st.warning("No rows resolved for that identifier. Try the default subject email.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Naive query", f"{cov['naive']} rows")
c2.metric("Resolved extract", f"{cov['resolved']} rows", f"+{cov['missed_by_naive']}")
c3.metric("Text-only rows", cov["mention_only"], help="No key join reaches these")
c4.metric("Redacted", cov["shared"], help="Rows containing a third party")

tabs = st.tabs(
    ["Identity", "Disclosure pack", "What naive missed", "Over-collection", "Erasure plan"]
)

with tabs[0]:
    st.subheader("Resolved identity, with evidence")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "use": "join" if k.usable else "HOLD - human decision",
                    "strength": k.strength,
                    "key_type": k.key_type,
                    "value": k.value,
                    "evidence": k.evidence,
                }
                for k in ident.keys
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    if ident.variants:
        st.markdown("**Stored spellings of the same mailbox** - what an exact match misses:")
        st.code("\n".join(ident.variants))

    cost = weak_link_cost(corpus, ident)
    if cost:
        st.error(
            f"Auto-including the weak link would have disclosed **{cost['orders']} orders, "
            f"{cost['payments']} payments and {cost['order_items']} line items belonging to "
            f"{cost['other_people']} different person(s)** to this requester. Over-resolution "
            "is a breach in the opposite direction, which is why weak links abstain."
        )

with tabs[1]:
    st.subheader("What gets handed to the requester")
    pack = disclosure_pack(corpus, ident, hits)
    for table in sorted(pack):
        redacted = sum(1 for h in hits if h.table == table and h.shared)
        label = f"{table} - {len(pack[table])} rows"
        if redacted:
            label += f"  ({redacted} redacted)"
        with st.expander(label, expanded=table in ("customers", "referrals")):
            st.dataframe(pd.DataFrame(pack[table]), hide_index=True, use_container_width=True)

    buffer = io.StringIO()
    json.dump(pack, buffer, indent=2, default=str)
    st.download_button(
        "Download disclosure pack (JSON)",
        buffer.getvalue(),
        file_name="dsar_disclosure.json",
        mime="application/json",
    )

with tabs[2]:
    st.subheader("Rows the ordinary query would have left out")
    naive = naive_extract(corpus, seed_email)
    missed = [h for h in hits if (h.table, h.pk) not in naive]
    st.dataframe(
        pd.DataFrame(
            [
                {"table": h.table, "pk": h.pk, "found_by": h.how, "why": h.reason}
                for h in missed
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "The baseline is not a strawman: it is a case-insensitive email match followed by a "
        "customer_id join. It still misses plus-addressed mailboxes, pre-login activity, and "
        "anything that names the subject only inside free text."
    )

with tabs[3]:
    st.subheader("What following every foreign key would have swept in")
    sweep = naive_fk_sweep(corpus, hits)
    ref = {k: v for k, v in sweep.items() if not k.startswith("_")}
    if ref:
        st.table(
            pd.DataFrame([{"reference table": k, "rows pulled in": v} for k, v in ref.items()])
        )
        st.info(
            "A product row is not personal data about anybody. It is reachable from the "
            "subject's order, which is not the same thing - so the edge is declared "
            "`reference` and the traversal stops."
        )
    if "_reverse_order_items" in sweep:
        st.error(
            f"One careless step further - joining those products back to `order_items` - reaches "
            f"**{sweep['_reverse_order_items']} line items belonging to "
            f"{sweep['_reverse_customers']} other customers.** This is how a DSAR export becomes "
            "a data breach."
        )

with tabs[4]:
    st.subheader("Erasure plan - not the same set of rows")
    counts: dict = {}
    for action in plan:
        counts[action.action] = counts.get(action.action, 0) + 1
    st.table(pd.DataFrame([{"action": k, "rows": v} for k, v in sorted(counts.items())]))
    blocked = [a for a in plan if a.action == RETAIN]
    st.warning(
        f"**{len(blocked)} of {len(hits)} disclosed rows cannot be deleted.** Art. 17 yields to a "
        "statutory retention period. Telling the requester the data is gone when it is not is its "
        "own compliance failure, so the plan names the basis and the release date."
    )
    st.dataframe(
        pd.DataFrame(
            [{"table": a.table, "pk": a.pk, "action": a.action, "basis": a.basis} for a in plan]
        ),
        hide_index=True,
        use_container_width=True,
        height=420,
    )

st.divider()
st.caption(
    "Corpus is generated deterministically - no database, no network, no real personal data. "
    "The subject map in `dsar.py` is the part you replace for a real warehouse: declare each "
    "table's subject keys, its scoped-via edge, whether it is reference data, and its retention rule."
)
