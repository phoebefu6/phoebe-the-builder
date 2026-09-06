"""Streamlit front end for the sort-order audit.

Paste a text column, pick a page size, and the app shows the ten orders, the
ties, and what pagination does to them. It never shows an order without
showing whether that order is determined.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st
from collate import (
    COLLATIONS,
    CORPUS,
    PAGE_SIZES,
    Row,
    Verdict,
    distinct_count,
    drift_matrix,
    findings,
    keyset_pagination,
    offset_pagination,
    order,
    positions,
    range_counts,
    tie_groups,
    tied_rows,
    unique_violations,
    verdict,
)

st.set_page_config(page_title="Sort-Order Drift", page_icon="🔤", layout="wide")

BADGE = {
    Verdict.STABLE_TOTAL: ("#e8eff2", "#2d5a68",
                           "stable-total - deterministic forever, but the order is not linguistic"),
    Verdict.TOTAL: ("#e7f2e8", "#2f6b39",
                    "total - linguistic and injective here: safe to paginate"),
    Verdict.TIED: ("#fbeeda", "#8a5410",
                   "tied - row order inside a tie is the plan's choice, and nothing errors"),
    Verdict.MERGING: ("#f9e3e0", "#a5291c",
                      "merging - ties are equality here, so DISTINCT, UNIQUE and row counts change"),
}
SEV_ICON = {"blocking": "🔴", "silent": "🟠", "advisory": "🔵"}


def parse_rows(text: str) -> List[Row]:
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        out.append(Row(i, line.rstrip("\n"), ""))
    return out


st.title("`ORDER BY name` is a collation, not an order")
st.caption(
    "A collation is three decisions the SQL never states: which sequence the "
    "characters are in, how many levels of difference count as a difference, and "
    "whether two strings that compare equal are the same value. The first makes "
    "two servers disagree. The second creates ties, and paginating a tie drops "
    "rows silently. The third changes how many rows a report returns."
)

with st.sidebar:
    st.header("Input")
    default = "\n".join(r.name for r in CORPUS)
    text = st.text_area("One name per line", value=default, height=320)
    page_size = st.select_slider("Page size", options=list(PAGE_SIZES), value=6)
    lo = st.text_input("Range predicate: name >=", value="A")
    hi = st.text_input("... and name <", value="N")
    st.caption(
        "The corpus ships pre-loaded. Every row in it is an ordinary name; "
        "two of them are the same string written two ways."
    )

rows = parse_rows(text)
if len(rows) < 2:
    st.warning("Two or more names needed.")
    st.stop()

# -- 1. verdict per collation ----------------------------------------------

st.subheader("1. What each collation makes of this column")
cols = st.columns(5)
for i, c in enumerate(COLLATIONS):
    v = verdict(c, rows)
    bg, fg, label = BADGE[v]
    with cols[i % 5]:
        st.markdown(
            f"<div style='background:{bg};color:{fg};padding:9px 11px;border-radius:8px;"
            f"margin-bottom:9px'><b>{c.key_name}</b><br><span style='font-size:0.78em'>"
            f"{v.value}<br>{len(tie_groups(c, rows))} tie groups, "
            f"{tied_rows(c, rows)} rows<br>DISTINCT {distinct_count(c, rows)}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(c.models)

st.caption(
    " · ".join(f"**{v.value}**: {BADGE[v][2].split(' - ', 1)[1]}" for v in Verdict)
)

# -- 2. the orders themselves ----------------------------------------------

st.subheader("2. Ten answers to the same query")
table = pd.DataFrame(
    {c.key_name: [r.display for r in order(rows, c)] for c in COLLATIONS},
    index=[f"row {i + 1}" for i in range(len(rows))],
)
st.dataframe(table, use_container_width=True, height=min(560, 38 * len(rows)))

pos = {c.key_name: positions(c, rows) for c in COLLATIONS}
movers = sorted(
    rows,
    key=lambda r: -(max(pos[c.key_name][r.id] for c in COLLATIONS)
                    - min(pos[c.key_name][r.id] for c in COLLATIONS)),
)[:6]
st.markdown("**Rows that move the most**")
st.dataframe(
    pd.DataFrame(
        [
            {
                "name": r.display,
                "first position": min(pos[c.key_name][r.id] for c in COLLATIONS),
                "last position": max(pos[c.key_name][r.id] for c in COLLATIONS),
                "spread": max(pos[c.key_name][r.id] for c in COLLATIONS)
                - min(pos[c.key_name][r.id] for c in COLLATIONS),
            }
            for r in movers
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

# -- 3. ties ----------------------------------------------------------------

st.subheader("3. The ties, and what they cost")
tied_any = False
for c in COLLATIONS:
    groups = tie_groups(c, rows)
    if not groups:
        continue
    tied_any = True
    kind = "nondeterministic" if not c.deterministic else "deterministic"
    st.markdown(f"**{c.key_name}** ({kind})")
    for g in groups:
        st.code(" = ".join(r.display for r in g), language=None)
    if not c.deterministic:
        v = unique_violations(c, rows)
        st.warning(
            f"These ties are equality: `COUNT(DISTINCT name)` reads "
            f"{distinct_count(c, rows)} instead of {len({r.name for r in rows})}, and a "
            f"`UNIQUE(name)` index rejects {len(v)} row pairs that are different strings."
        )
if not tied_any:
    st.success("No ties under any collation: every ORDER BY here is determined.")

# -- 4. pagination ----------------------------------------------------------

st.subheader(f"4. Paginating it, page size {page_size}")
pag = []
for c in COLLATIONS:
    off = offset_pagination(c, page_size, rows)
    tb = offset_pagination(c, page_size, rows, tiebreak=True)
    ks = keyset_pagination(c, page_size, rows, strict=True)
    kl = keyset_pagination(c, page_size, rows, strict=False)
    pag.append(
        {
            "collation": c.key_name,
            "OFFSET: never returned": len(off.lost),
            "OFFSET: returned twice": len(off.duplicated),
            "OFFSET + `, id`": "clean" if tb.clean else "DIRTY",
            "keyset `>`: lost": len(ks.lost),
            "keyset `>=`: repeated": len(kl.duplicated),
            "keyset `>=`: stalls": "yes" if kl.stalled else "no",
        }
    )
st.dataframe(pd.DataFrame(pag), use_container_width=True, hide_index=True)
st.caption(
    "Each page is a separate execution, so each may be handed a different "
    "physical row order. A stable sort preserves whatever it is given, so inside "
    "a tie group the physical order is the result order. `>=` keyset paging "
    "stalls even with no ties: the last row of a page always satisfies "
    "`name >= $last`."
)

worst = max(pag, key=lambda p: p["OFFSET: never returned"])
if worst["OFFSET: never returned"]:
    c = next(x for x in COLLATIONS if x.key_name == worst["collation"])
    a = offset_pagination(c, page_size, rows)
    st.error(
        f"**{c.key_name}, page size {page_size}**: never returned "
        f"{[next(r for r in rows if r.id == i).name for i in a.lost]}, returned twice "
        f"{[next(r for r in rows if r.id == i).name for i in a.duplicated]}. Every page "
        "was individually correct and nothing raised an error."
    )
else:
    st.info(
        f"OFFSET paging is exact at page size {page_size} for every collation here. "
        "Try another page size - whether a tie group straddles a boundary depends "
        "on it, which is why this bug survives testing."
    )

# -- 5. range predicate -----------------------------------------------------

st.subheader(f"5. `WHERE name >= {lo!r} AND name < {hi!r}`")
try:
    counts = range_counts(rows, lo, hi)
    st.dataframe(
        pd.DataFrame([{"collation": k, "rows matching": v} for k, v in counts.items()]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"{min(counts.values())} to {max(counts.values())} rows from the same table "
        "and the same predicate. Every A-M / N-Z split inherits this: shard keys, "
        "archive sweeps, alphabetical tabs, partition bounds."
    )
except Exception as exc:  # a bad literal is the user's, not a crash
    st.warning(f"Could not evaluate that range: {exc}")

# -- 6. drift ---------------------------------------------------------------

st.subheader("6. Pairwise drift")
m = drift_matrix(rows)
names = [c.key_name for c in COLLATIONS]
st.dataframe(
    pd.DataFrame([[m[(a, b)] for b in names] for a in names], index=names, columns=names),
    use_container_width=True,
)
st.caption(
    "Row pairs the two collations return in the opposite order. Pairs tied under "
    "either collation are excluded - they have no direction to disagree about."
)

# -- 7. findings ------------------------------------------------------------

st.subheader("7. Findings")
for f in findings(rows):
    with st.expander(f"{SEV_ICON[f.severity]} {f.severity} · {f.title}"):
        st.write(f.detail)

st.caption(
    "The fix is one clause: `ORDER BY name, id`. A unique tiebreak turns every "
    "collation above into a total order, and every pagination scheme into an exact "
    "one. What it cannot fix is the tailoring - for that, name the collation in "
    "the DDL and pin it, because glibc 2.28 changed en_US.UTF-8 and invalidated "
    "existing text indexes."
)
