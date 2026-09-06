"""Streamlit front end for the duration audit.

Type one duration string, or paste a config dump. The app never shows a single
number without showing who disagrees with it.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import streamlit as st
from durations import (
    CORPUS,
    DAY24,
    DEFAULT_ANCHORS,
    GRAMMARS,
    REFERENCE_ANCHOR,
    Verdict,
    _fmt,
    audit,
    audit_corpus,
    best_single_grammar,
    safe_form,
)

st.set_page_config(page_title="Duration Parser", page_icon="⏱", layout="wide")

BADGE = {
    Verdict.EXACT: ("#e7f2e8", "#2f6b39", "exact - every parser that accepts it agrees, no anchor needed"),
    Verdict.ANCHORED: ("#e8eff2", "#2d5a68", "anchored - the length depends on when you start"),
    Verdict.AMBIGUOUS: ("#fbeeda", "#8a5410", "ambiguous - two parsers accept it and return different numbers"),
    Verdict.REJECTED: ("#f9e3e0", "#a5291c", "rejected - no modelled parser accepts it"),
}
SEV_ICON = {"blocking": "🔴", "silent": "🟠", "advisory": "🔵"}

st.title("A duration string is not a number")
st.caption(
    "Eight conforming parsers, one string. `parse(text) -> timedelta` cannot say "
    "that two of them returned different numbers, because it has one number to return."
)

with st.sidebar:
    st.header("How to read this")
    st.markdown(
        "- **exact** is the only verdict safe to convert to a `timedelta`\n"
        "- **ambiguous** means every parser involved *succeeded* and they disagree\n"
        "- **anchored** means the answer needs an instant, not just the text"
    )
    st.divider()
    st.caption(f"reference anchor `{REFERENCE_ANCHOR:%Y-%m-%d %H:%M %Z}`, "
               f"{len(DEFAULT_ANCHORS)} anchors in the sweep")
    for g in GRAMMARS:
        st.markdown(f"**{g.name}** · _{g.kind}_  \n<span style='font-size:0.78rem;color:#666'>"
                    f"{g.note}</span>", unsafe_allow_html=True)

tab_one, tab_corpus = st.tabs(["One string", "A whole config"])

with tab_one:
    col_in, col_pick = st.columns([2, 1])
    with col_pick:
        preset = st.selectbox("or pick a known trap", ["-"] + [t for t, _ in CORPUS], index=0)
    with col_in:
        text = st.text_input("duration string", value="1h30" if preset == "-" else preset)
    if preset != "-" and preset != text:
        text = preset

    a = audit(text)
    bg, fg, label = BADGE[a.verdict]
    st.markdown(
        f"<div style='background:{bg};color:{fg};padding:0.6rem 0.9rem;border-radius:6px;"
        f"font-weight:600'>{a.verdict.value.upper()} &mdash; {label}</div>",
        unsafe_allow_html=True,
    )

    rows: List[Dict[str, str]] = []
    for r in a.readings:
        g = next(x for x in GRAMMARS if x.name == r.grammar)
        rows.append({
            "grammar": r.grammar,
            "kind": g.kind,
            "reads it as": _fmt(r.resolve(REFERENCE_ANCHOR)) if r.ok else "-",
            "seconds": f"{r.resolve(REFERENCE_ANCHOR):,.6g}" if r.ok else "-",
            "needs an anchor": "yes" if r.anchored else ("no" if r.ok else "-"),
            "why not": "" if r.ok else (r.error or ""),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if a.accepted:
        c1, c2, c3 = st.columns(3)
        c1.metric("parsers that accept it", f"{len(a.accepted)} of {len(a.readings)}")
        c2.metric("distinct values", len(a.distinct_values()))
        ratio = a.spread_ratio or 1.0
        c3.metric("highest / lowest", f"{ratio:,.0f}x" if ratio >= 10 else f"{ratio:.3g}x")

    if a.findings:
        st.subheader("Findings")
        for f in a.findings:
            st.markdown(f"{SEV_ICON[f.severity]} **{f.code}**"
                        + (f" · `{f.grammar}`" if f.grammar else "")
                        + f"  \n{f.message}")

    if any(r.anchored for r in a.accepted):
        st.subheader("The same string, started at 17 different instants")
        r = next(r for r in a.accepted if r.anchored)
        sweep = pd.DataFrame(
            {"anchor": [f"{x:%Y-%m-%d %H:%M}" for x in DEFAULT_ANCHORS],
             "elapsed days": [r.resolve(x) / DAY24 for x in DEFAULT_ANCHORS]}
        ).set_index("anchor")
        st.bar_chart(sweep, height=240)
        st.caption(f"`{r.grammar}` reads `{text}` as {r.nominal} plus {_fmt(r.exact_s)}; "
                   "the elapsed length is not a property of the string.")

    if a.accepted:
        low = min(r.resolve(REFERENCE_ANCHOR) for r in a.accepted)
        st.info(f"Unambiguous rewrite of the lowest reading: `{safe_form(low)}` "
                "— integer seconds, no calendar unit, no bare number, no colon.")

with tab_corpus:
    default = "\n".join(t for t, _ in CORPUS)
    dump = st.text_area("one duration string per line", value=default, height=260)
    texts = [ln.strip() for ln in dump.splitlines() if ln.strip()]
    if texts:
        rep = audit_corpus(texts)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("exact", rep.verdicts["exact"])
        c2.metric("anchored", rep.verdicts["anchored"])
        c3.metric("ambiguous", rep.verdicts["ambiguous"], help="every parser succeeded and they disagree")
        c4.metric("rejected", rep.verdicts["rejected"])
        name, acc, wrong = best_single_grammar(rep)
        st.warning(
            f"The best single parser here is **{name}**: it accepts {acc} of {rep.total} strings, "
            f"and {wrong} of those it reads differently from another parser. "
            f"No parser reads all {rep.total}."
        )
        table = pd.DataFrame([
            {
                "string": a.text,
                "verdict": a.verdict.value,
                "readers": len(a.accepted),
                "lowest": _fmt(a.min_s) if a.accepted else "-",
                "highest": _fmt(a.max_s) if a.accepted else "-",
                "x-grammar": (f"{a.spread_ratio:,.0f}" if (a.spread_ratio or 0) >= 10
                              else f"{a.spread_ratio:.4g}") if a.accepted else "-",
                "findings": len(a.findings),
            }
            for a in rep.audits
        ])
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.subheader("Coverage per parser")
        st.bar_chart(pd.DataFrame({"accepted": rep.accepted_by}), height=260)
