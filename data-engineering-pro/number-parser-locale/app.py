"""Paste a numeric string or a whole column; see every number it could be."""

from __future__ import annotations

from typing import List

import numlocale as N
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Number Parser Locale Audit", layout="wide")

VERDICT_COLOUR = {
    "agreed": "#2a7f62", "accept-drift": "#1d4ed8", "value-drift": "#c2410c",
    "magnitude-drift": "#c2410c", "silent-zero": "#7c1d6f",
    "sign-loss": "#7c1d6f", "sign-drift": "#7c1d6f", "rejected-by-all": "#94a3b8",
}

st.title("A numeric string does not contain a number")
st.caption(
    "A reader assigns one. Change the locale, the grouping rule, the strictness "
    "or the scanner underneath and the same bytes become a different quantity - "
    "usually with no error raised."
)

tab_one, tab_col, tab_border, tab_corpus = st.tabs(
    ["One string", "A whole column", "Border crossing", "The corpus"])


# --------------------------------------------------------------------------
with tab_one:
    raw = st.text_input("Numeric string, exactly as received", value="1.234")
    st.caption("Try: 1,234 - 1.234,56 - 12,34,567 - 1_000 - (1,234) - 1234- - "
               "9007199254740993 - an empty box")

    case = N.Case("input", raw, "pasted")
    row = N.read_all([case])["input"]
    v = N.verdict_for(case, row)

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("distinct readings", v.n_distinct)
    c2.metric("readers accepting", "%d / %d" % (len(v.accepted), len(N.reader_names())))
    if v.ratio is not None:
        c3.metric("extremes are apart by", "%s x" % N._fmt_ratio(v.ratio))

    st.markdown("**Verdict:** <span style='color:%s;font-weight:700'>%s</span>"
                % (VERDICT_COLOUR[v.verdict], v.verdict), unsafe_allow_html=True)
    for f in v.flags:
        st.warning(f)

    rows = []
    for name in N.reader_names():
        r = row[name]
        rows.append({
            "reader": name,
            "reads it as": r.display(),
            "status": {"ok": "accepted", N.REJECTED: "refused",
                       N.UNAVAILABLE: "not installed"}[r.status],
            "note": r.note,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if v.n_distinct > 1:
        st.error(
            "This string has %d defensible readings: %s. Nothing in the string "
            "says which was meant." % (
                v.n_distinct,
                ", ".join(format(d.normalize(), "f") for d in v.distinct)))


# --------------------------------------------------------------------------
with tab_col:
    st.subheader("Which locale could have written this column?")
    st.caption(
        "Only a reader that REFUSES carries information: a prefix parser accepts "
        "every string, so it never eliminates a candidate. A locale survives if "
        "it reads every row.")
    text = st.text_area("One value per line",
                        value="1.234\n2.500\n3.000\n1.750", height=170)
    strict = st.checkbox("strict grouping check", value=True,
                         help="Validates that separators fall where the locale's "
                              "pattern puts them. See the Border crossing tab "
                              "for what strict mode also refuses.")
    values: List[str] = [ln for ln in text.splitlines() if ln.strip() != ""]

    if values:
        hyps = N.locale_hypotheses(values, strict)
        d = N.decide_column(values, strict)
        badge = {"decided": ":green[DECIDED]", "ambiguous": ":red[AMBIGUOUS]",
                 "no-locale-fits": ":gray[NO LOCALE FITS]"}[d.verdict]
        st.markdown("### %s" % badge)
        if d.verdict == "ambiguous":
            st.error("%d locales read every row and they do not agree on the total: %s%s"
                     % (len(d.surviving),
                        ", ".join("%s = %s" % (k, v) for k, v in d.totals.items()),
                        ""
                        if d.spread is None else
                        "  (%s x apart)" % N._fmt_ratio(d.spread)))
        elif d.verdict == "no-locale-fits":
            st.warning("No locale reads every row. The column is not uniformly "
                       "formatted, or it is not numeric.")
        else:
            st.success("Total: %s" % list(d.totals.values())[0])

        st.dataframe(pd.DataFrame([{
            "locale": h.locale,
            "survives": "yes" if h.survives else "no",
            "eliminated by row": h.killed_by or "",
            "total": str(h.total) if h.total is not None else "",
        } for h in hyps]), width="stretch", hide_index=True)

        a = N.audit_column(values)
        st.markdown("**Findings**")
        for f in a.findings:
            st.write("- %s" % f)
        st.caption("Readers accepting every row: %d of %d. That number is not "
                   "evidence - see the note above."
                   % (len(a.readers_that_take_every_row), len(N.reader_names())))


# --------------------------------------------------------------------------
with tab_border:
    st.subheader("Write it in one locale, read it in another")
    st.caption("Nothing in a CSV records which locale wrote it, so the reader "
               "supplies its own.")
    cross = N.crossings()
    k1, k2, k3 = st.columns(3)
    k1.metric("correct", sum(1 for c in cross if c.status == "ok"))
    k2.metric("refused (recoverable)", sum(1 for c in cross if c.status == "error"))
    k3.metric("silently wrong", sum(1 for c in cross if c.status == "wrong"))

    st.markdown("**Every silently wrong run**")
    st.dataframe(pd.DataFrame([{
        "written by": c.wrote, "read by": c.read, "strict": c.strict,
        "rendered": c.rendered, "should be": str(c.target), "got": str(c.got),
        "factor": N._fmt_ratio(c.ratio) if c.ratio else "",
    } for c in cross if c.status == "wrong"]),
        width="stretch", hide_index=True)

    st.markdown("**The diagonal: the locale that wrote it is the one reading it**")
    diag = N.own_output_roundtrip()
    bad = [c for c in diag if c.status != "ok"]
    st.error("%d of %d fail, and all of them are strict-mode refusals of a "
             "correctly formatted amount." % (len(bad), len(diag)))
    st.dataframe(pd.DataFrame([{
        "locale": c.read, "it wrote": c.rendered, "strict": c.strict,
        "outcome": c.status,
    } for c in bad]), width="stretch", hide_index=True)
    st.caption(
        "Two causes: (a) Babel 2.11's strict check re-formats and compares "
        "strings, and format_decimal normalises '1,234.50' to '1,234.5', so a "
        "fixed-2dp money column is refused; (b) a pattern like '#,##0.00' can "
        "override a locale's own grouping - en_IN groups at 2,2,3, emits "
        "'1,234,567.89' under that pattern, and then refuses it.")


# --------------------------------------------------------------------------
with tab_corpus:
    st.subheader("35 strings a pipeline actually receives")
    cases = N.corpus()
    table = N.read_all(cases)
    verds = N.all_verdicts(cases, table)
    show_only_drift = st.checkbox("only strings with more than one reading", value=False)
    rows = []
    for v in verds:
        if show_only_drift and v.n_distinct < 2:
            continue
        rec = {"string": v.case.escaped(), "provenance": v.case.provenance,
               "verdict": v.verdict, "readings": v.n_distinct,
               "apart by": N._fmt_ratio(v.ratio) if v.ratio else ""}
        for n in N.reader_names():
            rec[n] = table[v.case.name][n].display()
        rows.append(rec)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.image("locale_audit.png",
             caption="Run `python make_chart.py` to regenerate.")
