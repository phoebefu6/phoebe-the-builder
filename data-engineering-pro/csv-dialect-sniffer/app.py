"""Streamlit UI for the CSV dialect sniffer.

Layout follows the module's thesis: the verdict first, the candidates it could not
choose between second, and the parsed table last - so an undecidable file cannot
be mistaken for a decided one just because a table rendered underneath it.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import sniff
import streamlit as st

st.set_page_config(page_title="CSV Dialect Sniffer", layout="wide")

st.title("CSV Dialect Sniffer")
st.caption(
    "`csv.Sniffer().sniff()` returns a dialect or raises - it has no third answer. "
    "This reports every dialect that parses the file cleanly, and says so when the "
    "bytes do not choose between them."
)

SAMPLES = sniff.sample_files()
BLURB = {
    "sensor.csv": "no header, comma decimals - two clean parses, 3 or 4 columns",
    "sales_eu.csv": "the same export with a header row, which settles it",
    "cp1252.csv": "Windows smart quotes - utf-8 is provably ruled out",
    "utf8_umlaut.csv": "utf-8 that latin-1 will mis-decode without raising",
    "bom.csv": "utf-8 BOM that becomes part of the first column name",
    "years.csv": "column labels that are integers",
    "alltext.csv": "all text, no header - undecidable by construction",
    "mac.csv": "bare \\r line endings",
    "quoted.csv": "a newline and a delimiter inside quoted fields",
    "dutch.csv": "apostrophe surnames - the wrong quotechar eats a record",
    "late.csv": "clean for the first kilobyte, then a quoted delimiter",
}

with st.sidebar:
    st.header("Input")
    mode = st.radio("Source", ["Sample file", "Paste text", "Upload"], index=0)
    raw: Optional[bytes] = None
    if mode == "Sample file":
        name = st.selectbox("Sample", list(SAMPLES), format_func=lambda n: n)
        st.caption(BLURB.get(name, ""))
        raw = SAMPLES[name]
    elif mode == "Paste text":
        name = "<pasted>"
        pasted = st.text_area("CSV text", height=200, value="a,b;c,d\n1,2;3,4\n5,6;7,8\n")
        raw = pasted.encode("utf-8")
        st.caption("Pasted text is encoded as utf-8, so the encoding experiments need a file.")
    else:
        name = "<upload>"
        up = st.file_uploader("CSV file", type=["csv", "tsv", "txt"])
        raw = up.read() if up else None

if not raw:
    st.info("Pick a sample, paste some text, or upload a file.")
    st.stop()

a = sniff.audit(raw, name)

# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #

badge = {"unambiguous": "✅", "contested": "⚠️", "undetermined": "❔"}[a.delimiter.status]
if a.decided:
    st.success("**Determined by the bytes.** Every question below has one answer.")
else:
    st.warning("**Not determined by the bytes.** At least one answer below is a choice, not a fact.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("delimiter", "{0} {1}".format(badge, a.delimiter.status))
c2.metric("encoding", a.encoding.verdict)
c3.metric("header", a.header.status if a.header else "-")
c4.metric("terminator", a.terminator.verdict if a.terminator else "-")

st.markdown("**{0}**".format(a.delimiter.reason))
if a.sniffer is not None:
    st.caption("`csv.Sniffer().sniff()` on this file picks {0!r}.".format(a.sniffer))
else:
    st.caption("`csv.Sniffer().sniff()` raises on this file.")

if a.notes:
    with st.expander("What this file does not tell you ({0})".format(len(a.notes)), expanded=True):
        for n in a.notes:
            st.markdown("- " + n)

# --------------------------------------------------------------------------- #
# candidates
# --------------------------------------------------------------------------- #

st.subheader("Candidate dialects")
rows: List[dict] = []
for s in sorted(a.delimiter.all_shapes, key=lambda s: (not s.viable, -s.modal, s.delimiter)):
    rows.append({
        "dialect": s.label,
        "viable": "yes" if s.viable else "no",
        "records": s.records,
        "fields": s.modal,
        "consistent": round(s.consistency, 3),
        "nl in field": s.fields_with_newline,
        "why": s.reason,
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if a.delimiter.status == "contested":
    st.error(
        "Two or more of these parse the file with no ragged row. Picking one is a tie-break, "
        "not a detection. The preferred candidate below is chosen by column count, then record "
        "count, then delimiter order - all preferences, none of them evidence."
    )

# --------------------------------------------------------------------------- #
# encoding detail
# --------------------------------------------------------------------------- #

st.subheader("Encoding")
enc_rows: List[dict] = []
for e in sniff.ENCODINGS:
    ok = a.encoding.decodes[e]
    sample = ""
    if ok:
        lines = a.encoding.texts[e].splitlines()
        sample = (lines[1] if len(lines) > 1 else (lines[0] if lines else ""))[:40]
    enc_rows.append({
        "encoding": e,
        "decodes": "yes" if ok else "NO",
        "row 1 as decoded": repr(sample) if ok else "-",
        "credible": "yes" if e in a.encoding.plausible else "no",
        "caveat": next((w for x, w in a.encoding.not_evidence if x == e), ""),
    })
st.dataframe(pd.DataFrame(enc_rows), use_container_width=True, hide_index=True)
st.caption(
    "A failed decode is a fact about the bytes. A successful one usually is not: "
    "latin-1 maps all 256 byte values and cannot fail on any input. "
    "{0} encodings decode this file to {1} different strings.".format(
        len(a.encoding.survived), a.encoding.distinct_texts)
)

# --------------------------------------------------------------------------- #
# header detail
# --------------------------------------------------------------------------- #

if a.header:
    st.subheader("Header")
    st.markdown("**{0}** - {1}".format(a.header.status, a.header.reason))
    st.caption(
        "`csv.Sniffer().has_header()` answers {0}. There is only one decidable direction: "
        "a first row that is text where the body is numeric or dated cannot be a data row. "
        "The converse is not decidable, so this module never claims a header is absent.".format(
            a.header.sniffer)
    )
    if a.header.first_types:
        st.dataframe(pd.DataFrame({
            "column": list(range(len(a.header.first_types))),
            "row 0 type": a.header.first_types,
            "body type": a.header.body_types,
        }), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# the table, last
# --------------------------------------------------------------------------- #

st.subheader("Parsed under the preferred dialect")
if a.delimiter.preferred:
    p = a.delimiter.preferred
    text = a.encoding.texts.get(a.encoding.verdict) or a.encoding.texts.get("utf-8") or ""
    parsed = [r for r in sniff.parse(text, p.delimiter, p.quotechar) if r]
    if a.header and a.header.status == "header" and len(parsed) > 1:
        df = pd.DataFrame(parsed[1:], columns=parsed[0])
    else:
        df = pd.DataFrame(parsed)
        st.caption("Rendered with no header row, because the file does not establish one.")
    st.dataframe(df, use_container_width=True)
    st.caption("dialect: {0} | {1} records x {2} fields".format(p.label, p.records, p.modal))
else:
    st.warning("No candidate dialect parses this file consistently, so there is no table to show.")

with st.expander("Raw bytes"):
    st.code(repr(raw[:600]) + (" ..." if len(raw) > 600 else ""), language="text")
