"""Streamlit front end for the line-ending audit.

Paste bytes (or pick one of the shipped blobs) and the app shows what each
runtime thinks the lines are. It never shows a line count without showing that
the count is a choice.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from lineends import (
    CORPUS,
    SPLITTERS,
    Blob,
    Verdict,
    chunk_drift,
    cr_contamination,
    detect_first,
    detect_majority,
    detect_strict,
    diff_blast,
    eol_histogram,
    findings,
    lines,
    naive_chunk_reader,
    roundtrip,
    trailing_cr_lines,
    verdict,
)

st.set_page_config(page_title="Line-Ending Detector", page_icon="⏎", layout="wide")

BADGE = {
    Verdict.AGREED: ("#e7f2e8", "#2f6b39", "agreed - every splitter returns the same lines"),
    Verdict.CONTENT_DRIFT: ("#fbeeda", "#8a5410",
                            "content-drift - same count, different bytes on the lines"),
    Verdict.COUNT_DRIFT: ("#f3e6da", "#a5551c",
                          "count-drift - they do not agree how many lines there are"),
    Verdict.DATA_SPLIT: ("#f9e3e0", "#a5291c",
                         "data-split - a terminator inside a value becomes a row"),
}
SEV_ICON = {"blocking": "🔴", "silent": "🟠", "advisory": "🔵"}

st.title("A file has no lines in it - a splitter makes them")
st.caption(
    "`wc -l` counts LF bytes. Python's text mode rewrites CRLF and CR to LF "
    "before your code sees the string. `str.splitlines()` also breaks on "
    "vertical tab, form feed, NEL and U+2028. A CSV reader keeps a CRLF that "
    "sits inside quotes. Same bytes, four different files."
)

with st.sidebar:
    st.header("Input")
    choice = st.selectbox(
        "Shipped blob", [b.label for b in CORPUS], index=3,
        help="Each one is an export somebody actually receives.",
    )
    picked = next(b for b in CORPUS if b.label == choice)
    st.caption(picked.note)
    custom = st.text_area(
        "Or paste your own (use \\r and \\n escapes)",
        value="",
        height=140,
        placeholder='id,name\\r\\n1,Alice\\r\\n2,"Smith\\rJones"\\n',
    )
    chunk = st.slider("Streaming chunk size", 2, 32, 8)

if custom.strip():
    data = custom.encode().decode("unicode_escape").encode("latin-1", "replace")
    blob = Blob(0, "pasted", data, "your bytes")
else:
    blob = picked

v = verdict(blob)
bg, fg, label = BADGE[v]
st.markdown(
    f"<div style='background:{bg};color:{fg};padding:12px 14px;border-radius:9px'>"
    f"<b>{blob.label}</b> - {label}</div>",
    unsafe_allow_html=True,
)
st.code(blob.one_line, language=None)

# -- 1. line counts ---------------------------------------------------------

st.subheader("1. What each runtime thinks the lines are")
rows = []
for s in SPLITTERS:
    got = lines(blob, s)
    rows.append(
        {
            "splitter": s.key,
            "models": s.models,
            "lines": len(got),
            "trailing CR": len(trailing_cr_lines(blob, s)),
            "first line": got[0].decode("utf-8", "replace") if got else "",
            "last line": got[-1].decode("utf-8", "replace") if got else "",
        }
    )
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)
lo, hi = df["lines"].min(), df["lines"].max()
if lo != hi:
    st.error(
        f"The same bytes are **{lo} to {hi} lines** depending on who reads them. "
        "Any row count, checksum or 'expected records' check inherits this."
    )
else:
    st.info(f"All ten agree on {lo} lines. Check the contents column - agreement "
            "on the count is not agreement on the lines.")

# -- 2. the invisible CR ----------------------------------------------------

st.subheader("2. The carriage return that is still in the value")
contaminated = {s.key: trailing_cr_lines(blob, s) for s in SPLITTERS}
if any(contaminated.values()):
    for key, lns in contaminated.items():
        if lns:
            st.markdown(f"**{key}** - {len(lns)} line(s) end with a CR")
            st.code("\n".join(repr(ln) for ln in lns[:5]), language=None)
    st.warning(
        "`int('2\\r')` raises. `'Bob\\r' == 'Bob'` is False. Both print "
        "identically in a log, an error message and a screenshot."
    )
else:
    st.success("No splitter leaves a CR on a line for this input.")

# -- 3. roundtrip -----------------------------------------------------------

st.subheader("3. Read it, write it back")
rt_rows = []
for s in SPLITTERS:
    rt = roundtrip(blob, s)
    rt_rows.append(
        {
            "splitter": s.key,
            "bytes changed": "yes" if rt.changed else "no",
            "CSV rows before": rt.before,
            "CSV rows after": rt.after,
            "landed inside a value": "YES" if rt.inside_value else "",
        }
    )
st.dataframe(pd.DataFrame(rt_rows), use_container_width=True, hide_index=True)
st.caption(
    "A change at the end of a line is formatting. A change that moves the CSV "
    "row count landed inside a value. Text mode is a transformation, not a read."
)

# -- 4. diff ----------------------------------------------------------------

st.subheader("4. What a one-field edit looks like in the diff")
row = next((t for t in diff_blast([blob]) if t[0].label == blob.label), None)
if row:
    _b, alone, normalised = row
    c1, c2 = st.columns(2)
    c1.metric("lines changed by the edit", alone)
    c2.metric("...if the commit also normalises endings", normalised,
              delta=normalised - alone if normalised != alone else None)
    if normalised > alone:
        st.warning(
            "This is what `* text=auto` does the first time it is switched on: "
            "the real change is one line, the review is the whole file, and "
            "`git blame` now points at the conversion commit."
        )

# -- 5. streaming -----------------------------------------------------------

st.subheader(f"5. A chunked reader at {chunk} bytes")
correct = lines(blob, next(s for s in SPLITTERS if s.key == "py_universal"))
chunked = naive_chunk_reader(blob.data, chunk)
c1, c2 = st.columns(2)
c1.write("**Correct**")
c1.code("\n".join(repr(x) for x in correct), language=None)
c2.write(f"**Chunked at {chunk}**")
c2.code("\n".join(repr(x) for x in chunked), language=None)
if chunked != correct:
    st.error(
        "A CRLF straddling the boundary leaves the CR at the end of one chunk "
        "and the LF at the start of the next. Correct at most buffer sizes, "
        "which is exactly why it survives testing."
    )
else:
    st.success("This buffer size happens to be safe for this file. Try another.")

# -- 6. detection -----------------------------------------------------------

st.subheader("6. 'Detect the line ending'")
h = eol_histogram(blob.data)
c1, c2, c3 = st.columns(3)
c1.metric("CRLF", h["CRLF"])
c2.metric("LF", h["LF"])
c3.metric("CR", h["CR"])
st.write(
    f"- first seen: **{detect_first(blob.data) or 'none'}**\n"
    f"- majority: **{detect_majority(blob.data) or 'none'}**\n"
    f"- strict: **{detect_strict(blob.data) or 'refuses - the file uses more than one'}**"
)
st.caption(
    "A detector that always returns one terminator is reporting a summary as if "
    "it were a fact. The honest return value is the histogram above."
)

# -- 7. findings ------------------------------------------------------------

st.subheader("7. Findings across the whole shipped corpus")
for f in findings():
    with st.expander(f"{SEV_ICON[f.severity]} {f.severity} · {f.title}"):
        st.write(f.detail)

st.caption(
    f"{len(chunk_drift())} chunk-boundary failures, "
    f"{sum(cr_contamination().values())} contaminated lines across the corpus. "
    "The fix is on the write path: read with `newline=''` and a parser that "
    "knows about quoting, write with one chosen terminator, and normalise once "
    "in its own commit with `.gitattributes` alongside."
)
