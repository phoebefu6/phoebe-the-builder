"""Streamlit front end for markdown-tabler.

Paste a CSV (or use the bundled lint export), pick the escaping policies, and get
the markdown table plus the list of cells that did not survive the conversion.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
from typing import List, Optional, Sequence

import pandas as pd
import streamlit as st
import tabler as T

try:
    from evidence import HAVE_PARSER, parse_back
except Exception:  # pragma: no cover
    HAVE_PARSER = False

st.set_page_config(page_title="markdown-tabler", page_icon="|", layout="wide")

SEV_ICON = {T.LOSS: "🔴", T.PORTABILITY: "🟠", T.COSMETIC: "🔵"}

st.title("What a markdown table cannot carry")
st.caption(
    "A GFM table has no error state. It truncates the row that is one cell too wide, trims the "
    "cell whose meaning is its indentation, and italicises the identifier with an underscore at "
    "each end - and reports none of it. This renders the table and returns what did not survive."
)

# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Input")
    source = st.radio(
        "Rows", ["Bundled lint export", "Paste CSV", "Upload CSV"], label_visibility="collapsed"
    )

    st.header("Policies")
    newline = st.selectbox(
        "Multi-line cells",
        ["br", "space", "strip"],
        help=(
            "br joins with an HTML <br> (renders as literal text where inline HTML is off); "
            "space flattens and loses the break; strip keeps the first line only."
        ),
    )
    escape_emphasis = st.checkbox(
        "Escape emphasis (* and _)",
        value=False,
        help="Recovers _id_field_ and 2*3*4, at the cost of backslashes in the source.",
    )
    pad_cells = st.checkbox("Pad columns to width", value=True)
    ambiguous_wide = st.checkbox(
        "Treat East Asian Ambiguous as wide",
        value=False,
        help="Greek, Cyrillic and box-drawing are one column in a Western font and two in a CJK font.",
    )


def _load() -> Optional[pd.DataFrame]:
    if source == "Paste CSV":
        text = st.text_area(
            "CSV",
            "column,type,note\nuser_id,bigint,primary key\n_meta_,json,underscores at both ends\n"
            'price|local,text,"a pipe, in a code span: `price|local`"\n  indented,text,leading spaces',
            height=170,
        )
        if not text.strip():
            return None
        return pd.read_csv(io.StringIO(text), dtype=str, skipinitialspace=False).fillna("")
    if source == "Upload CSV":
        up = st.file_uploader("CSV file", type=["csv"])
        if up is None:
            return None
        return pd.read_csv(up, dtype=str).fillna("")
    return None


rows: List[Sequence[object]]
headers: Sequence[object]

df = _load()
if df is not None:
    rows = df.values.tolist()
    headers = list(df.columns)
else:
    rows = T.SAMPLE_ROWS
    headers = T.SAMPLE_HEADERS
    if source == "Bundled lint export":
        st.caption(
            "12 findings from a code-review bot. Row 9 was written with six cells; the header has "
            "five. Every other hostile value is content a real linter emits."
        )

res = T.render(
    rows,
    headers,
    newline=newline,
    pad_cells=pad_cells,
    ambiguous_wide=ambiguous_wide,
    escape_emphasis=escape_emphasis,
)

# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------

loss = res.by_severity(T.LOSS)
port = res.by_severity(T.PORTABILITY)
cosm = res.by_severity(T.COSMETIC)

a, b, c, d = st.columns(4)
a.metric("Cells", len(rows) * len(headers))
b.metric("Loses content", len(loss))
c.metric("Portability", len(port))
d.metric("Cosmetic", len(cosm))

if loss:
    st.error(
        "%d finding%s change what the reader sees. This table is not safe to paste as-is."
        % (len(loss), "" if len(loss) == 1 else "s")
    )
elif port:
    st.warning(
        "Nothing is lost here, but %d cell%s depend%s on the renderer's settings."
        % (len(port), "" if len(port) == 1 else "s", "s" if len(port) == 1 else "")
    )
else:
    st.success("Every cell reaches the reader exactly as written.")

left, right = st.columns([1.05, 1])

with left:
    st.subheader("Markdown")
    st.code(res.markdown, language="markdown")
    st.download_button(
        "Download table.md", res.markdown, file_name="table.md", mime="text/markdown"
    )

    st.subheader("Rendered")
    st.markdown(res.markdown)

with right:
    st.subheader("What did not survive")
    if not res.findings:
        st.info("No findings.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "": SEV_ICON.get(f.severity, ""),
                        "code": f.code,
                        "where": f.where,
                        "column": f.column,
                        "detail": f.detail,
                    }
                    for f in res.findings
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    st.caption(
        "🔴 LOSS - the table does not contain what you put in.  "
        "🟠 PORTABILITY - renders here, not everywhere.  "
        "🔵 COSMETIC - output is right, the source is misaligned."
    )

    with st.expander("The three findings with no fix"):
        st.markdown(
            "- **EDGE_SPACE** - leading and trailing whitespace is trimmed by the renderer and "
            "has no escape. A code span preserves it and changes how the cell looks, which is a "
            "visual decision about somebody else's table, so it is not made for you.\n"
            "- **NEWLINE** - a table row is one line. Every option either leaves markdown "
            "(`<br>`) or loses the break. The severity follows the policy you pick, not the data.\n"
            "- **RAGGED_EXTRA** - the content never enters the table, so no escaping helps. This "
            "is why the audit runs before the render rather than diffing after it."
        )

# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------

if HAVE_PARSER:
    st.divider()
    st.subheader("Round trip")
    st.caption(
        "The audit's claim is that it is the diff a parser would give you. Rendering the table, "
        "parsing it back and comparing cell by cell is how that gets checked."
    )
    back = parse_back(res.markdown)
    head, body = back[0], back[1:]
    predicted = {(f.row, f.column) for f in res.findings}
    diffs = []
    for i, row in enumerate(body):
        for j, got in enumerate(row):
            want = T._stringify(rows[i][j], "{:g}", "") if j < len(rows[i]) else ""
            if got != want:
                codes = [
                    f.code
                    for f in res.findings
                    if (f.row, f.column) == (i, head[j]) and f.code != "WIDE_GLYPH"
                ]
                diffs.append(
                    {
                        "row": i,
                        "column": head[j],
                        "written": repr(want),
                        "read back": repr(got),
                        "audit said": ", ".join(codes) or "(UNPREDICTED)",
                    }
                )
    total = len(rows) * len(headers)
    unpredicted = sum(1 for d in diffs if d["audit said"] == "(UNPREDICTED)")
    st.write(
        "**%d of %d** cells read back byte-identical. **%d** differ, **%d** unpredicted."
        % (total - len(diffs), total, len(diffs), unpredicted)
    )
    if diffs:
        st.dataframe(pd.DataFrame(diffs), hide_index=True, width="stretch")
    dropped = [f for f in res.findings if f.code == "RAGGED_EXTRA"]
    if dropped:
        st.caption(
            "%d cell(s) are absent from this comparison because they are absent from the table: "
            "they were dropped at render time, and RAGGED_EXTRA is the only record that they "
            "existed. That is the failure a round trip can never reproduce." % len(dropped)
        )
else:
    st.divider()
    st.info("Install markdown-it-py to see the round trip:  pip install markdown-it-py")

st.divider()
st.caption(
    "Day 140 - automation-suite - "
    "[phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). "
    "`python3 evidence.py` for the full write-up, `python3 test_tabler.py` for the 38 tests."
)
