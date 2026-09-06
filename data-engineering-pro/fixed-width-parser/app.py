"""Streamlit UI: audit first, layout second, rows third.

The ordering is the argument. A fixed-width loader that shows you rows before it
shows you the verdict has already made the decision for you.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from fwf import (
    BALANCE_SPEC,
    CUSTOMER_SPEC,
    Date,
    Field,
    Implied,
    Int,
    Overpunch,
    Packed,
    RecordSpec,
    Text,
    audit,
    build_balance_file,
    build_customer_file,
    frame_records,
    parse,
)

st.set_page_config(page_title="Fixed-width parser", layout="wide")

KINDS = {
    "text": lambda arg: Text(),
    "int": lambda arg: Int(),
    "implied": lambda arg: Implied(int(arg or 0)),
    "overpunch": lambda arg: Overpunch(int(arg or 0)),
    "packed": lambda arg: Packed(int(arg or 0)),
    "date": lambda arg: Date(arg or "%Y%m%d"),
}


def spec_to_frame(spec: RecordSpec) -> pd.DataFrame:
    rows = []
    for f in spec.fields:
        kind = f.kind.describe()
        arg = ""
        if "scale=" in kind:
            arg = kind.split("scale=")[1].rstrip(")")
            kind = kind.split("(")[0]
        elif kind.startswith("date("):
            arg = kind[5:-1]
            kind = "date"
        rows.append({"name": f.name, "start": f.start, "length": f.length, "kind": kind, "arg": arg})
    return pd.DataFrame(rows)


def frame_to_spec(df: pd.DataFrame, index_base: int, encoding: str, length: Optional[int]) -> RecordSpec:
    fields = []
    for _, r in df.iterrows():
        if not str(r["name"]).strip():
            continue
        kind = KINDS[str(r["kind"])](str(r["arg"]).strip())
        fields.append(Field(str(r["name"]), int(r["start"]), int(r["length"]), kind))
    return RecordSpec(fields, index_base=index_base, encoding=encoding, length=length)


def display_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Decimals survive to here on purpose; stringify only for display."""
    return pd.DataFrame(
        [{k: ("" if v is None else str(v)) for k, v in row.items()} for row in rows]
    )


# --------------------------------------------------------------------------

st.title("Fixed-width parser")
st.caption(
    "A fixed-width file is defined in bytes. Reading it as characters is correct "
    "until the first byte above 0x7F, and silently wrong afterwards."
)

with st.sidebar:
    st.header("Input")
    source = st.radio(
        "File", ["Sample: customer master", "Sample: account balances (COMP-3)", "Upload"]
    )
    uploaded = st.file_uploader("Flat file", type=None) if source == "Upload" else None

    if source.startswith("Sample: customer"):
        data = build_customer_file()
        base_spec = CUSTOMER_SPEC
    elif source.startswith("Sample: account"):
        data = build_balance_file()
        base_spec = BALANCE_SPEC
    else:
        data = uploaded.read() if uploaded else b""
        base_spec = CUSTOMER_SPEC

    st.header("Conventions")
    index_base = st.radio(
        "Field offsets are",
        [1, 0],
        format_func=lambda b: "1-indexed (copybooks, data dictionaries)"
        if b == 1
        else "0-indexed (pandas colspecs)",
    )
    encoding = st.selectbox(
        "Encoding", ["utf-8", "latin-1", "cp037 (EBCDIC)", "ascii"], index=0
    ).split(" ")[0]
    framing = st.selectbox("Framing", ["auto", "lines", "block"], index=0)
    declare_length = st.checkbox("Declare record length", value=True)

st.subheader("Layout")
st.caption(
    "Edit the layout in place. `start` is expressed in the index base selected on the left - "
    "changing that radio moves every field by one byte without changing a single number here."
)
edited = st.data_editor(
    spec_to_frame(base_spec),
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "kind": st.column_config.SelectboxColumn(options=list(KINDS)),
        "arg": st.column_config.TextColumn(help="scale for implied/overpunch/packed, format for date"),
    },
)

if not len(data):
    st.info("Upload a file or pick a sample.")
    st.stop()

try:
    spec = frame_to_spec(
        edited, index_base, encoding, base_spec.record_length if declare_length else None
    )
except Exception as exc:  # noqa: BLE001 - surface any layout mistake to the user
    st.error(f"Layout error: {exc}")
    st.stop()

# ---- verdict first --------------------------------------------------------

report = audit(data, spec, framing)
if report.verdict.startswith("NOT SAFE"):
    st.error(report.verdict)
elif report.verdict.startswith("LOADS"):
    st.warning(report.verdict)
else:
    st.success(report.verdict)
for finding in report.findings:
    (st.warning if finding.startswith("WARNING") else st.info)(finding)

if report.verdict.startswith("NOT SAFE"):
    st.stop()

# ---- rows -----------------------------------------------------------------

result = parse(data, spec, framing)
c1, c2, c3, c4 = st.columns(4)
c1.metric("records", len(result.rows))
c2.metric("framing", result.framing)
c3.metric("field errors", len(result.errors))
c4.metric("record length", spec.record_length)

st.subheader("Parsed by byte offset")
st.dataframe(display_frame(result.rows), use_container_width=True)

if result.errors:
    with st.expander(f"{len(result.errors)} field error(s)"):
        st.dataframe(
            pd.DataFrame(result.errors, columns=["record", "field", "problem"]),
            use_container_width=True,
        )

# ---- the comparison -------------------------------------------------------

st.subheader("The same file, sliced by character")
st.caption(
    "This is the slice `pandas.read_fwf` takes. The comparison below is on the **raw text of "
    "each slice**, before any type coercion - otherwise the two readings would differ on every "
    "row simply because one returns Decimal and the other returns int, and the real signal "
    "would be buried. Highlighted rows are the ones where the two slicers grabbed different "
    "bytes."
)

records, _, _ = frame_records(data, spec, framing)
text_lines = data.decode(spec.encoding, errors="replace").splitlines()
byte_raw: List[Dict[str, str]] = []
char_raw: List[Dict[str, str]] = []
for i, rec in enumerate(records):
    line = text_lines[i] if i < len(text_lines) else ""
    byte_raw.append(
        {
            f.name: rec[slice(*spec.slice_of(f.name))].decode(spec.encoding, errors="replace").strip()
            for f in spec.fields
        }
    )
    char_raw.append({f.name: line[slice(*spec.slice_of(f.name))].strip() for f in spec.fields})

n = len(records)
disagree = [i for i in range(n) if byte_raw[i] != char_raw[i]]
if disagree:
    st.error(
        f"{len(disagree)} of {n} record(s) are sliced differently by the two readers: "
        f"rows {disagree}"
    )
else:
    st.success(
        f"Both slicers land on the same bytes for all {n} records - every byte in this file is "
        f"below 0x80, so character offsets happen to be byte offsets."
    )
st.dataframe(
    pd.DataFrame(char_raw).style.apply(
        lambda row: ["background-color: #f7dcdc" if row.name in disagree else "" for _ in row],
        axis=1,
    ),
    use_container_width=True,
)

# ---- export ---------------------------------------------------------------

csv = pd.DataFrame(result.rows).to_csv(index=False).encode("utf-8")
st.download_button("Download the byte-accurate parse as CSV", csv, "parsed.csv", "text/csv")

with st.expander("Layout as code"):
    st.code(spec.describe(), language="text")

st.caption(
    "Note on Decimal: money is parsed to decimal.Decimal, never float. A file that stores "
    "cents as integers is exact, and converting it to binary floating point on the way in "
    "throws that away for no reason."
)
