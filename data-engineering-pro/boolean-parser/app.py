"""Paste a string, watch sixteen readers disagree about it.

    streamlit run app.py
"""

from __future__ import annotations

import collections
from typing import List

import boolparse as B
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Boolean Parser Audit", layout="wide")

BADGE = {
    B.TRUE: "🔵 true",
    B.FALSE: "🟠 false",
    B.REFUSED: "🟢 refused",
    B.NOTBOOL: "⚪ not a boolean",
}

st.title("A string does not contain a boolean")
st.caption(
    "A reader assigns one — and a reader is an accept table, a normalisation and a "
    "failure policy. The word `true` in a config file carries none of them."
)


@st.cache_data(show_spinner=False)
def audit(texts: List[str]) -> pd.DataFrame:
    """Run every reader over an arbitrary list of strings."""
    samples = tuple(B.Sample(t, "custom", None) for t in texts)
    rows = {r.name: [x.verdict for x in r.fn(samples)] for r in B.READERS}
    return pd.DataFrame(rows, index=[B.show(t) for t in texts])


@st.cache_data(show_spinner=False)
def corpus_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {name: [r.verdict for r in readings] for name, readings in B.grid().items()},
        index=[s.label for s in B.CORPUS],
    )


tab_one, tab_all, tab_readers = st.tabs(
    ["One string", f"The whole corpus ({len(B.CORPUS)})", "The readers"]
)

# --------------------------------------------------------------------------
with tab_one:
    raw = st.text_input(
        "The value, exactly as it arrives",
        value="false",
        help="Trailing spaces and a stray \\r are the point — paste the real thing.",
    )
    keep = st.checkbox("Show what each reader actually returned", value=False)

    if raw is not None:
        sample = B.Sample(raw, "custom", None)
        readings = {r.name: r.fn((sample,))[0] for r in B.READERS}
        counts = collections.Counter(x.verdict for x in readings.values())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("read as true", counts[B.TRUE])
        c2.metric("read as false", counts[B.FALSE])
        c3.metric("refused", counts[B.REFUSED])
        c4.metric("not a boolean", counts[B.NOTBOOL])

        if counts[B.TRUE] and counts[B.FALSE]:
            st.error(
                f"`{B.show(raw)}` **flips sign**: {counts[B.TRUE]} readers call it true "
                f"and {counts[B.FALSE]} call it false, all of them confidently. Nothing "
                f"in a stack containing both can detect the disagreement."
            )
        elif counts[B.TRUE] or counts[B.FALSE]:
            st.success(f"`{B.show(raw)}` never flips sign across these sixteen readers.")

        table = pd.DataFrame(
            {
                "verdict": [BADGE[readings[r.name].verdict] for r in B.READERS],
                "returned": [readings[r.name].raw for r in B.READERS],
                "source": [r.source for r in B.READERS],
                "where you meet it": [r.seen_in for r in B.READERS],
            },
            index=[r.name for r in B.READERS],
        )
        if not keep:
            table = table.drop(columns=["returned"])
        st.dataframe(table, use_container_width=True)

        st.subheader("The three decisions, for this string")
        norm = B.normalisations(raw)
        acc = B.accepted_after(raw)
        st.dataframe(
            pd.DataFrame(
                {
                    "normalised to": [B.show(v) for v in norm.values()],
                    "in a true/false/yes/no/on/off/1/0 table?": list(acc.values()),
                },
                index=list(norm.keys()),
            ),
            use_container_width=True,
        )
        if acc["strip"] and not acc["as-written"]:
            st.warning(
                "This string is only accepted **after stripping**. Whatever wrote the "
                "file — a CRLF line ending, a space after the `=`, a BOM — is now part "
                "of the value."
            )
        if acc["casefold"] and not acc["lower"]:
            st.warning(
                "`casefold()` accepts this and `lower()` does not. They are different "
                "functions, not a strictness ordering."
            )

# --------------------------------------------------------------------------
with tab_all:
    st.markdown(
        f"**{len(B.CORPUS)} strings × {len(B.READERS)} readers = "
        f"{len(B.CORPUS) * len(B.READERS)} readings.** "
        f"`{len(B.unanimous())}` strings are read the same way by all sixteen."
    )
    frame = corpus_frame()

    def paint(v: str) -> str:
        return {
            B.TRUE: "background-color:#2f6f8f;color:white",
            B.FALSE: "background-color:#e0a458;color:#1d1a17",
            B.REFUSED: "background-color:#4f7942;color:white",
            B.NOTBOOL: "background-color:#cfc7bd;color:#1d1a17",
        }[v]

    st.dataframe(frame.style.map(paint), use_container_width=True, height=780)
    st.caption(
        "Blue = true, amber = false, green = refused, grey = returned something that is "
        "not a boolean. Green is the only column that tells you anything went wrong."
    )

# --------------------------------------------------------------------------
with tab_readers:
    refuse = B.refusal_counts()
    defer = B.notbool_counts()
    wrong = collections.Counter(name for _, name, _ in B.silently_wrong())
    st.dataframe(
        pd.DataFrame(
            {
                "stack": [r.stack for r in B.READERS],
                "source": [r.source for r in B.READERS],
                "refused": [refuse[r.name] for r in B.READERS],
                "deferred": [defer[r.name] for r in B.READERS],
                "silently wrong": [wrong[r.name] for r in B.READERS],
                "where you meet it": [r.seen_in for r in B.READERS],
            },
            index=[r.name for r in B.READERS],
        ),
        use_container_width=True,
    )
    st.markdown(
        f"""
**{len(B.never_refuse())} of {len(B.READERS)} readers never refuse anything.** They return a
confident boolean for every one of the {len(B.CORPUS)} strings, including `undefined`, a bare
UTF-8 BOM and the empty string. A reader that cannot fail cannot tell you that you spelled it
wrong — it can only be quietly wrong instead.

The two right-hand columns are mutually exclusive. No reader in this roster is both permissive
and safe.

The fix is not a better parser: **do not store a boolean as text.** A `BOOLEAN` column, or an
`INTEGER` holding 0 and 1, is never parsed by anybody.
"""
    )
