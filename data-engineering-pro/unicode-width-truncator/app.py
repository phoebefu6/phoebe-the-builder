"""Paste a string, pick an n, see every truncation it could receive."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import uwidth as U

st.set_page_config(page_title="Truncate to n", layout="wide")

st.title("Truncate to n")
st.caption(
    "\"Truncate to 20\" does not name an operation. A truncator is a unit, a boundary "
    "rule and a policy for the removed piece - and the integer carries none of them."
)


def show(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if ch == U.ZWJ:
            out.append("␍ZWJ␎")
        elif ch in (U.VS15, U.VS16):
            out.append("␍VS␎")
        elif 0xD800 <= cp <= 0xDFFF:
            out.append(f"␍D{cp:04X}␎")
        elif ch == "​":
            out.append("␍ZWSP␎")
        elif ch in U.BIDI_OPEN or ch in U.BIDI_CLOSE:
            out.append(f"␍U+{cp:04X}␎")
        elif cp < 0x20 or cp == 0x7F:
            out.append(f"␍{cp:02X}␎")
        else:
            out.append(ch)
    return "".join(out)


tab_one, tab_sinks, tab_corpus, tab_why = st.tabs(
    ["One string", "Will it fit?", "The corpus", "Why ten answers"]
)

with tab_one:
    preset = st.selectbox(
        "start from a corpus case, or type your own below",
        ["(type my own)"] + [c.name for c in U.CORPUS],
        index=6,
    )
    default_text = "" if preset == "(type my own)" else U.CASE_BY_NAME[preset].text
    default_n = 20 if preset == "(type my own)" else U.CASE_BY_NAME[preset].n
    col_a, col_b = st.columns([4, 1])
    text = col_a.text_input("string", value=default_text, key=f"text-{preset}")
    n = col_b.number_input("n", min_value=1, max_value=200, value=default_n, step=1)

    if text:
        case = U.Case("live", text, int(n), "pasted")
        spread = U.unit_spread(case)
        st.subheader("The same string is this long")
        st.dataframe(
            pd.DataFrame([spread]).T.rename(columns={0: "length"}),
            width="content",
        )

        cuts = U.cut_all(case)
        rows = []
        for name, cut in cuts.items():
            notes = []
            if cut.lone_surrogate:
                notes.append("LONE SURROGATE - no UTF-8 encoding")
            if U.has_replacement(cut.text):
                notes.append("U+FFFD")
            if cut.dangling:
                notes.append(f"ends in {cut.dangling}")
            change = U.identity_change(text, cut.text)
            if change:
                notes.append(change)
            if cut.overflows_own_limit:
                notes.append(f"OVER its own byte limit ({cut.bytes_out} > {n})")
            if cut.bidi_leak:
                notes.append("unbalanced bidi scope")
            rows.append({
                "truncator": name,
                "unit": U.TRUNCATOR_BY_NAME[name].unit,
                "output": show(cut.text),
                "bytes": cut.bytes_out,
                "cp": cut.code_points,
                "graphemes": cut.grapheme_count,
                "columns": cut.columns_out,
                "finding": "; ".join(notes),
            })
        st.subheader(f"{len({c.text for c in cuts.values()})} distinct strings come out")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        v = U.verdict_for(case, cuts)
        st.info(f"**{v.verdict}** - {v.detail}")

        st.subheader("What to do instead")
        st.caption(
            "Pick the truncator by what the thing you are protecting counts. "
            "`safe_truncate` measures in the sink's unit, keeps the ellipsis inside "
            "the budget, never splits a cluster, and drops trailing clusters until "
            "nothing dangles."
        )
        safe_rows = [{
            "sink": s.name,
            "counts": s.unit,
            "safe_truncate output": show(U.safe_truncate(text, int(n), s.name)),
            "fits": U.fits(s, U.safe_truncate(text, int(n), s.name), int(n)),
        } for s in U.SINKS]
        st.dataframe(pd.DataFrame(safe_rows), width="stretch", hide_index=True)

with tab_sinks:
    st.subheader("Truncating to n does not make it fit a limit of n")
    failed, total = U.sink_failure_rate()
    st.metric("runs where the truncated value is still over the limit", f"{failed} / {total}")
    t_names = [t.name for t in U.TRUNCATORS]
    grid = []
    for t in t_names:
        row = {"truncator": t, "unit": U.TRUNCATOR_BY_NAME[t].unit}
        for s in U.SINKS:
            bad = sum(
                1 for case in U.CORPUS
                if not U.fits(s, U.cut_all(case)[t].text, case.n)
            )
            row[s.name] = f"{bad}/{len(U.CORPUS)}"
        grid.append(row)
    st.dataframe(pd.DataFrame(grid), width="stretch", hide_index=True)
    st.caption(
        "Each cell: how many of the 26 corpus strings are still over the limit after "
        "being truncated to that same number. A truncator is correct relative to one "
        "unit and unrelated to every other."
    )
    st.dataframe(
        pd.DataFrame([{"sink": s.name, "counts": s.unit, "note": s.note} for s in U.SINKS]),
        width="stretch", hide_index=True,
    )

with tab_corpus:
    rows = []
    for case in U.CORPUS:
        v = U.verdict_for(case)
        rows.append({
            "case": case.name,
            "n": case.n,
            "source": case.source,
            "distinct outputs": v.distinct_outputs,
            "verdict": v.verdict,
            "flags": ", ".join(v.flags),
            "detail": v.detail,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.subheader("Census")
    c1, c2 = st.columns(2)
    c1.write("**verdict** (worst thing per case)")
    c1.dataframe(pd.DataFrame([U.verdict_census()]).T.rename(columns={0: "cases"}))
    c2.write("**flags** (every thing per case)")
    c2.dataframe(pd.DataFrame([U.flag_census()]).T.rename(columns={0: "cases"}))

with tab_why:
    st.dataframe(
        pd.DataFrame([
            {"truncator": t.name, "unit": t.unit, "where you meet it": t.seen_in}
            for t in U.TRUNCATORS
        ]),
        width="stretch", hide_index=True,
    )
    v = U.node_versions()
    st.caption(
        f"UTF-16 and ICU truncators run in a real node subprocess (ICU {v['icu']}, "
        f"Unicode {v['unicode']}). Grapheme clusters on the Python side come from "
        "regex's UAX #29 `\\X`."
    )
    st.subheader("Two UAX #29 implementations, two answers")
    dis = U.segmenter_disagreements()
    st.dataframe(
        pd.DataFrame([
            {"case": name, "text": U.CASE_BY_NAME[name].text, "regex": py, "ICU": js}
            for name, py, js in dis
        ]),
        width="stretch", hide_index=True,
    )
    st.caption(
        "Both are correct for the UCD they ship. An API in Node and a worker in "
        "Python disagree about how many characters a Hindi name has, and a limit of "
        "6 'characters' is two different limits inside one service."
    )
