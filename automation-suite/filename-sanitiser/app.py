"""Streamlit front end: paste a list of filenames, pick a target, read the verdict.

The interface deliberately takes a *corpus*, not a name. A single-name box would
reproduce the bug the tool exists to report.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import sanitise as S

st.set_page_config(page_title="filename-sanitiser", page_icon="📁", layout="wide")

VERDICT_STYLE = {
    S.Verdict.PORTABLE: ("✅", "#2a7f7f", "every name is writable and no two collide"),
    S.Verdict.LOSSY: ("⚠️", "#d98324", "writable, but two sources land on one file"),
    S.Verdict.REJECTED: ("⛔", "#b3402f", "at least one name cannot be written at all"),
}
SEV_ICON = {S.Severity.CRITICAL: "🔴", S.Severity.WARNING: "🟠", S.Severity.INFO: "🔵"}

st.title("filename-sanitiser")
st.caption(
    "`sanitise(name) -> str` cannot report a collision, because a collision is a "
    "fact about a **pair** of names and the function only has one in scope. "
    "This takes the whole list."
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

left, right = st.columns([2, 1])

with right:
    profile_name = st.selectbox(
        "Target filesystem",
        list(S.PROFILES),
        index=0,
        help="The same name is legal on one of these and unwritable on another.",
    )
    profile = S.PROFILES[profile_name]
    st.caption(profile.note)

    default_dest = r"C:\data" if profile_name.startswith("windows") else "/data"
    dest = st.text_input("Destination directory", default_dest)

    sanitiser = st.selectbox("Sanitiser", list(S.SANITISERS), index=4)
    st.caption(S.SANITISERS[sanitiser].__doc__.strip().split("\n")[0])

    fold = st.selectbox(
        "Case-fold model",
        list(S.FOLDS),
        index=2,
        help="Only simple_upper models a real volume's 1:1 case table.",
    )

with left:
    text = st.text_area(
        "One filename per line",
        "\n".join(S.SAMPLE_NAMES[:24]),
        height=300,
        help="Trailing spaces are significant here, and that is the point.",
    )

names = [ln for ln in text.split("\n") if ln.strip() or ln]
if not names:
    st.info("Paste some filenames to audit.")
    st.stop()

report = S.audit(names, profile, dest, sanitiser, fold)
delivered, overwritten, rejected = report.partition()

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

icon, colour, blurb = VERDICT_STYLE[report.verdict]
st.markdown(
    f"<div style='border-left:6px solid {colour};padding:.6rem 1rem;"
    f"background:rgba(0,0,0,.03);margin:.5rem 0 1rem'>"
    f"<span style='font-size:1.5rem'>{icon}</span> "
    f"<strong style='font-size:1.2rem;color:{colour}'>{report.verdict.value}</strong>"
    f" &mdash; {blurb}</div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("source names", len(names))
c2.metric("delivered", len(delivered))
c3.metric("overwritten", len(overwritten), delta=None if not overwritten else "silent")
c4.metric("rejected", len(rejected))

st.caption(
    "`delivered + overwritten + rejected` sums to the source count exactly. "
    "**Overwritten is the dangerous column**: a rejected write fails and raises "
    "somewhere; an overwritten one succeeds and a file is simply gone."
)

if report.verdict is S.Verdict.PORTABLE:
    st.caption(
        "`portable` is a claim that the bytes survive the round trip. It is not "
        "a claim that a human can tell two files apart - read the findings."
    )

# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

st.subheader("Findings")
if not report.findings:
    st.success("No findings.")
for f in report.findings:
    with st.expander(
        f"{SEV_ICON[f.severity]}  **{f.code}** — {f.message}  ·  {len(f.names)} name(s)"
    ):
        st.code("\n".join(repr(n) for n in f.names), language="text")
        if f.detail:
            st.json(
                {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()})
                 for k, v in f.detail.items()},
                expanded=False,
            )

# ---------------------------------------------------------------------------
# Mapping and collision groups
# ---------------------------------------------------------------------------

st.subheader("Where each name lands")
bucket = {n: "delivered" for n in delivered}
bucket.update({n: "overwritten" for n in overwritten})
bucket.update({n: "rejected" for n in rejected})

rows = []
for src in names:
    out = report.mapping.get(src, "")
    group = report.written.get(
        S.collision_key(out, profile, fold), []
    ) if out else []
    rows.append(
        {
            "source": repr(src)[1:-1],
            "written as": repr(out)[1:-1] if out else "(nothing)",
            "outcome": bucket.get(src, "?"),
            "shares target with": len(group) - 1 if group else 0,
            "why": (
                S.collision_reason(src, next(g for g in group if g != src))
                if len(group) > 1
                else ""
            ),
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Cross-target and cross-sanitiser comparison
# ---------------------------------------------------------------------------

st.subheader("The same corpus against every sanitiser")
comp = pd.DataFrame(
    [
        {
            "sanitiser": r.sanitiser,
            "delivered": r.delivered,
            "overwritten": r.overwritten,
            "rejected": r.rejected,
            "distinct outputs": r.distinct_out,
            "verdict": r.verdict.value,
        }
        for r in S.compare(names, profile, dest, fold)
    ]
)
st.dataframe(comp, use_container_width=True, hide_index=True)
st.caption(
    "`distinct outputs` is the size of the codomain. It falls as the sanitiser "
    "rewrites more, and every name it loses is two sources merging."
)

st.subheader("The same corpus and sanitiser against every target")
tgt = []
for pname, p in S.PROFILES.items():
    d = r"C:\data" if pname.startswith("windows") else "/data"
    none_r = S.audit(names, p, d, "passthrough", fold)
    san_r = S.audit(names, p, d, sanitiser, fold)
    tgt.append(
        {
            "target": pname,
            "unit": p.component_unit,
            "case table": p.case_table or "byte-exact",
            "normalisation": p.normalisation or "byte-exact",
            "delivered, no sanitiser": none_r.delivered,
            f"delivered, {sanitiser}": san_r.delivered,
            "change": san_r.delivered - none_r.delivered,
        }
    )
st.dataframe(pd.DataFrame(tgt), use_container_width=True, hide_index=True)
st.caption(
    "A sanitiser written against one target's rules is applied unconditionally, "
    "at upload time, before the target is known. On a permissive byte-exact "
    "volume every rewrite is pure loss."
)

with st.expander("Round trip: build the archive on one volume, open it on another"):
    rt = []
    for a in S.PROFILES.values():
        for b in S.PROFILES.values():
            r = S.round_trip(names, a, b)
            rt.append({"archived on": a.name, "extracted on": b.name,
                       "entries": r["entries"], "files on disk": r["files_on_disk"],
                       "lost": r["lost"]})
    st.dataframe(pd.DataFrame(rt), use_container_width=True, hide_index=True)
    st.caption(
        "Nothing is sanitised here. The archive is valid and `unzip` reports no "
        "error; the file count on disk is just lower than the count in the "
        "archive. The direction is not symmetric - byte-exact is the finer "
        "partition, so going *to* it never loses anything."
    )
