"""Streamlit front end: paste a list of titles, see the URLs you actually get.

Run: streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import slug as S
import streamlit as st

st.set_page_config(page_title="Slug collider", layout="wide")

VERDICT_STYLE = {
    S.Verdict.INJECTIVE: ("success", "injective"),
    S.Verdict.DEDUPED: ("warning", "deduped"),
    S.Verdict.LOSSY: ("error", "lossy"),
}

st.title("Slug collider")
st.caption(
    "A slugifier returns a string. It cannot return the fact that two titles "
    "just landed on the same one, that a third landed on nothing, or that the "
    "URL depends on the order you imported the rows."
)

with st.sidebar:
    st.header("Settings")
    profile_name = st.selectbox(
        "Slug algorithm", list(S.PROFILES), index=0,
        format_func=lambda n: f"{n}  -  {S.PROFILES[n].origin}",
    )
    use_cap = st.checkbox("Apply a length cap (slug column width)", value=False)
    cap = st.slider("cap (characters)", 8, 255, 50) if use_cap else None
    check_routes = st.checkbox("Check against reserved route segments", value=True)
    st.divider()
    st.caption(S.PROFILES[profile_name].fn.__doc__.strip().split("\n\n")[0])

left, right = st.columns([1, 1])

with left:
    st.subheader("Titles")
    raw = st.text_area(
        "one per line",
        value="\n".join(S.CORPUS),
        height=320,
        label_visibility="collapsed",
    )

titles = [t for t in (line.rstrip() for line in raw.splitlines()) if t]

if not titles:
    st.info("Paste some titles to audit.")
    st.stop()

report = S.audit(
    titles, profile_name, cap=cap, reserved=S.RESERVED if check_routes else set()
)

with right:
    st.subheader("Verdict")
    kind, label = VERDICT_STYLE[report.verdict]
    getattr(st, kind)(f"**{label}** - {report.reason}")

    distinct = len({s for s in report.slugs.values() if s})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("titles", len(titles))
    c2.metric("distinct URLs", distinct, delta=distinct - len(titles))
    c3.metric("findings", len(report.findings))
    c4.metric(
        "critical",
        sum(1 for f in report.findings if f.severity is S.Severity.CRITICAL),
    )

    counts = report.counts()
    if counts:
        st.bar_chart(pd.Series(counts).sort_values(ascending=False), height=180)

st.divider()

tab_slugs, tab_findings, tab_profiles, tab_order = st.tabs(
    ["Slugs", "Findings", "Compare algorithms", "Import order"]
)

with tab_slugs:
    by_slug = {}
    for t in titles:
        by_slug.setdefault(report.slugs[t], []).append(t)

    rows = []
    for t in titles:
        s = report.slugs[t]
        if not s:
            status = "empty"
        elif len(by_slug[s]) > 1:
            status = f"shared with {len(by_slug[s]) - 1} other"
        elif s in (S.RESERVED if check_routes else set()):
            status = "shadows a route"
        else:
            status = "ok"
        rows.append(
            {
                "title": t,
                "slug": s or "(empty)",
                "len": len(s),
                "status": status,
                "non-ASCII in title": sum(1 for ch in t if ord(ch) > 127),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption(
        "`assign()` resolves the shared ones the way a CMS does - by suffixing, "
        "in insertion order:"
    )
    assigned = S.assign(titles, profile_name, cap=cap)
    st.dataframe(
        pd.DataFrame(
            [{"title": t, "final URL": "/" + assigned[t]} for t in titles]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab_findings:
    if not report.findings:
        st.success("No findings. Every title has a distinct, usable URL.")
    for sev in (S.Severity.CRITICAL, S.Severity.HIGH, S.Severity.MEDIUM):
        block = [f for f in report.findings if f.severity is sev]
        if not block:
            continue
        st.markdown(f"#### {sev.value} ({len(block)})")
        for f in block:
            with st.expander(f"`{f.kind.value}`  -  {f.detail}", expanded=False):
                for t in f.titles:
                    codepoints = " ".join(
                        f"U+{ord(ch):04X}" for ch in t if ord(ch) > 127
                    )
                    st.write(f"- `{t}`" + (f"   ({codepoints})" if codepoints else ""))

with tab_profiles:
    st.caption(
        "Changing slugifier is a URL migration. This is its size, before you run it."
    )
    grid = S.compare(titles)
    df = pd.DataFrame(grid).T
    df.index.name = "title"
    st.dataframe(df, use_container_width=True)

    st.markdown("#### Titles whose URL changes between two algorithms")
    a, b = st.columns(2)
    pa = a.selectbox("from", list(S.PROFILES), index=0, key="from")
    pb = b.selectbox("to", list(S.PROFILES), index=1, key="to")
    diff = S.disagreements(titles, pa, pb)
    st.write(f"**{len(diff)} of {len(titles)}** titles get a different URL.")
    if diff:
        st.dataframe(
            pd.DataFrame(diff, columns=["title", pa, pb]),
            use_container_width=True,
            hide_index=True,
        )

with tab_order:
    st.caption(
        "Suffix de-duplication makes the URL a function of insertion order. "
        "These are four orders any real import could produce."
    )
    orders = {
        "as pasted": titles,
        "reversed": list(reversed(titles)),
        "A-Z": sorted(titles),
        "Z-A": sorted(titles, reverse=True),
    }
    n, unstable = S.order_sensitivity(titles, list(orders.values()), profile_name, cap)
    st.metric("titles that received more than one URL", f"{n} of {len(titles)}")
    if unstable:
        st.dataframe(
            pd.DataFrame(
                [{"title": t, "URLs seen": ", ".join("/" + u for u in urls)}
                 for t, urls in unstable]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Re-importing from a backup that iterates in a different order does "
            "not preserve these. Nothing errors; the old URLs 404."
        )
    else:
        st.success("No collisions, so no order dependence.")

st.divider()
st.caption(
    "Reproduce every number: `python3 evidence.py`  -  tests: "
    "`python3 -m pytest test_slug.py -q`"
)
