from __future__ import annotations

# Streamlit UI for the Business Glossary Manager. Browse and search terms, open a
# term card (definition, owner, status, synonyms, related, linked assets), and
# read the governance-issues panel grouped by severity - each finding explains
# WHY it fired. Loads the built-in sample glossary. Offline, no API keys.

import pandas as pd
import streamlit as st

from glossary import Issue, Term, make_sample_glossary

st.set_page_config(page_title="Business Glossary Manager", page_icon="📖", layout="wide")

st.title("📖 Business Glossary Manager")
st.caption(
    "\"Active user\" means five different things to five teams, so every metric "
    "argument starts from zero. This is the single home for what terms MEAN - "
    "each with an owner, definition, status, synonyms, related links, and the "
    "data assets it governs - plus a validator that surfaces governance gaps."
)

# One shared sample glossary for the session.
if "glossary" not in st.session_state:
    st.session_state["glossary"] = make_sample_glossary()
g = st.session_state["glossary"]

issues = g.validate()
by_term: dict[str, list[Issue]] = {}
for i in issues:
    by_term.setdefault(i.term.lower(), []).append(i)

SEV_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}
STATUS_ICON = {"approved": "✅", "draft": "✏️", "deprecated": "⚠️"}

with st.sidebar:
    st.header("Find a term")
    query = st.text_input("Search name / synonym / definition", "")
    st.divider()
    st.markdown(
        "**Governance checks**\n\n"
        "- ownerless (high)\n"
        "- no definition (high)\n"
        "- synonym collision (high)\n"
        "- deprecated but still linked (high)\n"
        "- broken related ref (medium)\n"
        "- orphan - no links (low)"
    )
    st.divider()
    st.download_button(
        "⬇️ Export glossary (Markdown)",
        g.to_markdown().encode(),
        file_name="business_glossary.md",
        mime="text/markdown",
        use_container_width=True,
    )

results = g.search(query)

total = len(g.all_terms())
highs = sum(1 for i in issues if i.severity == "high")
flagged = len({i.term.lower() for i in issues})

c1, c2, c3 = st.columns(3)
c1.metric("Terms", f"{total}")
c2.metric("Governance issues", f"{len(issues)}")
c3.metric("High severity", f"{highs}")

left, right = st.columns([3, 2])

with left:
    st.subheader(f"Terms ({len(results)})")
    if not results:
        st.info("No terms match that search.")
    for t in results:
        flags = by_term.get(t.name.lower(), [])
        badge = STATUS_ICON.get(t.status, "")
        flag_note = f"  ·  {len(flags)} issue(s)" if flags else ""
        with st.expander(f"{badge} {t.name}  ({t.status}){flag_note}"):
            st.markdown(f"**Definition:** {t.definition or '_missing_'}")
            st.markdown(f"**Owner:** {t.owner or '_unassigned_'}")
            if t.synonyms:
                st.markdown(f"**Synonyms:** {', '.join(t.synonyms)}")
            if t.related:
                st.markdown(f"**Related terms:** {', '.join(t.related)}")
            if t.assets:
                st.markdown(f"**Linked assets:** `{'`, `'.join(t.assets)}`")
            for i in flags:
                st.warning(f"{SEV_ICON.get(i.severity,'')} **{i.issue_type}** - {i.message}")

with right:
    st.subheader("Governance issues")
    if not issues:
        st.success("Clean glossary - no issues found. ✅")
    else:
        rows = [
            {"severity": i.severity, "type": i.issue_type, "term": i.term,
             "message": i.message}
            for i in issues
        ]
        df = pd.DataFrame(rows)
        pick = st.multiselect(
            "Filter by severity", ["high", "medium", "low"],
            default=["high", "medium", "low"],
        )
        view = df[df["severity"].isin(pick)] if pick else df
        st.dataframe(view, use_container_width=True, hide_index=True)

st.caption(
    "A glossary is only as good as its owners. The validator flags process gaps "
    "(ownerless terms, colliding synonyms, retired-but-live terms) - it cannot "
    "decide what a metric should mean. That is a stewardship conversation."
)
