"""What this catalog covers for a team.

    streamlit run app.py
"""

from __future__ import annotations

import capability as C
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Capability Map", layout="wide")

st.title("What this covers for a team")
st.caption(
    "Describe the symptom, or pick the role or the situation. Sourced from the "
    f"generated catalog - one reader, not a second parser. [repo]({C.REPO_URL})"
)

try:
    TOOLS = C.load()
except C.CatalogMissing as exc:
    st.error(str(exc))
    st.stop()

GROUPED = C.by_task(TOOLS)
TITLES = C.task_titles(TOOLS)


def table(tools, height=None):
    """One consistent table. The problem column is the one people read."""
    if not tools:
        st.info("Nothing here yet.")
        return
    st.dataframe(
        pd.DataFrame({
            "Tool": [t.name for t in tools],
            "The problem it was built for": [t.problem for t in tools],
            "Job": [t.task_title for t in tools],
            "Code": [t.repo_url for t in tools],
            "Notebook": [t.colab_url for t in tools],
        }),
        use_container_width=True, hide_index=True, height=height,
        column_config={
            "Code": st.column_config.LinkColumn("Code", display_text="open"),
            "Notebook": st.column_config.LinkColumn("Notebook", display_text="run"),
        },
    )


# --------------------------------------------------------------------------
query = st.text_input(
    "Describe what is going wrong",
    placeholder="stale dashboard · the CSV won't parse · two numbers disagree · "
                "the model looked fine offline",
    help="Searches the problem each tool was built for, not just its name.",
)

if query.strip():
    hits = C.search(query, TOOLS)
    if not hits:
        st.warning(
            "Nothing matches. Matching is plain substring over the problem text, the "
            "name, the slug and the job - no stemming - so a miss is a real gap in what "
            "the catalog *says* rather than a scoring artefact. `CRLF` is a case in "
            "point: the tool exists, and its own description never uses the word. Try "
            "the plainer phrase (`line ending`), or browse the tabs below."
        )
    else:
        jobs = sorted({t.task_title for t in hits})
        st.success(
            f"Found something in {len(jobs)} "
            f"{'job' if len(jobs) == 1 else 'different jobs'}: {', '.join(jobs)}."
        )
        table(sorted(hits, key=lambda t: (t.task_title, t.name.lower())))
    st.divider()

tab_task, tab_role, tab_scenario = st.tabs(
    ["By the job you arrived with", "By role", "By situation"]
)

# --------------------------------------------------------------------------
with tab_task:
    st.markdown(
        "Thirteen jobs. Pick the one you are actually trying to do - the technology a "
        "tool happens to use is rarely how anyone looks for it."
    )
    chosen = st.selectbox(
        "Job", [t for t in C.TASK_ORDER if GROUPED.get(t)],
        format_func=lambda t: TITLES.get(t, t),
    )
    st.subheader(TITLES.get(chosen, chosen))
    table(GROUPED[chosen], height=420)

    st.divider()
    st.markdown("**How deep each job goes**")
    depth = pd.DataFrame(
        {"tools": [len(GROUPED[t]) for t in C.TASK_ORDER]},
        index=[TITLES.get(t, t) for t in C.TASK_ORDER],
    )
    st.bar_chart(depth, horizontal=True, height=380)
    st.caption(
        "Depth, not progress. A short bar is a job with a few sharp tools, not an "
        "unfinished one."
    )

# --------------------------------------------------------------------------
with tab_role:
    st.markdown("Which of the thirteen jobs a role spends its week inside.")
    for name, blurb, tasks in C.ROLES:
        tools = C.for_role(name, TOOLS)
        with st.expander(f"**{name}** — {blurb}", expanded=(name == "Data engineer")):
            st.caption("Jobs: " + " · ".join(TITLES.get(t, t) for t in tasks))
            table(tools, height=330)

    st.divider()
    st.markdown("**Where the handoffs are**")
    st.caption(
        "Tools two roles both reach for. A high number is a shared surface - the place "
        "an argument about ownership actually happens."
    )
    overlap = C.role_overlap(TOOLS)
    names = [r[0] for r in C.ROLES]
    grid = pd.DataFrame(
        [[("" if a == b else overlap.get((a, b), overlap.get((b, a), 0)))
          for b in names] for a in names],
        index=names, columns=names,
    )
    st.dataframe(grid, use_container_width=True)

# --------------------------------------------------------------------------
with tab_scenario:
    st.markdown(
        "A real afternoon crosses three or four jobs. This is the order they come in."
    )
    picked = st.radio(
        "Situation", [s[0] for s in C.SCENARIOS],
        format_func=lambda n: n, horizontal=False,
    )
    blurb = next(s[1] for s in C.SCENARIOS if s[0] == picked)
    st.markdown(f"*{blurb}*")

    steps = C.for_scenario(picked, TOOLS)
    for i, (task, tools) in enumerate(steps, 1):
        st.subheader(f"{i}. {TITLES.get(task, task)}")
        table(tools[:8], height=None if len(tools[:8]) < 5 else 240)
        if len(tools) > 8:
            st.caption(
                f"{len(tools) - 8} more under this job - see the *By the job* tab. "
                "Truncated here so the sequence stays readable."
            )
