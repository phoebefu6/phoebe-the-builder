from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from story_gen import backlog_to_markdown, generate_backlog

st.set_page_config(page_title="User Story Generator", page_icon="📝", layout="wide")
st.title("📝 User Story Generator")
st.caption("Paste raw feature ideas → get INVEST-scored user stories with Given/When/Then acceptance criteria.")

SAMPLE = """Users should be able to reset their password by email
Search projects by tag and owner
As a team lead, I want to export the sprint report to PDF, so that I can share it with stakeholders
Make the app easy and fast for everyone
Upload a CSV of customers and validate it
Notify me when a deployment fails"""

with st.sidebar:
    st.header("Input")
    persona = st.text_input("Default persona (optional)", placeholder="e.g. project manager")
    use_sample = st.checkbox("Use sample ideas", value=True)
    st.markdown("---")
    st.markdown(
        "**Tip:** one idea per line. Ideas already in "
        "'As a X, I want Y, so that Z' form are parsed as-is."
    )
    if os.environ.get("ANTHROPIC_API_KEY"):
        st.success("Claude key detected — polish available")

ideas = st.text_area("Feature ideas (one per line)", value=SAMPLE if use_sample else "", height=170)

if st.button("Generate stories", type="primary") and ideas.strip():
    stories = generate_backlog(ideas, persona)
    st.session_state["stories"] = stories

stories = st.session_state.get("stories", [])
if stories:
    df = pd.DataFrame(
        {
            "Story": [s.capability[:60] for s in stories],
            "Type": [s.feature_type for s in stories],
            "INVEST": [s.invest_score for s in stories],
            "Flags": [len(s.invest_flags) for s in stories],
        }
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Backlog quality")
        st.dataframe(df, use_container_width=True, hide_index=True)
    with col2:
        fig, ax = plt.subplots(figsize=(4, 3))
        colors = ["#2a9d8f" if s >= 80 else "#e9c46a" if s >= 60 else "#e76f51" for s in df["INVEST"]]
        ax.barh(range(len(df)), df["INVEST"], color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([f"Story {i + 1}" for i in range(len(df))], fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("INVEST score")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    for i, s in enumerate(stories, 1):
        badge = "🟢" if s.invest_score >= 80 else "🟡" if s.invest_score >= 60 else "🔴"
        with st.expander(f"{badge} Story {i}: {s.capability.capitalize()}  ·  {s.invest_score}/100", expanded=i == 1):
            st.markdown(f"**{s.story}**")
            st.markdown(f"*Feature type: `{s.feature_type}` · original idea: “{s.raw}”*")
            st.markdown("**Acceptance criteria:**")
            for ac in s.acceptance_criteria:
                st.markdown(f"- {ac}")
            if s.invest_flags:
                st.markdown("**Improve this story:**")
                for flag in s.invest_flags:
                    st.warning(flag)

    st.download_button(
        "⬇️ Download backlog as Markdown",
        backlog_to_markdown(stories),
        file_name="user_stories.md",
        mime="text/markdown",
    )
else:
    st.info("Enter feature ideas and click **Generate stories**.")
