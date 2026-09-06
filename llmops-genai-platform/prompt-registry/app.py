from __future__ import annotations

"""Streamlit UI for the Prompt Registry & Versioning tool."""

import os
from typing import Dict

import streamlit as st
from registry import PromptRegistry, extract_variables

REGISTRY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_registry")

STAGE_COLORS = {
    "production": "#1a7f37",  # green
    "staging": "#bf8700",     # amber
    "draft": "#8b949e",       # grey
}

SEED = {
    "support_triage": [
        "You are a support agent. Answer the customer question:\n{question}",
        (
            "You are a helpful support agent for {product}.\n"
            "Answer the customer question clearly and concisely:\n{question}"
        ),
        (
            "You are a helpful support agent for {product}.\n"
            "Rules: be concise, never invent policy, escalate billing issues.\n"
            "Customer question:\n{question}\n\nAnswer:"
        ),
    ],
    "summarize_ticket": [
        "Summarize this ticket in one sentence:\n{ticket}",
        "Summarize this ticket in one sentence and label priority (low/med/high):\n{ticket}",
    ],
}


def seed_registry(reg: PromptRegistry) -> None:
    """Populate a fresh registry with a couple of versioned prompts."""
    for name, bodies in SEED.items():
        for i, body in enumerate(bodies, start=1):
            reg.commit(name, body, tags=["seed"], author="phoebe",
                       created_at=f"2026-07-18T09:0{i}:00")
    # Put one clear production per prompt.
    reg.promote("support_triage", 3, "production")
    reg.promote("summarize_ticket", 2, "production")


@st.cache_resource
def get_registry() -> PromptRegistry:
    reg = PromptRegistry(REGISTRY_DIR)
    if not reg.names():
        seed_registry(reg)
    return reg


def stage_badge(stage: str) -> str:
    color = STAGE_COLORS.get(stage, "#8b949e")
    return (
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:10px;font-size:0.8em'>{stage}</span>"
    )


def main() -> None:
    st.set_page_config(page_title="Prompt Registry", page_icon="📝", layout="wide")
    st.title("📝 Prompt Registry & Versioning")
    st.caption(
        "Stop scattering prompts across the codebase. Version them, diff them, "
        "promote one to production, roll back instantly."
    )

    reg = get_registry()
    names = reg.names()

    tab_browse, tab_commit, tab_diff, tab_render = st.tabs(
        ["📚 Browse", "➕ Commit", "🔀 Diff", "▶️ Render"]
    )

    # ---- Browse --------------------------------------------------------
    with tab_browse:
        st.subheader("All prompt versions")
        frame = reg.to_frame()
        st.dataframe(frame, use_container_width=True, hide_index=True)

        st.markdown("**Live (production) prompts:**")
        for name in names:
            prod = reg.production(name)
            if prod:
                st.markdown(
                    f"- `{name}` &rarr; v{prod['version']} "
                    f"{stage_badge('production')}",
                    unsafe_allow_html=True,
                )

    # ---- Commit --------------------------------------------------------
    with tab_commit:
        st.subheader("Commit a new prompt version")
        name = st.text_input("Prompt name", value=names[0] if names else "my_prompt")
        body = st.text_area(
            "Prompt body (use {placeholders} for variables)",
            height=200,
            value="You are a helpful assistant.\nAnswer: {question}",
        )
        found = extract_variables(body)
        st.caption(f"Detected variables: {found or 'none'}")
        if st.button("Commit version", type="primary"):
            rec = reg.commit(name, body, tags=["ui"], author="phoebe")
            st.success(
                f"Committed `{name}` v{rec['version']} (hash {rec['hash']}). "
                f"Variables: {rec['variables'] or 'none'}"
            )
            st.rerun()

    # ---- Diff ----------------------------------------------------------
    with tab_diff:
        st.subheader("Diff two versions")
        if not names:
            st.info("No prompts yet - commit one first.")
        else:
            name = st.selectbox("Prompt", names, key="diff_name")
            versions = sorted(r["version"] for r in reg.list_prompts(name))
            if len(versions) < 2:
                st.info("Need at least two versions to diff.")
            else:
                c1, c2 = st.columns(2)
                v1 = c1.selectbox("From version", versions, index=0)
                v2 = c2.selectbox("To version", versions, index=len(versions) - 1)
                diff_text = reg.diff(name, v1, v2)
                if diff_text.strip():
                    st.code(diff_text, language="diff")
                else:
                    st.info("These versions are identical.")

    # ---- Render --------------------------------------------------------
    with tab_render:
        st.subheader("Render the production prompt")
        if not names:
            st.info("No prompts yet.")
        else:
            name = st.selectbox("Prompt", names, key="render_name")
            prod = reg.production(name)
            record = prod or reg.get(name)
            st.markdown(
                f"Using v{record['version']} {stage_badge(record['stage'])}",
                unsafe_allow_html=True,
            )
            required = record["variables"]
            values: Dict = {}
            for var in required:
                values[var] = st.text_input(f"{{{var}}}", key=f"var_{name}_{var}")
            if st.button("Render", type="primary"):
                try:
                    out = reg.render(name, values, version=record["version"])
                    st.text_area("Rendered prompt", value=out, height=220)
                except KeyError as exc:
                    st.error(str(exc))


if __name__ == "__main__":
    main()
