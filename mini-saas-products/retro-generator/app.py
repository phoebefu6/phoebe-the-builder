from __future__ import annotations

import streamlit as st
from retro_generator import SprintData, generate_retro

st.set_page_config(page_title="Sprint Retrospective Generator", page_icon="🔄")
st.title("🔄 Sprint Retrospective Generator")
st.caption(
    "Retros are unstructured and repetitive. Feed in the sprint numbers and get a structured "
    "retro — observations and action items — in your team's format."
)

with st.sidebar:
    st.subheader("Sprint metrics")
    name = st.text_input("Sprint name", "Sprint 24")
    planned = st.number_input("Planned points", 1, 200, 40)
    completed = st.number_input("Completed points", 0, 200, 31)
    prev = st.number_input("Previous sprint completed", 0, 200, 35)
    carryover = st.number_input("Carryover items", 0, 50, 4)
    bugs = st.number_input("Bugs found", 0, 100, 6)
    incidents = st.number_input("Production incidents", 0, 50, 1)
    mood = st.slider("Team mood (1-5)", 1.0, 5.0, 3.2, 0.1)
    fmt = st.selectbox("Format", ["start-stop-continue", "went-well-improve", "4Ls"])
    use_claude = st.checkbox("Claude narrative (needs ANTHROPIC_API_KEY)", value=False)

notes_raw = st.text_area("Team notes (one per line)",
                         "New CI pipeline saved review time.\nOnboarding of two contractors slowed pairing.")

if st.button("Generate retro", type="primary"):
    data = SprintData(
        sprint_name=name, planned_points=int(planned), completed_points=int(completed),
        carryover_items=int(carryover), bugs_found=int(bugs), incidents=int(incidents),
        team_mood=float(mood), prev_completed_points=int(prev),
        notes=[n for n in notes_raw.splitlines() if n.strip()],
    )
    retro = generate_retro(data, fmt=fmt, use_claude=use_claude)

    c1, c2, c3 = st.columns(3)
    c1.metric("Completion", f"{data.completion_rate:.0%}")
    c2.metric("Velocity Δ", f"{data.velocity_delta:+d}")
    c3.metric("Action items", len(retro.action_items))

    md = retro.to_markdown()
    st.download_button("Download retro.md", md, file_name="retro.md", mime="text/markdown")
    st.markdown("---")
    st.markdown(md)
