from __future__ import annotations

import streamlit as st
from screener import SAMPLE_JOB, SAMPLE_RESUMES, extract_required_skills, screen_resume

BAND_ICON = {"advance": "🟢", "maybe": "🟠", "reject": "🔴"}

st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("Resume Screener")
st.caption("Rank a stack of resumes by skills fit — a triage aid, not a hiring decision.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Anthropic API key (optional)", type="password")
    if not api_key:
        st.info("No key set — using deterministic skill-overlap scoring.")

job_desc = st.text_area("Job description", value=SAMPLE_JOB, height=220)

if job_desc.strip():
    st.caption("Detected required skills: " + ", ".join(extract_required_skills(job_desc)) or "—")

default_resumes = "\n\n===\n\n".join(SAMPLE_RESUMES.values())
resumes_raw = st.text_area(
    "Resumes (separate each with a line containing only ===)",
    value=default_resumes,
    height=260,
)

if st.button("Screen resumes", type="primary"):
    resumes = [r.strip() for r in resumes_raw.split("===") if r.strip()]
    if not job_desc.strip() or not resumes:
        st.warning("Add a job description and at least one resume.")
        st.stop()

    with st.spinner(f"Screening {len(resumes)} resume(s)..."):
        results = []
        for i, r in enumerate(resumes):
            try:
                results.append((i + 1, screen_resume(r, job_desc, api_key=api_key or None), r))
            except Exception as e:
                st.error(f"Resume {i + 1} failed: {e}")

    results.sort(key=lambda x: x[1].score, reverse=True)

    st.subheader("Ranked candidates")
    for rank, (orig_idx, res, text) in enumerate(results, 1):
        icon = BAND_ICON.get(res.recommendation, "⚪")
        with st.expander(f"#{rank} — Resume {orig_idx} · {icon} {res.score}/100 · {res.recommendation.upper()}"):
            c1, c2 = st.columns(2)
            c1.markdown("**Matched:** " + (", ".join(res.matched_skills) or "none"))
            c2.markdown("**Missing:** " + (", ".join(res.missing_skills) or "none"))
            if res.years_experience:
                st.caption(f"Years experience: {res.years_experience:.0f}")
            st.caption(res.rationale)

    st.info(
        "⚠️ Screening aid only. Scores reflect skill keyword overlap, not candidate quality. "
        "A human must review every resume — especially 'maybe' and 'reject' bands — before any "
        "hiring decision. Never auto-reject."
    )
