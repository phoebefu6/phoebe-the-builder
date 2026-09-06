from __future__ import annotations

# Streamlit front-end for the Consent & Purpose Tracker. Upload your consent log and
# your processing register (or use the built-in sample), and see - by severity - every
# processing activity that can't point to a valid lawful basis, plus the headline
# "% of processing with valid basis" metric. All logic lives in tracker.py.
import pandas as pd
import streamlit as st
from tracker import (
    AS_OF,
    audit_consent,
    compliance_summary,
    findings_frame,
    make_sample_data,
)

st.set_page_config(page_title="Consent & Purpose Tracker", page_icon="🛡️", layout="wide")

st.title("🛡️ Consent & Purpose Tracker")
st.caption(
    "Cross-check what you're actually doing with personal data against what each "
    "subject consented to - and prove your lawful basis before someone asks."
)

_SEV_COLORS = {"high": "#c0392b", "medium": "#e67e22", "low": "#f1c40f"}


def _load_csv(upload, expected_cols: list) -> "pd.DataFrame | None":
    """Read an uploaded CSV and warn (do not crash) on missing columns."""
    if upload is None:
        return None
    df = pd.read_csv(upload)
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.error(f"Missing expected columns: {missing}")
        return None
    return df


with st.sidebar:
    st.header("Inputs")
    st.write(
        "Provide two CSVs, or tick the box to use the built-in sample with "
        "one planted issue of each kind plus some clean valid cases."
    )
    use_sample = st.checkbox("Use sample data", value=True)

    consent_cols = ["subject_id", "purpose", "status", "legal_basis", "timestamp", "expiry"]
    processing_cols = ["activity_id", "purpose", "dataset", "subjects_touched"]

    consent_up = st.file_uploader(
        "Consent records CSV", type="csv",
        help="columns: " + ", ".join(consent_cols),
    )
    processing_up = st.file_uploader(
        "Processing activities CSV", type="csv",
        help="columns: " + ", ".join(processing_cols),
    )

    st.markdown(f"**Audit as-of:** `{AS_OF.date()}` (fixed for reproducibility)")

# Resolve inputs: uploads win when present, otherwise fall back to the sample.
if use_sample and consent_up is None and processing_up is None:
    consent_df, processing_df = make_sample_data()
    st.info("Showing built-in sample data. Uncheck 'Use sample data' and upload CSVs to audit your own.")
else:
    consent_df = _load_csv(consent_up, consent_cols)
    processing_df = _load_csv(processing_up, processing_cols)
    if consent_df is None or processing_df is None:
        st.warning("Upload both CSVs (or tick 'Use sample data') to run the audit.")
        st.stop()

# Run the audit.
findings = audit_consent(consent_df, processing_df, AS_OF)
summary = compliance_summary(processing_df, findings)
frame = findings_frame(findings)

# Headline metrics.
c1, c2, c3, c4 = st.columns(4)
c1.metric("% with valid basis", f"{summary['pct_valid_basis']}%")
c2.metric("Processing checks", summary["total_checks"])
c3.metric("Valid basis", summary["valid_basis"])
c4.metric("Flagged for review", summary["flagged"])

st.divider()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Inputs")
    st.markdown("**Consent records**")
    st.dataframe(consent_df, use_container_width=True, hide_index=True)
    st.markdown("**Processing activities**")
    st.dataframe(processing_df, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("Findings by severity")
    if frame.empty:
        st.success("No findings - every processing activity has a valid basis. 🎉")
    else:
        by_issue = summary["by_issue"]
        chips = "  ".join(f"`{k}: {v}`" for k, v in by_issue.items())
        st.markdown(f"Issue breakdown: {chips}")
        for sev in ["high", "medium", "low"]:
            block = frame[frame["severity"] == sev]
            if block.empty:
                continue
            color = _SEV_COLORS.get(sev, "#7f8c8d")
            st.markdown(
                f"<h4 style='color:{color};margin-bottom:0'>{sev.upper()} ({len(block)})</h4>",
                unsafe_allow_html=True,
            )
            for _, r in block.iterrows():
                with st.expander(
                    f"{r['issue']} - {r['subject_id']} / {r['purpose']} "
                    f"(activity {r['activity_id']}, dataset {r['dataset']})"
                ):
                    st.write(r["reason"])

st.divider()
st.caption(
    "Findings are a compliance review queue, not legal advice. Legal owns the "
    "lawful-basis call - this tool surfaces the gaps and explains each one."
)
