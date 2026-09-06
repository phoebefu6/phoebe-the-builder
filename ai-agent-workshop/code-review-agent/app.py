from __future__ import annotations

import pandas as pd
import streamlit as st
from reviewer import fetch_pr_diff, review_diff

st.set_page_config(page_title="Code Review Agent", page_icon="🔍")
st.title("🔍 Code Review Agent")
st.caption("Paste a diff or a GitHub PR and get automated findings + a Claude-written review summary.")

SAMPLE_DIFF = '''--- a/app/utils.py
+++ b/app/utils.py
@@ -10,6 +10,12 @@
 def process(data):
+    api_key = "sk-1234567890abcdef"
+    try:
+        result = call_api(data, api_key)
+    except:
+        print("failed")
+    # TODO: add retry logic
+    return result
'''

with st.sidebar:
    st.subheader("Settings")
    st.info("Set ANTHROPIC_API_KEY for a Claude-written summary. Set GITHUB_TOKEN to fetch private PRs.")
    source = st.radio("Diff source", ["Paste diff", "GitHub PR"])

diff_text = ""
if source == "Paste diff":
    diff_text = st.text_area("Unified diff", value=SAMPLE_DIFF, height=220)
else:
    repo = st.text_input("Repo (owner/name)", value="phoebefu6/phoebe-the-builder")
    pr_number = st.number_input("PR number", min_value=1, value=1, step=1)
    if st.button("Fetch diff"):
        try:
            diff_text = fetch_pr_diff(repo, int(pr_number))
            st.session_state["fetched_diff"] = diff_text
        except Exception as e:
            st.error(f"Could not fetch PR: {e}")
    diff_text = st.session_state.get("fetched_diff", "")
    if diff_text:
        st.text_area("Fetched diff", value=diff_text, height=220)

if st.button("Review", type="primary"):
    if not diff_text.strip():
        st.error("No diff to review.")
    else:
        with st.spinner("Agent scanning diff for issues..."):
            result = review_diff(diff_text)

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        severity_color = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪"}

        st.subheader("Review Summary")
        st.write(result.summary)

        st.subheader(f"Findings ({len(result.findings)})")
        if result.findings:
            sorted_findings = sorted(result.findings, key=lambda f: severity_order[f.severity])
            df = pd.DataFrame([
                {
                    "Severity": f"{severity_color[f.severity]} {f.severity}",
                    "File": f.file,
                    "Line": f.line,
                    "Issue": f.message,
                }
                for f in sorted_findings
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.success("No issues found.")
