from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from a11y_checker import check_html, grade, report_markdown, score

st.set_page_config(page_title="Accessibility Checker", page_icon="♿", layout="wide")
st.title("♿ Accessibility Checker")
st.caption("Paste HTML → WCAG-referenced findings with severity, score, and a downloadable report.")

SAMPLE = """<html>
<head><meta charset="utf-8"></head>
<body>
  <h1>Product Page</h1>
  <h4>Details</h4>
  <img src="hero.png">
  <img src="logo.png" alt="image">
  <a href="/buy">click here</a>
  <a href="/spec"></a>
  <div onclick="openMenu()">Menu</div>
  <form>
    <input type="text" name="email">
    <button></button>
  </form>
  <p tabindex="3">Featured</p>
</body>
</html>"""

with st.sidebar:
    st.header("Input")
    use_sample = st.checkbox("Use sample HTML (11 planted issues)", value=True)
    uploaded = st.file_uploader("...or upload an .html file", type=["html", "htm"])

if uploaded:
    html = uploaded.read().decode("utf-8", errors="replace")
else:
    html = st.text_area("HTML to check", value=SAMPLE if use_sample else "", height=260)

if st.button("Run checks", type="primary") and html.strip():
    findings = check_html(html)
    s = score(findings)

    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{s}/100")
    c2.metric("Grade", grade(s))
    c3.metric("Issues", len(findings))

    if findings:
        df = pd.DataFrame([{"Severity": f.severity, "Rule": f.rule, "WCAG": f.wcag,
                            "Element": f.element, "Problem": f.message} for f in findings])
        order = ["critical", "serious", "moderate", "minor"]
        df["Severity"] = pd.Categorical(df["Severity"], categories=order, ordered=True)
        df = df.sort_values("Severity")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df, use_container_width=True, hide_index=True)
        with col2:
            counts = df["Severity"].value_counts().reindex(order).fillna(0)
            colors = ["#d62828", "#e76f51", "#e9c46a", "#8ecae6"]
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(counts.index, counts.values, color=colors)
            ax.set_ylabel("Findings")
            ax.set_title("Issues by severity")
            fig.tight_layout()
            st.pyplot(fig)

        st.download_button("⬇️ Download report (Markdown)", report_markdown(findings),
                           file_name="a11y_report.md", mime="text/markdown")
    else:
        st.success("No issues found by automated checks. Still test with a keyboard and screen reader.")
else:
    st.info("Paste HTML (or keep the sample) and click **Run checks**.")
