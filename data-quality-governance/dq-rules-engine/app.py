from __future__ import annotations

# Streamlit front end for the DQ Rules Engine. Upload a CSV (or use the built-in
# sample), see the declarative rule set, run it, and read a pass/fail board with
# severity colour and sample offending values. The whole point: make the rules
# visible and runnable, so they stop rotting in a wiki nobody enforces.
from typing import List

import pandas as pd
import streamlit as st
from engine import (
    make_sample_data,
    rollup,
    run_rules,
    sample_rules,
    summarize,
)

st.set_page_config(page_title="DQ Rules Engine", page_icon="check", layout="wide")

st.title("DQ Rules Engine")
st.caption(
    "Declarative data-quality checks that run on every batch - so a bad batch is "
    "caught before it lands in a report, not after."
)

# --- data source ---------------------------------------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use built-in sample orders table", value=uploaded is None)

if uploaded is not None and not use_sample:
    df = pd.read_csv(uploaded)
    rules = sample_rules()  # sample rules; edit below for your own columns
    st.sidebar.info(
        "Loaded your CSV. The default rule set targets the sample schema - edit "
        "the rules in the code, or upload data matching order_id / customer_id / "
        "amount / status / email to see them fire."
    )
else:
    df = make_sample_data()
    rules = sample_rules()

st.subheader("Data preview")
st.caption(f"{len(df):,} rows x {df.shape[1]} columns")
st.dataframe(df.head(20), use_container_width=True)

# --- the rules -----------------------------------------------------------
st.subheader("Rules")
st.caption("Each rule is a plain dict - this is what would live in YAML or a governance table instead of a wiki page.")
rule_rows = []
for r in rules:
    rule_rows.append({
        "rule": r.get("name", r["type"]),
        "type": r["type"],
        "column": r.get("column", "-"),
        "severity": r.get("severity", "error"),
    })
st.dataframe(pd.DataFrame(rule_rows), use_container_width=True)

# --- run -----------------------------------------------------------------
if st.button("Run rules", type="primary"):
    results = run_rules(df, rules)
    roll = rollup(results)

    status = roll["overall_status"]
    banner = {"PASS": st.success, "WARN": st.warning, "FAIL": st.error}[status]
    banner(
        f"Overall: {status}  -  {roll['passed']}/{roll['total']} rules passed, "
        f"{roll['errors_failed']} error-level failures, "
        f"{roll['warns_failed']} warn-level failures."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Rules passed", f"{roll['passed']}/{roll['total']}")
    c2.metric("Error failures", roll["errors_failed"])
    c3.metric("Warn failures", roll["warns_failed"])

    st.subheader("Results")
    summary = summarize(results)

    def _row_style(row: pd.Series) -> List[str]:
        # Colour by outcome + severity: red for a failing error (a real stop),
        # amber for a failing warn (a smell), soft green for a pass.
        if row["status"] == "PASS":
            colour = "background-color: rgba(46, 160, 67, 0.15)"
        elif row["severity"] == "error":
            colour = "background-color: rgba(248, 81, 73, 0.20)"
        else:
            colour = "background-color: rgba(210, 153, 34, 0.20)"
        return [colour] * len(row)

    st.dataframe(summary.style.apply(_row_style, axis=1), use_container_width=True)

    # --- failing detail with sample offending values --------------------
    failures = [r for r in results if not r.passed]
    if failures:
        st.subheader("Failing rules - what to look at")
        for r in failures:
            icon = "[ERROR]" if r.severity == "error" else "[WARN]"
            with st.expander(f"{icon} {r.name} - {r.message}", expanded=r.severity == "error"):
                st.write(f"**Type:** `{r.rule_type}`  |  **Column:** {r.column or '-'}")
                st.write(f"**Violations:** {r.n_violations} of {r.n_checked} checked")
                if r.samples:
                    st.write("**Sample offending values:**")
                    st.code("\n".join(str(s) for s in r.samples))
    else:
        st.balloons()
        st.success("Every rule passed - this batch is clean.")

    st.caption(
        "Reminder: rules are heuristics a human owns. A flagged value may be a "
        "legitimate exception; treat this as a prioritised review queue, not an "
        "auto-reject gate."
    )
else:
    st.info("Pick a data source in the sidebar, then click **Run rules**.")
