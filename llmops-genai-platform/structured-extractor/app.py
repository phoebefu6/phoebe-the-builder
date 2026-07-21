from __future__ import annotations

# Streamlit UI for the Structured Output Enforcer. Paste any raw LLM response,
# declare the fields you expected, and watch the extract -> repair -> coerce ->
# validate pipeline turn messy text into a validated object (or tell you
# exactly why it can't). Fully offline - no API keys.

import pandas as pd
import streamlit as st

from enforcer import (
    SAMPLE_OUTPUTS,
    TICKET_SCHEMA,
    Field,
    Schema,
    enforce,
    run_sample,
    summarize,
)

st.set_page_config(page_title="Structured Output Enforcer", page_icon="🧩",
                   layout="wide")

st.title("🧩 Structured Output Enforcer")
st.caption(
    "You asked the LLM for JSON and got a code fence, a chatty preamble, a "
    "trailing comma, or Python's True/None. Plain json.loads() throws and your "
    "pipeline dies. This enforcer pulls the JSON out of the noise, repairs the "
    "common malformations, coerces types, and validates your schema - and "
    "tells you exactly what it fixed."
)

# --- Batch view over the sample messy outputs ---------------------------- #
rows = run_sample()
s = summarize(rows)

c1, c2, c3 = st.columns(3)
c1.metric("Naive json.loads()", f"{s['naive_pct']}%", f"{s['naive_ok']}/{s['total']} parse")
c2.metric("With enforcer", f"{s['valid_pct']}%", f"{s['valid']}/{s['total']} valid")
c3.metric("Outputs recovered", f"+{s['recovered']}", "would have crashed")

st.subheader("Sample model outputs")
df = pd.DataFrame(rows)[["case", "valid", "repaired", "repairs", "errors"]]
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

# --- Try your own -------------------------------------------------------- #
st.subheader("Try your own")
left, right = st.columns([3, 2])

with right:
    st.markdown("**Expected fields**")
    st.caption("Edit the schema; extra keys in the output are kept as-is.")
    schema_df = st.data_editor(
        pd.DataFrame([{"name": f.name, "type": f.type, "required": f.required}
                      for f in TICKET_SCHEMA.fields]),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="schema_editor",
    )

with left:
    preset = st.selectbox("Load a sample output", ["(custom)"] + [c for c, _ in SAMPLE_OUTPUTS])
    default_raw = ""
    if preset != "(custom)":
        default_raw = dict(SAMPLE_OUTPUTS)[preset]
    raw = st.text_area("Raw LLM response", value=default_raw, height=200,
                       placeholder='```json\n{"category": "billing", ...}\n```')

if st.button("Enforce", type="primary") and raw.strip():
    fields = [
        Field(str(r["name"]), str(r["type"]), bool(r["required"]))
        for _, r in schema_df.iterrows() if str(r["name"]).strip()
    ]
    res = enforce(raw, Schema(fields))

    if res.ok:
        st.success("Valid" + (" (after repair)" if res.repaired else ""))
        st.json(res.data)
    else:
        st.error("Invalid - " + "; ".join(res.errors))
        if res.data:
            st.caption("Best-effort parse:")
            st.json(res.data)

    if res.repairs:
        st.markdown("**Repairs applied:** " + ", ".join(res.repairs))

st.divider()
st.caption("Day 87 of Phoebe's FDE build sprint · LLMOps & GenAI Platform · "
           "offline lexical repair, swap in your real LLM response text.")
