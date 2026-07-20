from __future__ import annotations

# Streamlit UI for the LLM Guardrail Filter. Paste text, pick the direction
# (input to the model, or output from it), and see every rule that fired plus
# the final ALLOW / REDACT / BLOCK verdict. Fully offline.

import pandas as pd
import streamlit as st

from guardrails import (
    SAMPLE_INPUTS,
    SAMPLE_OUTPUTS,
    default_input_engine,
    default_output_engine,
)

st.set_page_config(page_title="LLM Guardrail Filter", page_icon="🛡️", layout="wide")

st.title("🛡️ LLM Guardrail Filter")
st.caption(
    "Cheap deterministic checks around the model: block prompt injection on the "
    "way in, redact PII and block secrets / toxicity / off-topic replies on the "
    "way out. No API keys - runs fully offline."
)

_BADGE = {"allow": "🟢 ALLOW", "redact": "🟡 REDACT", "block": "🔴 BLOCK"}

tab_try, tab_batch = st.tabs(["🧪 Try it", "📋 Sample traffic"])

with tab_try:
    direction = st.radio(
        "Direction", ["Input (to model)", "Output (from model)"], horizontal=True
    )
    is_input = direction.startswith("Input")
    default = SAMPLE_INPUTS[1] if is_input else SAMPLE_OUTPUTS[1]
    text = st.text_area("Text to check", value=default, height=120)

    if st.button("Run guardrails", type="primary"):
        engine = default_input_engine() if is_input else default_output_engine()
        v = engine.check(text)
        st.subheader(_BADGE[v.action.value])
        if v.hits:
            st.dataframe(
                pd.DataFrame(
                    [{"rule": h.rule, "action": h.action.value, "reason": h.reason} for h in v.hits]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No rules fired - clean.")
        if v.redacted:
            st.markdown("**Redacted text (safe to pass on):**")
            st.code(v.text)
        elif v.blocked:
            st.error("Blocked - this text should not reach the model / user.")

with tab_batch:
    st.subheader("Input guardrails")
    inp = default_input_engine()
    st.dataframe(
        pd.DataFrame(
            [
                {"text": t, "verdict": _BADGE[inp.check(t).action.value],
                 "rules": ", ".join(h.rule for h in inp.check(t).hits) or "-"}
                for t in SAMPLE_INPUTS
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Output guardrails")
    out = default_output_engine()
    rows = []
    for t in SAMPLE_OUTPUTS:
        v = out.check(t)
        rows.append(
            {
                "text": t,
                "verdict": _BADGE[v.action.value],
                "rules": ", ".join(h.rule for h in v.hits) or "-",
                "cleaned": v.text if v.redacted else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
st.caption("Day 84 of Phoebe's FDE portfolio · LLMOps & GenAI Platform · `python guardrails.py` for the CLI.")
