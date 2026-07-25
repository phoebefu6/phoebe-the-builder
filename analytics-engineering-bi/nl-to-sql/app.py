from __future__ import annotations

import streamlit as st

from nl2sql import (
    SAMPLE_QUESTIONS,
    SCHEMA,
    run_sql,
    sample_dataframe,
    translate,
)

st.set_page_config(page_title="Natural Language to SQL", layout="wide")
st.title("Natural Language to SQL")
st.caption("Let non-analysts ask questions in English — with guardrails that keep it read-only and in-schema.")

with st.sidebar:
    st.subheader("Schema")
    st.code(
        f"TABLE {SCHEMA.table}\n" + "\n".join(f"  {c} : {t}" for c, t in SCHEMA.columns.items()),
        language="text",
    )
    st.subheader("Guardrails")
    st.markdown(
        "- SELECT only (no writes/DDL)\n- Whitelisted table + columns\n"
        "- No multi-statements\n- Auto `LIMIT` injected"
    )
    st.caption("Set ANTHROPIC_API_KEY to use Claude. Falls back to a rule-based translator.")

df = sample_dataframe()

st.subheader("Ask a question")
example = st.selectbox("Examples", ["(type your own)"] + SAMPLE_QUESTIONS)
default_q = "" if example == "(type your own)" else example
question = st.text_input("Question", value=default_q or "Top 3 regions by total amount")

if st.button("Translate & run", type="primary"):
    if not question.strip():
        st.warning("Ask something first.")
        st.stop()

    guard = translate(question, SCHEMA)

    st.markdown("**Generated SQL**")
    st.code(guard.sql, language="sql")

    if not guard.ok:
        st.error("Blocked by guardrails:\n- " + "\n- ".join(guard.issues))
        st.stop()

    try:
        result = run_sql(guard.sql, df, SCHEMA.table)
    except Exception as e:
        st.error(f"Execution error: {e}")
        st.stop()

    st.markdown("**Result**")
    st.dataframe(result, use_container_width=True)

with st.expander("Sample data"):
    st.dataframe(df, use_container_width=True)
