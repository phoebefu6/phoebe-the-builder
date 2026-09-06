"""Schema from Samples - infer a JSON Schema contract from example LLM outputs."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from schema_infer import (
    Policy,
    drifted_corpus,
    evaluate,
    infer_schema,
    loose_schema,
    sample_corpus,
    strict_schema,
    to_json,
)

st.set_page_config(page_title="Schema from Samples", page_icon="📐", layout="wide")

st.title("📐 Schema from Samples")
st.caption(
    "Paste example LLM outputs (JSON Lines or a JSON array). Get back a frequency-aware "
    "JSON Schema contract, a findings report of what the evidence could not support, and "
    "a benchmark against the loose and strict schemas people usually ship."
)

with st.sidebar:
    st.header("Policy")
    required_at = st.slider("Required at presence ≥", 0.80, 1.00, 0.98, 0.01)
    enum_max_distinct = st.slider("Enum: max distinct values", 2, 30, 8)
    enum_min_support = st.slider("Enum: min occurrences per value", 1, 10, 2)
    closed = st.checkbox("Closed objects (additionalProperties: false)", value=True)
    bounds = st.checkbox("Numeric bounds (rounded out)", value=True)
    st.divider()
    use_sample = st.button("Load sample corpus (60 support-ticket extractions)")

policy = Policy(
    required_at=required_at,
    enum_max_distinct=enum_max_distinct,
    enum_min_support=enum_min_support,
    closed_objects=closed,
    numeric_bounds=bounds,
)

if "raw" not in st.session_state:
    st.session_state.raw = ""
if use_sample:
    st.session_state.raw = "\n".join(json.dumps(r) for r in sample_corpus(60))

raw = st.text_area(
    "Example outputs - one JSON object per line, or a single JSON array",
    value=st.session_state.raw,
    height=200,
    placeholder='{"category": "billing", "priority": "high", ...}\n{"category": "technical", ...}',
)


def parse_samples(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    samples = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Line {i} is not valid JSON: {e}") from e
    return samples


if raw.strip():
    try:
        samples = parse_samples(raw)
    except (ValueError, json.JSONDecodeError) as e:
        st.error(str(e))
        st.stop()

    if not samples:
        st.info("No samples yet.")
        st.stop()

    inf = infer_schema(samples, policy=policy)

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Inferred contract")
        st.code(to_json(inf.schema), language="json")
        st.download_button(
            "Download schema.json",
            to_json(inf.schema),
            file_name="schema.json",
            mime="application/json",
        )

    with right:
        st.subheader("Findings - what the evidence could not support")
        findings = inf.findings_table()
        if findings:
            icon = {"block": "🟥", "warn": "🟧", "info": "🟦"}
            for f in findings:
                st.markdown(f"{icon[f['severity']]} **`{f['path']}`** - {f['kind']}\n\n{f['detail']}")
        else:
            st.success("No abstentions - every constraint is fully evidenced.")

        st.subheader("Field evidence")
        st.dataframe(pd.DataFrame(inf.field_table()), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Benchmark vs. the usual suspects")
    st.caption(
        "Scored on the built-in holdout + 6 labelled drift cases when using the sample corpus; "
        "for your own data, split it yourself and validate with any JSON Schema library."
    )

    if st.session_state.raw and samples == sample_corpus(60):
        train, holdout = samples[:45], samples[45:]
        inf_t = infer_schema(train, policy=policy)
        rows = []
        for label, schema in [
            ("inferred (this tool)", inf_t.schema),
            ("loose: union-of-everything", loose_schema(train)),
            ("strict: require + enum all", strict_schema(train)),
        ]:
            r = evaluate(schema, holdout, drifted_corpus())
            rows.append(
                {
                    "schema": label,
                    "valid holdout rejected": f"{r['false_reject_rate']:.0%}",
                    "drift caught": f"{r['catch_rate']:.0%}",
                    "missed drift": ", ".join(r["missed"]) or "-",
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("Load the sample corpus from the sidebar to see the three-way benchmark.")
else:
    st.info("Paste outputs above, or load the sample corpus from the sidebar.")
