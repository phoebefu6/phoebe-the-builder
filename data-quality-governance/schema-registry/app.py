from __future__ import annotations

# Streamlit front-end for the Schema Registry. Pick a dataset, read its version
# history, then paste/edit a PROPOSED schema and see the field-level change list
# plus the compatibility verdict - color-coded so a steward can approve or block
# a producer's change BEFORE it ships and breaks downstream jobs silently.
import json
from typing import Dict, List

import pandas as pd
import streamlit as st
from registry import (
    Field,
    Schema,
    changes_frame,
    compare,
    make_sample_registry,
    sample_proposals,
)

st.set_page_config(page_title="Schema Registry", page_icon="R", layout="wide")

VERDICT_COLOR = {
    "FULL": "#1a7f37",       # green - safe both directions
    "BACKWARD": "#1f6feb",   # blue - backward only
    "FORWARD": "#9a6700",    # amber - forward only
    "BREAKING": "#cf222e",   # red - do not ship without a human sign-off
}


def fields_to_records(fields: List[Field]) -> List[Dict[str, object]]:
    return [
        {"name": f.name, "type": f.type, "nullable": f.nullable, "required": f.required}
        for f in fields
    ]


def records_to_fields(records: List[Dict[str, object]]) -> List[Field]:
    out: List[Field] = []
    for r in records:
        name = str(r.get("name", "")).strip()
        if not name:
            continue
        out.append(Field(
            name=name,
            type=str(r.get("type", "string")).strip(),
            nullable=bool(r.get("nullable", False)),
            required=bool(r.get("required", True)),
        ))
    return out


@st.cache_resource
def get_registry():
    return make_sample_registry()


reg = get_registry()

st.title("Schema Registry")
st.caption(
    "Register schema versions per dataset and check a proposed change for "
    "compatibility - BACKWARD, FORWARD, FULL, or BREAKING - before it ships."
)

# --- dataset picker + version history -------------------------------------
dataset = st.sidebar.selectbox("Dataset", reg.datasets())
versions = reg.versions(dataset)
latest = reg.latest(dataset)

st.header(f"Version history - {dataset}")
hist_rows = []
for s in versions:
    hist_rows.append({
        "version": f"v{s.version}",
        "fields": ", ".join(
            f"{f.name}:{f.type}"
            f"{'?' if f.nullable else ''}{'' if f.required else ' (opt)'}"
            for f in s.fields
        ),
    })
st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

st.subheader(f"Latest registered schema (v{latest.version})")
st.dataframe(
    pd.DataFrame(fields_to_records(latest.fields)),
    use_container_width=True, hide_index=True,
)

# --- proposed schema ------------------------------------------------------
st.header("Propose a new schema version")

proposals = sample_proposals()
choice = st.radio(
    "Start from a sample or edit your own:",
    ["Sample: BREAKING change", "Sample: safe change", "Edit latest as-is"],
    horizontal=True,
)
if choice == "Sample: BREAKING change":
    seed = proposals["breaking"]
elif choice == "Sample: safe change":
    seed = proposals["safe"]
else:
    seed = latest.fields

st.caption("Edit the fields below (add rows, change type, toggle nullable/required):")
edited = st.data_editor(
    pd.DataFrame(fields_to_records(seed)),
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "name": st.column_config.TextColumn("name", required=True),
        "type": st.column_config.SelectboxColumn(
            "type", options=["int", "long", "float", "double", "string", "bool"]
        ),
        "nullable": st.column_config.CheckboxColumn("nullable"),
        "required": st.column_config.CheckboxColumn("required"),
    },
    key="editor",
)

with st.expander("Or paste a JSON schema (list of {name, type, nullable, required})"):
    pasted = st.text_area("JSON", value="", height=160)
    use_json = st.checkbox("Use the pasted JSON instead of the table")

if use_json and pasted.strip():
    try:
        proposed_fields = records_to_fields(json.loads(pasted))
    except (ValueError, TypeError) as exc:
        st.error(f"Could not parse JSON: {exc}")
        proposed_fields = []
else:
    proposed_fields = records_to_fields(edited.to_dict("records"))

# --- verdict --------------------------------------------------------------
if st.button("Check compatibility", type="primary") and proposed_fields:
    proposed = Schema(dataset, latest.version + 1, proposed_fields)
    result = compare(latest, proposed)
    verdict = str(result["verdict"])
    color = VERDICT_COLOR.get(verdict, "#57606a")

    st.markdown(
        f"<div style='padding:14px 18px;border-radius:8px;background:{color};"
        f"color:white;font-size:1.25rem;font-weight:700;'>"
        f"VERDICT: {verdict}</div>",
        unsafe_allow_html=True,
    )
    st.write(result["summary"])

    changes = result["changes"]
    if not changes:
        st.info("No field changes detected - identical to the latest version.")
    else:
        cf = changes_frame(changes)
        st.subheader("Field-level changes")
        st.dataframe(cf, use_container_width=True, hide_index=True)

    if verdict == "BREAKING":
        st.error(
            "This change breaks compatibility in both directions. The rules are a "
            "heuristic - a human should confirm intent before shipping it."
        )
