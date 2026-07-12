from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dedup import (add_match_keys, dedupe, longest_value, make_sample_customers,
                   most_complete_record, most_recent, source_priority)

st.set_page_config(page_title="Dedup & Survivorship", page_icon="🧬", layout="wide")
st.title("🧬 Dedup & Survivorship Pipeline")
st.caption("Duplicate records everywhere — normalize, cluster, and merge into golden records with field-level provenance.")

RULE_NAMES = {"most_recent": most_recent, "longest_value": longest_value,
              "most_complete_record": most_complete_record}

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("CSV (needs record_id, source, updated_at)", type=["csv"])
    st.header("Survivorship rules")
    priority = st.text_input("Source priority (comma-sep)", "crm,web,import")
    email_rule = "source_priority"
    phone_rule = st.selectbox("phone", list(RULE_NAMES), index=0)
    address_rule = st.selectbox("address", list(RULE_NAMES), index=1)
    name_rule = st.selectbox("name", list(RULE_NAMES), index=2)

if st.button("Run dedup", type="primary"):
    df = pd.read_csv(uploaded) if uploaded else make_sample_customers()
    work = add_match_keys(df)
    prio = source_priority([s.strip() for s in priority.split(",")])
    rules = {"email": prio, "loyalty_tier": prio,
             "phone": RULE_NAMES[phone_rule], "address": RULE_NAMES[address_rule],
             "name": RULE_NAMES[name_rule]}
    golden, rep = dedupe(work, ["email_norm", "phone_norm"], rules)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw rows", rep.input_rows)
    c2.metric("Golden records", rep.golden_rows)
    c3.metric("Duplicates merged", rep.dup_rows_merged)
    c4.metric("Multi-record clusters", int((golden["cluster_size"] > 1).sum()))

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Golden records (merged clusters first)")
        st.dataframe(golden.sort_values("cluster_size", ascending=False),
                     use_container_width=True, hide_index=True, height=320)
    with col2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))
        golden["cluster_size"].value_counts().sort_index().plot.bar(ax=ax1, color="#457b9d")
        ax1.set_title("Cluster size distribution")
        ax1.set_xlabel("records per cluster")
        if len(rep.field_provenance):
            rep.field_provenance["winning_source"].value_counts().plot.barh(ax=ax2, color="#2a9d8f")
            ax2.set_title("Field wins by source")
        fig.tight_layout()
        st.pyplot(fig)

    if len(rep.field_provenance):
        st.subheader("Field-level provenance (who won each field, and why)")
        st.dataframe(rep.field_provenance, use_container_width=True, hide_index=True, height=240)
else:
    st.info("Upload a CSV or use the sample, then click **Run dedup**.")
