from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from contract import diff_contracts, exit_code, load_contract, validate_dataset

st.set_page_config(page_title="Data Contract Validator", page_icon="📜", layout="wide")
st.title("📜 Data Contract Validator")
st.caption("Producers break schemas without warning — put a contract in CI and catch it before consumers do.")

CONTRACT = """dataset: orders
owner: checkout-team
columns:
  - name: order_id
    type: string
    nullable: false
    unique: true
  - name: amount
    type: float
    nullable: false
    min: 0
  - name: status
    type: string
    allowed: [pending, paid, shipped, cancelled]
  - name: created_at
    type: datetime
    nullable: false
freshness:
  column: created_at
  max_age_hours: 24
  as_of: 2026-07-11 12:00:00
"""

SAMPLE_CSV = """order_id,amount,status,created_at,coupon_code
ord-1,25.00,paid,2026-07-11 09:00:00,SAVE10
ord-2,-4.00,paid,2026-07-11 09:05:00,
ord-3,60.00,refunded,2026-07-11 09:10:00,
ord-3,60.00,paid,2026-07-11 09:12:00,
ord-5,,pending,2026-07-11 09:20:00,
"""

tab1, tab2 = st.tabs(["Validate dataset", "Diff contract versions"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        contract_text = st.text_area("Contract (YAML)", CONTRACT, height=330)
    with col2:
        csv_text = st.text_area("Dataset (CSV)", SAMPLE_CSV, height=330)
    if st.button("Validate", type="primary"):
        contract = load_contract(contract_text)
        df = pd.read_csv(io.StringIO(csv_text))
        violations = validate_dataset(df, contract)
        code = exit_code(violations)
        (st.error if code else st.success)(f"CI exit code: {code}")
        if violations:
            vdf = pd.DataFrame([{"Level": v.level, "Column": v.column, "Rule": v.rule,
                                 "Detail": v.detail} for v in violations])
            c1, c2 = st.columns([2, 1])
            with c1:
                st.dataframe(vdf, use_container_width=True, hide_index=True)
            with c2:
                counts = vdf.groupby("Rule").size().sort_values()
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.barh(counts.index, counts.values, color="#e76f51")
                ax.set_title("Violations by rule")
                fig.tight_layout()
                st.pyplot(fig)

with tab2:
    NEW_CONTRACT = CONTRACT.replace("  - name: status\n    type: string\n    allowed: [pending, paid, shipped, cancelled]\n",
                                    "").replace("type: float", "type: string")
    col1, col2 = st.columns(2)
    with col1:
        old_text = st.text_area("Old contract", CONTRACT, height=280)
    with col2:
        new_text = st.text_area("New contract (proposed)", NEW_CONTRACT, height=280)
    if st.button("Diff contracts", type="primary"):
        changes = diff_contracts(load_contract(old_text), load_contract(new_text))
        code = exit_code(changes)
        (st.error if code else st.success)(
            f"{'BREAKING changes — block the release' if code else 'No breaking changes'} (exit {code})")
        for v in changes:
            (st.error if v.level == "error" else st.warning)(f"`{v.column}` — {v.rule}: {v.detail}")
