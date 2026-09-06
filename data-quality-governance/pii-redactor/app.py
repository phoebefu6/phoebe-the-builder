"""Streamlit UI for the policy-driven PII redactor.

Run: streamlit run app.py
"""

from __future__ import annotations

import secrets

import matplotlib.pyplot as plt
import pandas as pd
import redact as R
import streamlit as st

st.set_page_config(page_title="PII Redactor", layout="wide")

st.title("Policy-Driven PII Redactor")
st.caption(
    "Redact an extract, keep the joins working, and score what could still "
    "re-identify someone."
)

# ------------------------------------------------------------------ session
if "salt" not in st.session_state:
    st.session_state.salt = secrets.token_bytes(32)

# ------------------------------------------------------------------- inputs
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("CSV to redact", type="csv")
    if uploaded is None:
        st.caption("No file: using the bundled synthetic member table.")
        n_rows = st.slider("Sample rows", 100, 5000, 400, 100)
        df, claims = R.sample_data(n=n_rows)
    else:
        df = pd.read_csv(uploaded)
        claims = None

    st.header("Tokenization key")
    st.caption(
        "A random 32-byte salt is generated per session and never written to "
        "disk. Same salt gives the same tokens, which is what preserves joins "
        "across tables."
    )
    if st.button("Rotate salt"):
        st.session_state.salt = secrets.token_bytes(32)
        st.rerun()

    st.header("Re-identification check")
    quasi = st.multiselect(
        "Quasi-identifiers",
        list(df.columns),
        default=[c for c in R.QUASI_IDS if c in df.columns],
        help="Columns that are not identifiers on their own but can single "
        "someone out in combination.",
    )
    target_k = st.slider("Target k", 2, 25, 5)

salt = st.session_state.salt

# --------------------------------------------------------------- the policy
st.subheader("1. Redaction policy")
st.caption(
    "One rule per column. `tokenize` keeps joins, `mask` does not. "
    "`generalize` is the only strategy that reduces re-identification risk."
)

default_policy = {c: R.DEFAULT_POLICY.get(c, {"strategy": "keep"}) for c in df.columns}
editor_rows = []
for col in df.columns:
    rule = default_policy[col]
    editor_rows.append(
        {
            "column": col,
            "strategy": rule.get("strategy", "keep"),
            "sample_value": str(df[col].dropna().iloc[0])[:40]
            if df[col].notna().any()
            else "",
            "distinct": int(df[col].nunique(dropna=True)),
        }
    )

edited = st.data_editor(
    pd.DataFrame(editor_rows),
    hide_index=True,
    use_container_width=True,
    disabled=["column", "sample_value", "distinct"],
    column_config={
        "strategy": st.column_config.SelectboxColumn(
            "strategy", options=list(R.STRATEGIES), required=True
        )
    },
)

policy = {}
for row in edited.itertuples():
    rule = dict(default_policy.get(row.column, {"strategy": "keep"}))
    if rule.get("strategy") != row.strategy:
        rule = {"strategy": row.strategy}
        if row.strategy == "generalize":
            # pick a sensible default kind from the original policy or dtype
            base = R.DEFAULT_POLICY.get(row.column, {}).get("params")
            rule["params"] = base or {"kind": "numeric_band", "width": 10000.0}
    policy[row.column] = rule

try:
    redacted, audit = R.apply_policy(df, policy, salt)
except (ValueError, TypeError) as exc:
    st.error(f"Policy failed on this data: {exc}")
    st.stop()

# --------------------------------------------------------------- the result
st.subheader("2. Redacted output")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Columns in", len(df.columns))
c2.metric("Columns out", len(redacted.columns))
c3.metric("Dropped", len(df.columns) - len(redacted.columns))
c4.metric("Join-safe columns", int(audit["join_safe"].sum()))

st.dataframe(redacted.head(12), use_container_width=True)
st.download_button(
    "Download redacted CSV",
    redacted.to_csv(index=False).encode(),
    file_name="redacted.csv",
    mime="text/csv",
)

with st.expander("Audit log (what was done to each column)", expanded=True):
    st.dataframe(audit, use_container_width=True, hide_index=True)
    st.download_button(
        "Download audit log",
        audit.to_csv(index=False).encode(),
        file_name="redaction_audit.csv",
        mime="text/csv",
    )

for note in R.low_cardinality_warnings(df, audit):
    st.warning(note)

# ----------------------------------------------------- re-identification
st.subheader("3. Could someone still be singled out?")
if not quasi:
    st.info("Pick at least one quasi-identifier to score re-identification risk.")
else:
    before = R.k_anonymity(df, quasi)
    after = R.k_anonymity(redacted, quasi)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("k before", before["k"] if before["k"] is not None else "n/a")
    c2.metric("k after", after["k"] if after["k"] is not None else "n/a")
    c3.metric(
        "Unique rows before",
        f"{before['singleton_share'] * 100:.1f}%",
    )
    c4.metric(
        "Unique rows after",
        f"{after['singleton_share'] * 100:.1f}%",
        delta=f"{(after['singleton_share'] - before['singleton_share']) * 100:.1f} pts",
        delta_color="inverse",
    )

    if after["k"] is not None and after["k"] < target_k:
        kept, suppressed = R.suppress_below_k(redacted, quasi, target_k)
        share = suppressed / len(redacted) if len(redacted) else 0.0
        if share >= 1.0:
            st.error(
                f"k = {after['k']}, below your target of {target_k}. Reaching "
                f"k={target_k} by suppression would delete every row, so "
                "generalization has to get coarser instead. Try wider bands or "
                "drop a quasi-identifier."
            )
        else:
            st.warning(
                f"k = {after['k']}, below your target of {target_k}. Suppressing "
                f"the {suppressed:,} rows in classes smaller than {target_k} "
                f"({share * 100:.1f}% of the extract) would get you there, at the "
                "cost of biasing the extract toward common cases."
            )
    elif after["k"] is not None:
        st.success(f"k = {after['k']}, clears your target of {target_k}.")

    st.markdown(
        "**Singleton share is the number to watch, not k.** In any real dataset "
        "some rare combination exists, so k pins to 1 and stops being "
        "informative. The share of rows that are unique tells you how much of "
        "the extract is actually exposed."
    )

    # generalization ladder, only for the bundled schema
    if all(c in df.columns for c in ("postal_code", "date_of_birth", "gender")):
        st.markdown("#### How coarse do you have to go?")
        ladder = R.generalization_ladder(df, target_k=target_k)
        st.dataframe(ladder, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(
            range(len(ladder)),
            ladder["singleton_share"] * 100,
            marker="o",
            color="#2b6cb0",
            lw=2,
        )
        ax.set_xticks(range(len(ladder)))
        ax.set_xticklabels(ladder["level"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Rows unique on quasi-IDs (%)")
        ax.set_title("Generalization is a dial with a utility cost")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

# -------------------------------------------------------------- join proof
if claims is not None:
    st.subheader("4. Did the joins survive?")
    claims_red, _ = R.apply_policy(
        claims, {"member_id": {"strategy": "tokenize"}}, salt
    )
    proof = R.verify_join(df, claims, redacted, claims_red, "member_id")
    c1, c2, c3 = st.columns(3)
    c1.metric("Join rows before", f"{proof['rows_before']:,}")
    c2.metric("Join rows after", f"{proof['rows_after']:,}")
    c3.metric("Referential integrity", "preserved" if proof["preserved"] else "BROKEN")
    if proof["preserved"]:
        st.success(
            "member_id was tokenized with the same salt in both tables, so the "
            "extract still joins. Masking it instead would have silently broken "
            "this."
        )
    else:
        st.error(
            "The join no longer returns the same rows. Check that the key uses "
            "`tokenize` (not `mask` or `hash`) in every table sharing it."
        )

st.divider()
st.caption(
    "Day 124 of Phoebe's FDE portfolio - Data Quality & Governance. "
    "Tokenization is pseudonymization, not anonymization: under GDPR and PDPA "
    "the result is still personal data."
)
