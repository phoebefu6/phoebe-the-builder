from __future__ import annotations

# Streamlit UI for the Prompt Linter. Paste a prompt, get findings with fixes,
# a 0-100 score, and a CI pass/fail verdict. Fully offline - static analysis,
# no model calls.
import pandas as pd
import streamlit as st
from lint import (
    CLEAN_PROMPT,
    RULES,
    SLOPPY_PROMPT,
    gate,
    lint,
)

st.set_page_config(page_title="Prompt Linter", page_icon="🔎", layout="wide")

st.title("🔎 Prompt Linter")
st.caption(
    "Prompts are the only part of an LLM system that ships with no review gate. "
    f"This applies {len(RULES)} static rules - ambiguity, contradictions, missing "
    "output contract, injection risk - and returns a fix for each. No model calls, "
    "so it runs in CI on every prompt change."
)

SEV_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}

with st.sidebar:
    st.header("Sample prompts")
    choice = st.radio("Load", ["Sloppy (real-world)", "Clean (fixed)", "Blank"], index=0)
    st.header("CI gate")
    min_score = st.slider("Minimum score", 0, 100, 80, 5)
    max_high = st.number_input("Max high-severity allowed", 0, 10, 0)

preset = {"Sloppy (real-world)": SLOPPY_PROMPT, "Clean (fixed)": CLEAN_PROMPT, "Blank": ""}[choice]
prompt = st.text_area("Prompt", preset, height=260, key=choice)

result = lint(prompt)
g = gate(prompt, min_score=int(min_score), max_high=int(max_high))
sev = result["by_severity"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Score", f"{result['score']}/100", result["grade"])
c2.metric("🔴 High", sev["high"])
c3.metric("🟠 Medium", sev["medium"])
c4.metric("🟡 Low", sev["low"])
c5.metric("CI gate", "PASS" if g["passed"] else "FAIL")

st.progress(result["score"] / 100)

if g["passed"]:
    st.success(f"Gate passed - score {g['score']}, no blocking findings.")
else:
    st.error("Gate failed: " + "; ".join(g["reasons"]))

tab_find, tab_table, tab_rules = st.tabs(["🩹 Findings & fixes", "📋 Table", "📖 Rule reference"])

with tab_find:
    if not result["findings"]:
        st.success("No findings. This prompt states its output contract, bounds its "
                   "length, delimits its input, and names a fallback.")
    for f in result["findings"]:
        loc = f" · line {f['line']}" if f["line"] else ""
        with st.expander(
            f"{SEV_ICON[f['severity']]} **{f['rule_id']}** · {f['category']}{loc} — {f['message']}",
            expanded=f["severity"] == "high",
        ):
            if f["snippet"]:
                st.code(f["snippet"], language=None)
            st.markdown(f"**Fix:** {f['fix']}")

with tab_table:
    if result["findings"]:
        df = pd.DataFrame(result["findings"])[
            ["rule_id", "severity", "category", "line", "message", "fix"]
        ]
        st.dataframe(df, width="stretch", hide_index=True)
        st.bar_chart(df.groupby("category").size().rename("findings"))
    else:
        st.info("Nothing to tabulate.")

with tab_rules:
    st.write(
        "Every rule maps to a failure that shows up in production, not a style "
        "preference. Severity reflects blast radius, not how easy it is to fix."
    )
    ref = [
        ("PL001", "high", "output-contract", "No output format - response shape left to the model"),
        ("PL002", "medium", "ambiguity", "Undecidable wording ('a few', 'appropriate')"),
        ("PL003", "high", "contradiction", "Instructions that cannot both be satisfied"),
        ("PL004", "high", "injection-risk", "Interpolated input with no data boundary"),
        ("PL005", "medium", "injection-risk", "No data-not-instructions guard"),
        ("PL006", "high", "unfinished", "TODO / placeholder left in the prompt"),
        ("PL007", "low", "framing", "No role or task framing up front"),
        ("PL008", "medium", "framing", "Mostly prohibitions, no positive instruction"),
        ("PL009", "low", "output-contract", "No length bound - cost and latency vary"),
        ("PL010", "medium", "grounding", "Classification task with no examples"),
        ("PL011", "medium", "grounding", "No fallback for the unanswerable case"),
        ("PL012", "low", "efficiency", "Politeness padding billed on every call"),
    ]
    st.dataframe(
        pd.DataFrame(ref, columns=["rule", "severity", "category", "what it catches"]),
        width="stretch", hide_index=True,
    )
    st.info(
        "Scoring: 100 minus 15 per high, 7 per medium, 3 per low, floored at 0. "
        "Wire `gate()` into CI to block a prompt change on score or high-severity count."
    )

st.divider()
st.caption(
    "Day 126 of Phoebe's daily FDE build - LLMOps & GenAI Platform line. "
    "Pairs with Day 84 (guardrails), Day 81 (prompt registry), Day 125 (cost estimator)."
)
