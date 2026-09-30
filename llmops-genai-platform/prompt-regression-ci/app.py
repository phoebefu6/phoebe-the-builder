"""Streamlit front end: read a prompt PR's behavioural diff against its noise floor.

Paste per-case results for the baseline prompt and the candidate, one line per case, k runs as a 0/1
string (e.g. `101`). The app lists the cases each rule calls broken, the cases whose baseline runs
disagree (flaky), and how many "broken" cases a no-op change would produce at these pass rates. The
study from results.json is on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import gate as G
import numpy as np
import streamlit as st

st.set_page_config(page_title="Prompt regression CI", layout="wide")
R = json.load(open("results.json"))


def _default(k: int = 3) -> tuple:
    pi_a = G.held_out_set()
    pi_b, _ = G.scenario("hard break", pi_a)
    rng = np.random.default_rng(11)
    a = (rng.random((G.N, k)) < pi_a[:, None]).astype(int)
    b = (rng.random((G.N, k)) < pi_b[:, None]).astype(int)
    return "\n".join("".join(map(str, r)) for r in a), "\n".join("".join(map(str, r)) for r in b)


def _parse(text: str) -> list:
    rows = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if any(set(r) - {"0", "1"} for r in rows):
        raise ValueError("each line must be a string of 0s and 1s, one character per run")
    if len({len(r) for r in rows}) > 1:
        raise ValueError("every case needs the same number of runs")
    return [[int(c) for c in r] for r in rows]


st.title("Did the prompt change break something, or is that the noise?")
st.markdown(
    "An LLM does not return the same answer twice, so a diff of two runs lists 'broken' cases even when "
    "nothing changed. Read the list against its noise floor before blocking a PR."
)
tab1, tab2 = st.tabs(["Your diff", "The study"])

with tab1:
    da, db = _default()
    c1, c2 = st.columns(2)
    raw_a = c1.text_area("Baseline prompt: one line per case, k runs (0/1)", da, height=240, key="base")
    raw_b = c2.text_area("Candidate prompt: same cases, same order, same k", db, height=240, key="cand")
    try:
        res = G.audit(_parse(raw_a), _parse(raw_b))
    except ValueError as err:
        st.warning(f"Cannot read this input: {err}")
        st.stop()

    k = res["k"]
    primary = "fisher" if k >= 4 else ("majority" if k % 2 and k >= 3 else "naive")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("cases x runs", f"{res['n']} x {k}")
    m2.metric("pass rate, baseline -> candidate", f"{res['pass_a']:.1%} -> {res['pass_b']:.1%}")
    m3.metric("naive diff (first run only)", f"{len(res['naive'])} broken")
    m4.metric(f"{primary} rule", f"{len(res[primary])} broken")

    if k == 1:
        st.info(f"ONE RUN PER CASE. The naive diff lists {len(res['naive'])} cases and there is no way to tell which "
                "are real: in the study a no-op PR listed 7.8 on average. Re-run with k >= 3.")
    elif res[primary]:
        st.error(f"LIKELY REAL BREAK in {len(res[primary])} case(s): {res[primary][:20]}. These fail under the "
                 f"candidate on most runs and pass under the baseline on most runs ({primary} rule).")
    elif res["naive"]:
        st.warning(f"DIFF IS NOISE. The naive diff lists {len(res['naive'])} cases; at these pass rates a no-op "
                   f"change is expected to list {res['noop_floor_naive']:.1f}, and none survive the {primary} rule. "
                   f"{len(res['flaky'])} cases disagree across their own baseline runs.")
    else:
        st.success("CLEAN. No case passed under the baseline and failed under the candidate.")

    st.caption(f"flaky in the baseline (runs disagree): {res['flaky'][:40]}")

with tab2:
    st.subheader("A no-op PR, by rule")
    st.dataframe([{k: r[k] for k in ("rule", "noop_expected_flags", "noop_red_at_T1", "T", "noop_red_at_T",
                                     "month_any_false_red_T", "calls_per_pr")} for r in R["rules"]], width="stretch")
    st.subheader("P(red) at the calibrated threshold")
    st.dataframe([{"rule": r["rule"], **{sc: round(r[sc]["red_at_T"], 3) for sc in G.SCENARIOS[1:]}}
                  for r in R["rules"]], width="stretch")
