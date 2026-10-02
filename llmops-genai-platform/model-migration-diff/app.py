"""Streamlit front end: read a model migration's diff for what the score change hides.

Paste one line per case: `intent,A,B` or `intent,A,B,A_rerun` (0/1 each, one run). The app shows the
score change, how many cases broke and how many were fixed, per-intent movement with an exact McNemar
(Bonferroni over intents), and - if you give a rerun of the old model - the exact sign test of churn
against the old model's own noise. The study from results.json is on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import migrate as M
import numpy as np
import streamlit as st

st.set_page_config(page_title="Model migration diff", layout="wide")
R = json.load(open("results.json"))


def _default() -> str:
    pa, intent, kind = M.eval_set()
    pb, _ = M.scenario("net zero, concentrated", pa, intent, kind)
    rng = np.random.default_rng(11)
    a, b, r = ((rng.random(M.N) < p).astype(int) for p in (pa, pb, pa))
    names = [n for n, _ in M.INTENTS]
    return "\n".join(f"{names[intent[i]]},{a[i]},{b[i]},{r[i]}" for i in range(M.N))


def _parse(text: str) -> tuple:
    rows, rerun = [], []
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    widths = {len(ln.split(",")) for ln in lines}
    if not lines or len(widths) > 1 or widths - {3, 4}:
        raise ValueError("every line must be intent,A,B or intent,A,B,A_rerun - all lines the same shape")
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if any(p not in ("0", "1") for p in parts[1:]):
            raise ValueError(f"results must be 0 or 1: {ln!r}")
        rows.append((parts[0], int(parts[1]), int(parts[2])))
        if len(parts) == 4:
            rerun.append(int(parts[3]))
    return rows, rerun


st.title("The score didn't move. What did?")
st.markdown(
    "A model migration can keep the same score while dozens of cases change behaviour: the cases it "
    "broke and the cases it fixed cancel in the total. Read the diff by case and by intent, and test the "
    "churn against the old model's own noise."
)
tab1, tab2 = st.tabs(["Your migration", "The study"])

with tab1:
    raw = st.text_area("One line per case: intent,A,B[,A_rerun]  (0/1, one run each)", _default(), height=240,
                       key="rows")
    try:
        res = M.audit(*_parse(raw))
    except ValueError as err:
        st.warning(f"Cannot read this input: {err}")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("cases", res["n"])
    m2.metric("score, A -> B", f"{res['score_a']:.1%} -> {res['score_b']:.1%}")
    m3.metric("broke / fixed", f"{res['broke']} / {res['fixed']}")
    m4.metric("McNemar p (whole set)", f"{res['mcnemar_p']:.3f}")

    hit = [r for r in res["intents"] if r["flag"]]
    churn_real = res["sign_p"] is not None and res["sign_p"] < M.ALPHA
    if hit:
        names = ", ".join(f"{r['intent']} {r['a']:.0%} -> {r['b']:.0%}" for r in hit)
        st.error(f"HIDDEN REGRESSION in {names}. The whole-set score moved {res['score_b'] - res['score_a']:+.1%}; "
                 "this intent broke on its own (exact McNemar, Bonferroni over intents).")
    elif res["mcnemar_p"] < M.ALPHA:
        st.error(f"SCORE MOVED: {res['broke']} broke vs {res['fixed']} fixed, McNemar p = {res['mcnemar_p']:.3f}.")
    elif churn_real:
        st.warning(f"BEHAVIOUR CHANGED, SCORE DID NOT. {res['churn']} cases differ between A and B against "
                   f"{res['noise_churn']} between A and its rerun (sign test p = {res['sign_p']:.4f}). Read the "
                   "flipped cases before calling this migration equivalent.")
    elif res["sign_p"] is None:
        st.info(f"NO RERUN GIVEN. {res['churn']} cases differ and nothing here can say how many of them are noise. "
                "Add a fourth column: model A run a second time.")
    else:
        st.success(f"NO DETECTABLE CHANGE. {res['churn']} cases differ, against {res['noise_churn']} between A and its "
                   "own rerun.")

    st.dataframe([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in res["intents"]],
                 width="stretch")

with tab2:
    st.subheader("P(gate fires) under each declared migration - exact")
    st.dataframe([{"migration": r["scenario"], "score": f"{r['score_a']:.1%} -> {r['score_b']:.1%}",
                   "cases changed": r["cases_changed"], **{g: round(r["gates"][g], 3) for g in M.GATES}}
                  for r in R["scenarios"]], width="stretch")
    st.subheader("A net-zero spread migration of growing size")
    st.dataframe([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in R["sweep"]],
                 width="stretch")
