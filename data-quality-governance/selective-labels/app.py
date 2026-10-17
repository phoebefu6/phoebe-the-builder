"""Streamlit front end: change the world the past decisions came from and see what selective labels do.

Override any world setting as `key=value` (sel_u = how much officers leaned on the unrecorded interview read,
bar_b = the extra bar group B had to clear). One seed on a smaller population so a run takes seconds; the full
three-seed study is on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import selective as S
import streamlit as st

st.set_page_config(page_title="Selective labels", layout="wide")

st.title("We only know outcomes for the people we approved")
tab_use, tab_study = st.tabs(["Your world", "The study"])

with tab_use:
    text = st.text_input(
        "World settings (key=value, comma separated)",
        "",
        key="world",
        help="Keys: " + ", ".join(f"{k} (default {v})" for k, v in S.WORLD.items()),
    )
    try:
        w = S.parse_world(text)
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    r = S.one_seed(S.SEEDS[0], w, n=20_000)
    o, lm = r["eval_full_model"], r["eval_lab_model"]
    gap_ipw = abs(o["ipw"]["repay"] - o["truth"]["repay"])
    if lm["labeled"]["repay"] > o["labeled"]["repay"] and lm["truth"]["repay"] < o["truth"]["repay"]:
        st.warning(
            f"The labeled rows rank the worse model first: {lm['labeled']['repay']:.1%} vs {o['labeled']['repay']:.1%}, "
            f"while the truth is {lm['truth']['repay']:.1%} vs {o['truth']['repay']:.1%}."
        )
    elif gap_ipw < 0.01:
        st.success(
            f"Selection runs on recorded data only: IPW recovers the oracle's repay rate within {gap_ipw * 100:.1f} pts."
        )
    else:
        st.info(f"IPW is off by {gap_ipw * 100:.1f} pts: the officers used something the data does not hold.")
    st.caption(
        f"Repay rate, all applicants {r['pop_repay']:.1%}; labeled rows {r['hist_repay_labeled']:.1%}. Share of B "
        f"approved: label-trained model {r['share_b_approved']['labeled']:.1%}, oracle {r['share_b_approved']['oracle']:.1%}."
    )
    st.table(
        [
            {
                "estimator": k,
                "oracle repay": f"{o[k]['repay']:.1%}",
                "label-trained repay": f"{lm[k]['repay']:.1%}",
                "oracle TPR gap A-B": f"{o[k]['tpr_gap'] * 100:+.1f} pts",
            }
            for k in ("truth", "labeled", "rejected_bad", "ipw", "explore_0.05")
        ]
    )

with tab_study:
    st.image("audit.png")
    st.code(open("evidence.txt").read())
