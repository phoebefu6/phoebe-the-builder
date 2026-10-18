"""Streamlit front end: price a random-approval slice for your own loan economics.

Set the margin on a repaid loan, the loss on a default and the confidence you need; the app reads the study
(results.json) and names the cheapest uniform and targeted slice that ranks the two challengers right at every seed,
rescaled to your economics. The figure and evidence are on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import explore as E
import streamlit as st

st.set_page_config(page_title="Exploration budget", layout="wide")
st.title("What does a random-approval slice cost, and what does it buy?")
R = json.load(open("results.json"))["main"]
tab_use, tab_study = st.tabs(["Your economics", "The study"])


def rescale(cost: float, gain: float, loss: float, repay: float) -> float:
    """The study's net cost was computed at E.GAIN / E.LOSS; rescale per extra loan at that pool's repay rate."""
    per_old = E.LOSS * (1 - repay) - E.GAIN * repay
    per_new = loss * (1 - repay) - gain * repay
    return cost / per_old * per_new if per_old else float("nan")


with tab_use:
    c1, c2, c3 = st.columns(3)
    gain = c1.number_input("Margin on a repaid loan ($)", 0.0, 100_000.0, E.GAIN, key="gain")
    loss = c2.number_input("Loss on a default ($)", 0.0, 1_000_000.0, E.LOSS, key="loss")
    target = c3.slider("Required chance of ranking right (worst seed)", 0.80, 0.99, 0.95, key="target")
    best = {d: E.cheapest(R, d, target) for d in ("uniform", "targeted")}
    pools = {"uniform": R["rejected_repay"]["mean"], "targeted": R["challenger_only_repay"]["mean"]}
    cost = {d: rescale(b["net_cost"], gain, loss, pools[d]) if b else None for d, b in best.items()}
    if cost["targeted"] is not None and cost["targeted"] <= 0:
        st.success(
            "At these economics the challenger-only loans make money: targeted exploration pays for itself."
        )
    elif cost["uniform"] and cost["targeted"]:
        st.warning(
            f"Targeted {best['targeted']['share']:.1%} costs ${cost['targeted']:,.0f} per {E.PER:,} applicants; "
            f"uniform needs {best['uniform']['share']:.1%} at ${cost['uniform']:,.0f}. Targeted cannot audit group B's TPR "
            f"({R['targeted_0.05']['tpr_b_bias']['mean'] * 100:+.1f} pts bias); keep a small uniform slice for that."
        )
    else:  # guard only: both designs reach 100% at the top of the study grid
        st.info(
            "No design on the grid reaches that confidence at every seed; widen the grid or relax the target."
        )
    st.table(
        [
            {
                "design": f"{d} {s:.2%}",
                "extra loans": f"{R[f'{d}_{s:g}']['extra_loans']['mean']:,.0f}",
                "net cost (your economics)": f"${rescale(R[f'{d}_{s:g}']['net_cost']['mean'], gain, loss, pools[d]):,.0f}",
                "P right (worst seed)": f"{R[f'{d}_{s:g}']['p_right']['lo']:.1%}",
            }
            for d, grid in (("uniform", E.UNIFORM), ("targeted", E.TARGETED))
            for s in grid
        ]
    )

with tab_study:
    st.image("audit.png")
    st.code(open("evidence.txt").read())
