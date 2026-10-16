"""Streamlit front end: describe how your training set was assembled and see the per-segment audit and its cost.

Override any training-set cell share as `cell=share` (shares must sum to 1). The audit checks the marginals AND the
cells; the model is refitted on that training set and scored on the deployment population. One seed, so a run takes
seconds; the full three-seed study is on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import audit as A
import streamlit as st

st.set_page_config(page_title="Sampling bias audit", layout="wide")

st.title("The training set is not the population")
tab_use, tab_study = st.tabs(["Your training set", "The study"])

with tab_use:
    text = st.text_input(
        "Training-set cell shares (cell=share, comma separated)",
        "",
        key="shares",
        help="Cells: "
        + ", ".join(A.CELLS)
        + ". Default: "
        + ", ".join(f"{k}={v}" for k, v in A.QUOTA.items()),
    )
    try:
        q = A.parse_shares(text)
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    r = A.one_seed(A.SEEDS[0], quota=q, n_pop=20_000, with_curve=False)
    au = r["audit"]
    flagged = [k for k, v in au.items() if v["flag"]]
    marg = [k for k in flagged if au[k]["kind"] == "marginal"]
    if flagged and not marg:
        st.warning(
            f"The one-attribute audit passes, and {len(flagged)} cells are flagged: {', '.join(flagged)}."
        )
    elif flagged:
        st.warning(f"Flagged: {', '.join(flagged)}.")
    else:
        st.success("Every marginal and every cell is inside [0.80, 1.25] of the population share.")
    gq, gp = r["quota"]["B-rural"]["good_rejected"], r["proportional"]["B-rural"]["good_rejected"]
    st.caption(
        f"Rural B good payers rejected: {gq:.1%} trained on your set, {gp:.1%} trained on a proportional "
        f"sample. Overall accuracy on your own test split {r['quota_on_own_test']['overall']['acc']:.1%}, "
        f"in deployment {r['quota']['overall']['acc']:.1%}."
    )
    st.table(
        [
            {
                "segment": k,
                "kind": v["kind"],
                "rows": v["rows"],
                "train": f"{v['train_share']:.1%}",
                "population": f"{v['pop_share']:.1%}",
                "ratio": f"{v['ratio']:.2f}",
                "flag": "FLAG" if v["flag"] else "ok",
            }
            for k, v in au.items()
        ]
    )

with tab_study:
    st.image("audit.png")
    st.code(open("evidence.txt").read())
