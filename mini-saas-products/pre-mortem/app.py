"""Score a real risk register, and watch the orderings disagree.

    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import premortem as P
import streamlit as st

st.set_page_config(page_title="Pre-mortem", layout="wide")

st.title("A pre-mortem produces a risk model")
st.caption(
    "The exercise is cheap and it works. What follows it - scoring the output on a 5x5 "
    "matrix - cannot rank the risks it summarises."
)

tab_plan, tab_reg, tab_matrix = st.tabs(
    ["The plan's own odds", "Your register", "Why not a matrix"]
)

# --------------------------------------------------------------------------
with tab_plan:
    st.markdown(
        "One confidence per step. Nobody states the plan's number, because no one "
        "person owns it."
    )
    n_steps = st.slider("Steps in the plan", 3, 30, len(P.PLAN))
    p_step = st.slider("Confidence per step", 0.70, 0.999, 0.95, step=0.005)
    shock = st.slider("Chance of one common cause hitting everything", 0.0, 0.5, 0.12,
                      step=0.01)

    indep = p_step ** n_steps
    steps = tuple(P.Step(f"step {i + 1}", p_step) for i in range(n_steps))
    corr = P.correlated_plan_success(rho_shock=shock, steps=steps, n=120_000)

    c1, c2, c3 = st.columns(3)
    c1.metric("Weakest single step", f"{p_step:.3f}")
    c2.metric("Plan, independent steps", f"{indep:.3f}")
    c3.metric("Plan, with a common cause", f"{corr['correlated']:.3f}",
              delta=f"{corr['correlated'] - indep:+.3f}")

    running = np.cumprod([p_step] * n_steps)
    st.line_chart(
        pd.DataFrame({"P(everything so far worked)": running},
                     index=range(1, n_steps + 1)),
        height=260,
    )
    if indep < 0.5:
        st.error(
            f"**{n_steps} steps at {p_step:.0%} succeeds {indep:.0%} of the time.** Not one "
            "step looks alarming. The product does, and independence is the optimistic "
            "assumption on top of it."
        )
    else:
        st.info(
            f"{n_steps} steps at {p_step:.0%} gives {indep:.0%}. Add steps or shave the "
            "per-step confidence and watch how fast this falls."
        )

# --------------------------------------------------------------------------
with tab_reg:
    st.markdown(
        "Four numbers per cause. The last two are what most registers are missing, and "
        "they are the only two that decide anything."
    )
    default = pd.DataFrame(
        [{"id": m.id, "cause": m.cause, "probability": m.probability, "loss": m.loss,
          "prevention_cost": m.prevention_cost, "prevention_effect": m.prevention_effect}
         for m in P.MODES]
    )
    edited = st.data_editor(default, num_rows="dynamic", use_container_width=True,
                           height=380, key="register")

    modes = []
    for _i, row in edited.iterrows():
        try:
            m = P.FailureMode(
                str(row["id"]), str(row["cause"]), float(row["probability"]),
                float(row["loss"]), float(row["prevention_cost"]),
                float(row["prevention_effect"]),
            )
        except (TypeError, ValueError):
            continue
        if 0 < m.probability < 1 and m.loss > 0 and 0 < m.prevention_effect <= 1:
            modes.append(m)

    if len(modes) < 2:
        st.info("Give it at least two complete rows.")
    else:
        scale = P.SCALES_BY_NAME[
            st.selectbox("Risk scale", [s.name for s in P.SCALES],
                         help="Both are the shape found in real corporate templates.")
        ]
        q = P.ranking_quality(scale, modes)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total expected loss", f"{P.total_expected_loss(modes):,.0f}")
        c2.metric("Pairs ranked backwards", q["inversions"],
                  help="Matrix says A is bigger; A's expected loss is smaller.")
        c3.metric("Inversion rate", f"{q['inversion_rate']:.0%}")
        c4.metric("Pairs it cannot order", f"{q['undecided_rate']:.0%}")

        orders = {
            "matrix score": [m.id for m in P.by_matrix(scale, modes)],
            "expected loss": [m.id for m in P.by_expected_loss(modes)],
            "prevention value": [m.id for m in P.by_prevention_value(modes)],
            "avoided per unit spent": [m.id for m in P.by_prevention_ratio(modes)],
        }
        st.subheader("Four orderings of the same register")
        st.dataframe(pd.DataFrame(orders, index=range(1, len(modes) + 1)),
                     use_container_width=True)

        if orders["matrix score"] != orders["prevention value"]:
            m_top, v_top = orders["matrix score"][0], orders["prevention value"][0]
            st.warning(
                f"The matrix puts **{m_top}** first. Once cost of prevention is known the "
                f"answer is **{v_top}**. Only the second ordering asked what fixing it costs."
            )

        st.subheader("Spend a budget")
        budget = st.number_input("Prevention budget", min_value=0.0,
                                 value=100_000.0, step=10_000.0)
        greedy = P.budget_allocation(budget, modes)
        exact = P.optimal_allocation(budget, modes) if len(modes) <= 18 else None
        cols = st.columns(3)
        cols[0].metric("Matrix order avoids", f"{greedy['matrix_avoided']:,.0f}")
        cols[1].metric("Ratio order avoids", f"{greedy['ratio_avoided']:,.0f}")
        if exact:
            cols[2].metric("Exact best set avoids", f"{exact['avoided']:,.0f}")
            short = exact["avoided"] - greedy["matrix_avoided"]
            if short > 0:
                st.error(
                    f"Working down the matrix leaves **{short:,.0f}** of achievable loss "
                    f"avoidance unbought - {short / exact['avoided']:.0%} of what this budget "
                    f"could have prevented. Choosing a set under a budget is a knapsack, not "
                    f"a sort, and no ordering is guaranteed to find it."
                )
            st.caption(f"Exact best set: {', '.join(exact['bought'])}")
        else:
            cols[2].metric("Exact best set", "too many rows",
                           help="Brute force is exact up to 18 rows.")

# --------------------------------------------------------------------------
with tab_matrix:
    scale = P.SCALES[0]
    st.markdown(
        f"""
Cox (2008) showed qualitative risk matrices cannot reproduce the ordering of the
quantitative risks they summarise. On the reference register the default 5x5 inverts
**{P.ranking_quality(scale)['inversion_rate']:.0%}** of the pairs it orders and cannot order
**{P.ranking_quality(scale)['undecided_rate']:.0%}** of them at all.

Three separate failures, all visible on one register:
"""
    )
    o = P.ordinal_product_is_meaningless(scale)
    c = P.range_compression(scale)
    d = P.scale_disagreement()
    st.markdown(
        f"""
**1. The arithmetic has no unit.** Band 4 is not twice band 2. {o['cells']} cells collapse
to {o['distinct_scores']} distinct scores, and {o['colliding_scores']} of those scores are
shared by more than one cell. Score 12 comes from {o['example'][1]} - a 30% chance of a
2,000,000 loss and a 60% chance of a 500,000 loss score identically, and their expected
losses differ by a factor of two.

**2. Ranges collapse.** {c['shared_cells']} cells hold more than one risk. In the worst,
{c['worst_pair'][0]} and {c['worst_pair'][1]} score identically while one carries
{c['worst_ratio']:.1f}x the expected loss of the other. Every reader downstream treats them
as equivalent.

**3. The bin edges decide the answer.** Two conventional scales disagree on
{d['n_flips']} pairs, and they name different top risks - {d['top_by_a']} against
{d['top_by_b']}. The most consequential output of the exercise depends on which template
the organisation downloaded.
"""
    )
    st.subheader("The reference register, both scales")
    a, b = P.SCALES
    st.dataframe(
        pd.DataFrame(
            {
                "expected loss": [m.expected_loss for m in P.by_expected_loss()],
                f"{a.name} cell": [str(a.cell(m)) for m in P.by_expected_loss()],
                f"{a.name} score": [a.score(m) for m in P.by_expected_loss()],
                f"{b.name} cell": [str(b.cell(m)) for m in P.by_expected_loss()],
                f"{b.name} score": [b.score(m) for m in P.by_expected_loss()],
            },
            index=[m.id for m in P.by_expected_loss()],
        ),
        use_container_width=True,
    )
    st.caption(
        "Sorted by true expected loss, descending. Neither score column is monotone in it."
    )
