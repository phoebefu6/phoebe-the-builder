"""Streamlit front end: what CUPED is worth on YOUR numbers, and what it is not.

Deliberately reports five things the usual CUPED explainer leaves out - that the
saving is rho squared, what theta = 1 would cost instead, whether your new-user
share has pushed mean-imputation past its break-even, the interval on a
correlation measured off a heavy tail, and the one covariate-timing rule that
turns the whole thing into an effect-destroying machine.
"""

from __future__ import annotations

import cuped
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CUPED variance", page_icon="📉", layout="wide")

st.title("CUPED: what the pre-period is worth")
st.caption(
    "Variance reduction is a bet on a correlation you already collected. Everything it can "
    "ever give you is that correlation squared - and two implementation details reverse its sign."
)

with st.sidebar:
    st.header("Your experiment")
    rho = st.slider("pre-period / in-experiment correlation", 0.0, 0.95, 0.60, 0.01,
                    help="Measure this on last quarter's data BEFORE promising a timeline.")
    per_arm = st.number_input("users per arm", 100, 10_000_000, 3_000, step=500)
    weeks = st.number_input("planned test length (weeks)", 1, 52, 6)
    new_share = st.slider("share of users with no pre-period", 0.0, 0.95, 0.0, 0.05)
    st.divider()
    st.caption("Optional: the pre-period window is usually WIDER than the test window, "
               "which is what makes theta = 1 dangerous.")
    sd_ratio = st.slider("sd(pre-window) / sd(test window)", 0.5, 3.0, 1.0, 0.1)

mult = cuped.sample_size_multiplier(rho)
c1, c2, c3, c4 = st.columns(4)
c1.metric("variance removed", f"{cuped.variance_reduction(rho):.1%}",
          help="Exactly rho squared. Nothing else.")
c2.metric("traffic needed", f"{mult:.2f}x")
c3.metric(f"{weeks} weeks becomes", f"{weeks * mult:.1f} weeks")
c4.metric("rho needed to halve it", f"{cuped.rho_for_saving(0.5):.3f}")

if rho < 0.5:
    st.info(
        f"At rho = {rho:.2f} CUPED returns {cuped.variance_reduction(rho):.0%} of the sample. "
        "That is real and free, and it is not a schedule change. 'CUPED halves your test' is a "
        f"statement about rho = {cuped.rho_for_saving(0.5):.3f}, not about CUPED."
    )

st.subheader("1. The saving is rho squared, not rho")
tab = []
for r in (0.2, 0.3, 0.4, 0.5, 0.6, 0.707, 0.8, 0.9):
    m = cuped.sample_size_multiplier(r)
    tab.append({"rho": f"{r:.3f}", "variance removed": f"{1 - m:.1%}",
                "traffic needed": f"{m:.3f}x",
                f"{weeks}-week test becomes": f"{weeks * m:.2f} weeks",
                "": "  <- you are here" if abs(r - rho) < 0.026 else ""})
st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch")

st.subheader("2. Fit the coefficient. Never assume it is 1.")
ratio = cuped.variance_ratio_unit_theta(rho, sd_ratio * 4.0, 4.0)
d1, d2, d3 = st.columns(3)
d1.metric("theta* (the fitted value)", f"{rho / sd_ratio:.4f}")
d2.metric("variance if you use theta = 1", f"{ratio:.3f}x",
          delta=f"{ratio - mult:+.3f} vs fitted", delta_color="inverse")
d3.metric("variance with fitted theta", f"{mult:.3f}x")
if ratio > 1.0:
    st.error(
        f"**At these settings, subtracting each user's pre-period value multiplies the variance "
        f"by {ratio:.2f}x** - the test needs {ratio:.1f}x the traffic instead of {mult:.2f}x. "
        f"theta = 1 hurts whenever sd(pre) > 2 x rho x sd(post), i.e. whenever the sd ratio "
        f"exceeds {2 * rho:.2f}. A month of history against a week of test is the normal case."
    )
else:
    st.success(
        f"theta = 1 happens to be safe here ({ratio:.3f}x), because the sd ratio {sd_ratio:.1f} "
        f"is below 2 x rho = {2 * rho:.2f}. It stops being safe as soon as the pre-window widens, "
        "and fitting theta is never worse."
    )

st.subheader("3. New users: stratify, do not impute")
imp = cuped.reduction_mean_impute(rho, new_share)
strat = cuped.reduction_stratified(rho, new_share)
e1, e2, e3 = st.columns(3)
e1.metric("mean-impute the covariate", f"{imp:+.3f}")
e2.metric("stratify on 'has a pre-period'", f"{strat:+.3f}")
e3.metric("break-even for imputation", f"{cuped.impute_breakeven_share():.0%} new users")
if new_share > 0.5:
    st.error(
        f"**{new_share:.0%} of users have no pre-period, so mean-imputation is a variance "
        f"INCREASE of {-imp:.1%}** - worse than not adjusting at all. Stratifying still returns "
        f"{strat:.1%}. The per-user variance does fall by (1-f)rho^2, which is what every "
        "write-up quotes, but the estimator is a difference of arm MEANS and the imputed arm "
        "mean is the mean of the returning users only: its variance is sigma^2/(n(1-f)), not "
        "sigma^2/n. That leaves rho^2(2 - 1/(1-f)), which is zero at f = 0.5 for any rho."
    )
elif new_share > 0.2:
    st.warning(
        f"At {new_share:.0%} new users, mean-imputation delivers {imp:.1%} where stratifying "
        f"delivers {strat:.1%} - you are giving up {strat - imp:.1%} for three lines of code."
    )

st.subheader("4. Is your metric heavy-tailed?")
st.write(
    "CUPED runs on the Pearson correlation of the metric **as reported**, not of its log. On a "
    "revenue-shaped metric a strong log-scale relationship is a much weaker reported one, and "
    "the sample correlation - the number you would compute to plan with - is biased upward and "
    "unstable."
)
log_rho = st.slider("correlation on the LOG scale", 0.1, 0.95, 0.80, 0.05)
rows = []
for sig in (0.5, 1.0, 1.5, 2.0):
    pr = cuped.lognormal_pearson_rho(log_rho, sig)
    rows.append({"tail weight (lognormal sigma)": sig,
                 "correlation on the reported scale": f"{pr:.4f}",
                 "variance removed": f"{pr * pr:.1%}",
                 "vs the log-scale promise": f"{log_rho ** 2:.1%}"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(
    "Measured on this build: at sigma 2.0 the sample correlation reads 0.5315 +/- 0.126 against "
    "a population 0.4391 - biased up 21%, and wide enough that two honest analysts on the same "
    "table quote correlations 0.3 apart. Cap or winsorise first, and quote an interval."
)

st.subheader("5. The covariate must predate assignment")
st.error(
    "**This is the one that is not a statistics question.** A covariate the treatment also "
    "moved - same-period engagement, a metric the variant changed - makes the adjustment remove "
    "the treatment effect as if it were noise. Measured on this build: the estimate came out "
    "**59% low**, coverage fell to **0.70** against a nominal 0.95, and power fell from 0.51 to "
    "0.17. The diff is three lines and reviews clean. Put a hard cutoff at the assignment "
    "timestamp on the covariate table, and make it visible to a reviewer."
)

with st.expander("Run it on your own two columns"):
    st.write(
        "Paste `pre,post,arm` per user (arm = 0 control, 1 treatment). The app fits theta on the "
        "arms pooled, reports the adjusted and unadjusted effect, and shows the reduction it "
        "actually delivered next to the rho^2 it promised."
    )
    default = "\n".join(
        f"{p:.3f},{q:.3f},{a}" for p, q, a in
        (lambda rr: [(x, y, i % 2) for i, (x, y) in enumerate(
            zip(rr.normal(10, 4, 40), rr.normal(10, 4, 40)))])(np.random.default_rng(3))
    )
    raw = st.text_area("pre,post,arm", default, height=150)
    pre, post, arm = [], [], []
    for line in raw.strip().splitlines():
        parts = line.split(",")
        if len(parts) != 3:
            continue
        try:
            pre.append(float(parts[0]))
            post.append(float(parts[1]))
            arm.append(int(parts[2]))
        except ValueError:
            continue
    pre, post, arm = np.array(pre), np.array(post), np.array(arm)
    if len(pre) > 6 and set(arm.tolist()) == {0, 1}:
        d = {"pre_c": pre[arm == 0][None, :], "post_c": post[arm == 0][None, :],
             "pre_t": pre[arm == 1][None, :], "post_t": post[arm == 1][None, :],
             "new_c": np.zeros((1, (arm == 0).sum()), bool),
             "new_t": np.zeros((1, (arm == 1).sum()), bool)}
        n = min(d["pre_c"].shape[1], d["pre_t"].shape[1])
        d = {k: v[:, :n] for k, v in d.items()}
        e0, s0 = cuped.adj_none(d)
        e1, s1 = cuped.adj_cuped(d)
        seen_rho = float(np.corrcoef(pre, post)[0, 1])
        f1, f2, f3 = st.columns(3)
        f1.metric("unadjusted effect", f"{e0[0]:.4f}", delta=f"se {s0[0]:.4f}")
        f2.metric("CUPED effect", f"{e1[0]:.4f}", delta=f"se {s1[0]:.4f}")
        f3.metric("correlation in your data", f"{seen_rho:.4f}",
                  delta=f"promises {seen_rho ** 2:.1%}")
        st.caption(
            f"Standard error {s0[0]:.4f} -> {s1[0]:.4f}, a variance reduction of "
            f"{1 - (s1[0] / s0[0]) ** 2:.1%} against the {seen_rho ** 2:.1%} that this "
            "correlation promises. On a handful of rows the two will not match; that gap is the "
            "reason to report both."
        )
    else:
        st.info("Need at least 7 rows with both arms present.")

st.divider()
st.caption(
    "Day 166 of the FDE portfolio. Full measured argument: `python evidence.py`; assertions in "
    "`test_cuped.py`; figure from `make_chart.py`. "
    "Reference: Deng, Xu, Kohavi & Walker (2013), WSDM."
)
