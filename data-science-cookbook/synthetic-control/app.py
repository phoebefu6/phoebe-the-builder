"""Pick a failure mode, watch the pre-period plot stay convincing while the answer moves.

Every panel here is a live refit - nothing is read from results.json - so the app is a way of
poking at the estimator rather than a slideshow of the evidence run.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from synth import fit_synth, hull_excess, make_panel, placebo_pvalue, pvalue_floor  # noqa: E402

INK, BAD, COOL, GOOD, GRID, WARN = "#16222e", "#b3402f", "#2b6ca3", "#1f7a5c", "#dfe5ea", "#c98a1a"

MODES = {
    "Clean - the assumptions hold": {},
    "Outside the convex hull (the treated market is above its whole pool)": {"hull_shift": 5.0},
    "A donor was treated too (spillover into the control group)": {"contaminated": True},
    "Anticipation (the market moved before the official date)": {"anticipation": 4},
}


def build(mode: str, n_donors: int, t_pre: int, effect: float, seed: int) -> np.ndarray:
    kw = dict(MODES[mode])
    contaminate = kw.pop("contaminated", False)
    rng = np.random.default_rng(seed)
    Y = make_panel(rng, n_donors=n_donors, t_pre=t_pre, effect=effect, **kw)
    if contaminate:
        top = [int(i) for i in np.argsort(fit_synth(Y, 0, t_pre).weights)[::-1][:3]]
        Y = make_panel(
            np.random.default_rng(seed),
            n_donors=n_donors,
            t_pre=t_pre,
            effect=effect,
            contaminated={i: 1.0 for i in top},
        )
    return Y


def main() -> None:
    st.set_page_config(page_title="Synthetic control", layout="wide")
    st.title("One market, one intervention")
    st.caption(
        "There is no control group, so one gets fitted from a weighted average of the other "
        "markets. The fit is always convincing. The estimate is not always right, and the "
        "first thing does not tell you about the second."
    )

    with st.sidebar:
        mode = st.selectbox("What is wrong with this study", list(MODES))
        n_donors = st.slider("donors in the pool", 5, 60, 20)
        t_pre = st.slider("pre-periods of history", 8, 120, 30)
        effect = st.slider("true effect", 0.0, 10.0, 5.0, 0.5)
        seed = st.number_input("seed", 0, 9999, 1000, step=1)

    Y = build(mode, n_donors, t_pre, float(effect), int(seed))
    f = fit_synth(Y, 0, t_pre)
    synthetic = Y[0] - f.gaps
    p, stats = placebo_pvalue(Y, 0, t_pre)
    excess = hull_excess(Y, 0, t_pre)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("true effect", f"{effect:.2f}")
    c2.metric("estimate", f"{f.att:.2f}", f"{f.att - effect:+.2f}")
    c3.metric("pre-period RMSPE", f"{f.pre_rmspe:.3f}", "the credential")
    c4.metric("placebo p", f"{p:.3f}", f"floor {pvalue_floor(n_donors):.3f}", delta_color="off")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))
    t = np.arange(Y.shape[1])
    for row in Y[1:]:
        ax1.plot(t, row, color=GRID, lw=0.7)
    ax1.plot(t, Y[0], color=BAD, lw=2.3, label="treated")
    ax1.plot(t, synthetic, color=COOL, lw=2.0, ls="--", label="synthetic")
    ax1.axvline(t_pre - 0.5, color=INK, lw=1.1)
    ax1.set_title("the plot that gets published", fontsize=11, loc="left")
    ax1.legend(fontsize=8, frameon=False)

    for i in range(1, Y.shape[0]):
        ax2.plot(t, fit_synth(Y, i, t_pre).gaps, color=GRID, lw=0.8)
    ax2.plot(t, f.gaps, color=BAD, lw=2.3)
    ax2.axhline(0, color=INK, lw=0.9)
    ax2.axvline(t_pre - 0.5, color=INK, lw=1.1)
    ax2.set_title("the same fit against every placebo (grey)", fontsize=11, loc="left")
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("What the diagnostics can and cannot see")
    st.write(
        f"- **Hull excess** (post period): `{excess['post']:.3f}`. The part of the treated path "
        "lying outside the donor range. Anything above zero is bias the weights cannot remove, "
        "and it is computable without knowing the truth."
    )
    st.write(
        f"- **Pre-period RMSPE**: `{f.pre_rmspe:.3f}`. Only anticipation moves this. Spillover "
        "into a donor is a post-period event, so this stays perfect while the estimate is wrong."
    )
    st.write(
        f"- **p-value floor**: `{pvalue_floor(n_donors):.3f}` with {n_donors} donors. "
        + (
            "A 0.05 result is unreachable at this pool size, at any effect size."
            if pvalue_floor(n_donors) > 0.05
            else "0.05 is reachable at this pool size."
        )
    )
    st.caption(
        f"Test statistic: post/pre MSPE ratio, {f.ratio:.1f} for the treated unit against a "
        f"placebo median of {np.median(np.delete(stats, 0)):.1f}."
    )


if __name__ == "__main__":
    main()
