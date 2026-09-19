"""Streamlit front end: set your own design, and it measures what the Shapiro gate would cost you.

The point of the UI is that you do not have to trust the tables in the README. Pick a population
shape and a sample size, and the Monte-Carlo runs live under a null that is exactly true - so
every rejection it counts is a false positive, with no interpretation needed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import normality as N
import streamlit as st

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
SHAPIRO, TTEST, GATED, WILCOXON = "#c8562b", "#2f6fdb", "#7a4fa3", "#9aa4b0"

st.set_page_config(page_title="The normality-test trap", layout="wide")
st.title("The normality-test trap")
st.caption(
    "Shapiro-Wilk says your data is not normal. Should you abandon the t-test? "
    "Set your design below and find out from a simulation where the null is exactly true - "
    "so every rejection counted is a false positive, by construction."
)

with st.sidebar:
    st.header("Your design")
    dist = st.selectbox("Population shape", N.DISTRIBUTIONS, index=4, key="dist")
    skew, kurt = N.SHAPE[dist]
    st.caption(f"skew {skew:.2f} | excess kurtosis {kurt:.2f}")
    n = st.select_slider("Sample size n", options=list(N.GRID_N), value=50, key="n")
    alpha = st.select_slider("alpha", options=[0.01, 0.025, 0.05, 0.10], value=0.05, key="alpha")
    reps = st.select_slider(
        "Monte-Carlo replicates", options=[2_000, 5_000, 10_000, 20_000], value=20_000, key="reps"
    )
    st.caption(
        "20,000 is what the full study runs. Below that the interval on a measured rate gets "
        "wider than the band being judged, and the honest verdict becomes 'cannot tell'."
    )
    st.markdown("---")
    st.caption(
        "Every population is standardised to mean 0 and sd 1, so the only thing that changes "
        "between shapes is skew and tail weight."
    )

@st.cache_data(show_spinner="Running the simulation...")
def measure(dist: str, n: int, reps: int, alpha: float) -> N.CellResult:
    """Cached so dragging one control does not re-run every other cell in the sidebar."""
    return N.run_cell(dist, n, reps, alpha, seed=0)


cell = measure(dist, int(n), int(reps), float(alpha))
ci_lo, ci_hi = cell.t_error_interval
# Three outcomes, not two. "Not shown to be broken" is not the same claim as "works", and at a
# low replicate count it is the only claim available - the band is 0.010 wide and the interval
# can be wider than that.
clean = ci_lo >= N.BAND_LO and ci_hi <= N.BAND_HI
undecided = not clean and not cell.t_is_broken

c1, c2, c3, c4 = st.columns(4)
c1.metric("Shapiro rejects normality", f"{cell.shapiro_reject_rate:.1%}")
c2.metric(
    "t-test false positives",
    f"{cell.t_error:.4f}",
    delta=f"99% CI [{ci_lo:.4f}, {ci_hi:.4f}]",
    delta_color="off",
)
c3.metric("Wilcoxon false positives", f"{cell.wilcoxon_error:.4f}")
c4.metric(
    "Shapiro-gated procedure",
    f"{cell.conditional_error:.4f}",
    delta=f"{cell.conditional_error - cell.t_error:+.4f} vs t alone",
    delta_color="inverse",
)

fires = cell.shapiro_reject_rate >= 0.50
if undecided:
    st.info(
        f"**Cannot tell at {int(reps):,} replicates.** The t-test measured {cell.t_error:.4f}, but "
        f"its 99% interval is [{ci_lo:.4f}, {ci_hi:.4f}], which still overlaps the "
        f"[{N.BAND_LO}, {N.BAND_HI}] band. That is absence of evidence, not evidence of absence - "
        f"raise the replicate count to decide. (Shapiro rejects {cell.shapiro_reject_rate:.0%} "
        f"here either way, and the gated procedure runs at {cell.conditional_error:.4f}.)"
    )
elif fires and clean:
    st.error(
        f"**FALSE ALARM.** Shapiro rejects {cell.shapiro_reject_rate:.0%} of samples from this "
        f"population at n={n}, while the t-test's whole 99% interval [{ci_lo:.4f}, {ci_hi:.4f}] "
        f"sits inside the band it is entitled to. The gate is firing on a test that works. "
        f"Following it swaps a {cell.t_error:.4f} procedure for a {cell.conditional_error:.4f} one."
    )
elif fires and cell.t_is_broken:
    st.warning(
        f"**The gate is right here.** The t-test really is running at {cell.t_error:.4f} "
        f"against a nominal {alpha}. But check the tails below before trusting the fix - and "
        f"note the gated procedure lands at {cell.conditional_error:.4f}."
    )
elif not fires and cell.t_is_broken:
    st.warning(
        f"**BLIND SPOT.** Shapiro only rejects {cell.shapiro_reject_rate:.0%} of samples here, "
        f"so it would most likely have waved this through - while the t-test is actually "
        f"running at {cell.t_error:.4f} against a nominal {alpha}."
    )
else:
    st.success(
        f"Shapiro is quiet ({cell.shapiro_reject_rate:.0%} rejection) and the t-test's whole "
        f"interval [{ci_lo:.4f}, {ci_hi:.4f}] sits inside its band. Agreement - which is what "
        f"makes the other three corners worth knowing about."
    )

left, right = st.columns(2)

with left:
    st.subheader("What each procedure's error rate actually is")
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=BG)
    ax.set_facecolor(BG)
    names = ["t-test alone", "Wilcoxon alone", "Shapiro-gated"]
    vals = [cell.t_error, cell.wilcoxon_error, cell.conditional_error]
    ax.barh(names, vals, color=[TTEST, WILCOXON, GATED], height=0.55)
    ax.axvline(alpha, color=INK, ls="--", lw=1.8)
    for i, v in enumerate(vals):
        ax.text(max(v, 1e-3) * 1.08, i, f"{v:.4f}", va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(max(min(vals) * 0.5, 1e-3), 3.0)
    ax.set_xlabel("false-positive rate under a TRUE null (log scale)", color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, lw=0.7, axis="x")
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    st.pyplot(fig)
    st.caption(
        "The dashed line is what you asked for. Anything to the right of it is error you are "
        "paying for without having chosen to."
    )

with right:
    st.subheader("Which way the t-test goes wrong")
    fig2, ax2 = plt.subplots(figsize=(6.4, 3.4), facecolor=BG)
    ax2.set_facecolor(BG)
    ax2.bar(
        ["rejected LOW", "rejected HIGH"],
        [cell.t_error_lower_tail, cell.t_error_upper_tail],
        color=[TTEST, SHAPIRO],
        width=0.5,
    )
    ax2.axhline(alpha / 2, color=INK, ls="--", lw=1.8)
    ax2.text(1.35, alpha / 2 * 1.05, f"{alpha / 2:.3f} each side", color=INK, fontsize=8, ha="right")
    ax2.set_ylabel("one-sided false-positive rate", color=MUTED, fontsize=9)
    ax2.grid(True, color=GRID, lw=0.7, axis="y")
    ax2.set_axisbelow(True)
    for s in ax2.spines.values():
        s.set_color(GRID)
    st.pyplot(fig2)
    st.caption(
        f"Asymmetry {cell.tail_asymmetry:.1f}x. A skewed population makes the t-test reject in "
        "one direction far more than the other, and the two-sided p-value never shows it."
    )

st.markdown("---")
st.subheader("Conditioning on what Shapiro said")
a, b = st.columns(2)
a.metric("t-test error on the samples Shapiro PASSED", f"{cell.t_error_given_shapiro_passed:.4f}")
b.metric("t-test error on the samples Shapiro REJECTED", f"{cell.t_error_given_shapiro_rejected:.4f}")
st.caption(
    "Both columns come from the same simulated samples. If the gate carried real information "
    "about whether the t-test is trustworthy on THIS sample, these two numbers would differ. "
    "Where they do not, the gate is sorting samples on a property the t-test does not depend on."
)
