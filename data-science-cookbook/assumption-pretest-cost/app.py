"""Streamlit front end: describe your design, and it measures what the pretest would cost you.

The point of the UI is that you do not have to trust the tables in the README. Set your own group
sizes, your own variance gap and your own pretest, and the Monte-Carlo runs live under a null that
is exactly true - so every rejection it counts is a false positive, with no interpretation needed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pretest as P
import streamlit as st

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
STUDENT, WELCH, GATED, PRETEST = "#c8562b", "#2f6fdb", "#7a4fa3", "#9aa4b0"

st.set_page_config(page_title="What the variance pretest costs", layout="wide")
st.title("What the variance pretest costs")
st.caption(
    "Run Levene, then pick Student's or Welch's. It is one procedure, not two, and it has its "
    "own error rate. Set your design and find out from a simulation where the null is exactly "
    "true - so every rejection counted is a false positive, by construction."
)

with st.sidebar:
    st.header("Your design")
    n1 = st.number_input("Group 1 size", min_value=3, max_value=2000, value=50, step=1, key="n1")
    n2 = st.number_input("Group 2 size", min_value=3, max_value=2000, value=10, step=1, key="n2")
    sd_ratio = st.slider("Group 2's sd (group 1 fixed at 1.0)", 0.2, 6.0, 1.5, 0.1, key="sd")
    dist = st.selectbox("Population shape", P.DISTRIBUTIONS, index=0, key="dist")
    st.markdown("---")
    st.header("Your pretest")
    which = st.selectbox("Which equal-variance test", P.PRETESTS, index=1, key="pre")
    pretest_alpha = st.select_slider(
        "Pretest alpha", options=[0.01, 0.05, 0.10, 0.20, 0.50, 1.0], value=0.05, key="pa"
    )
    st.caption("1.0 means it always fires - which is the same thing as skipping it and using Welch.")
    st.markdown("---")
    alpha = st.select_slider("Test alpha", options=[0.01, 0.025, 0.05, 0.10], value=0.05, key="alpha")
    reps = st.select_slider(
        "Monte-Carlo replicates", options=[2_000, 5_000, 10_000, 20_000], value=20_000, key="reps"
    )
    st.caption(
        "20,000 is what the full study runs. Below that the interval on a measured rate gets "
        "wider than the band being judged, and the honest verdict becomes 'cannot tell'."
    )


@st.cache_data(show_spinner="Running the simulation...")
def measure(n1: int, n2: int, sd_ratio: float, dist: str, which: str, alpha: float,
            pretest_alpha: float, reps: int) -> P.CellResult:
    """Cached so nudging one control does not re-run everything else in the sidebar."""
    d = P.Design(n1, n2, sd_ratio, dist)
    return P.run_cell(d, which, reps, alpha, pretest_alpha, seed=0)


cell = measure(int(n1), int(n2), float(sd_ratio), dist, which, float(alpha),
               float(pretest_alpha), int(reps))
lo, hi = cell.gated_interval
half = (P.wilson(alpha, int(reps))[1] - P.wilson(alpha, int(reps))[0]) / 2
# Three outcomes, not two: at a low replicate count "not shown to be broken" is the only claim
# available, and it is not the same claim as "this procedure is fine".
clean = lo >= P.BAND_LO and hi <= P.BAND_HI if alpha == 0.05 else not P.is_broken(cell.gated_error, int(reps))
undecided = not clean and not cell.gated_is_broken

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pretest fires on", f"{cell.pretest_reject_rate:.1%}", help="share of samples where it rejects equal variances")
c2.metric("Student always", f"{cell.student_error:.4f}", delta=f"{cell.student_error - alpha:+.4f}", delta_color="inverse")
c3.metric("Welch always", f"{cell.welch_error:.4f}", delta=f"{cell.welch_error - alpha:+.4f}", delta_color="inverse")
c4.metric("Pretest, then pick", f"{cell.gated_error:.4f}", delta=f"99% CI [{lo:.4f}, {hi:.4f}]", delta_color="off")

if undecided:
    st.info(
        f"**Cannot tell at {int(reps):,} replicates.** The two-stage procedure measured "
        f"{cell.gated_error:.4f}, but its 99% interval [{lo:.4f}, {hi:.4f}] still overlaps the "
        f"band. That is absence of evidence, not evidence of absence - raise the replicate count "
        f"to decide."
    )
elif cell.gated_is_broken:
    st.error(
        f"**The two-stage procedure is not a {alpha} test here.** It runs at {cell.gated_error:.4f}, "
        f"{cell.gated_error / alpha:.1f}x nominal, while Welch on the identical samples holds "
        f"{cell.welch_error:.4f}. The pretest clears "
        f"{cell.share_passed:.0%} of samples, and those are the ones that go to Student's."
    )
elif cell.gated_beats_welch_materially:
    st.success(
        f"**Here the pretest genuinely helped.** The two-stage procedure lands at "
        f"{cell.gated_error:.4f} against Welch's {cell.welch_error:.4f}, and the gap is larger "
        f"than the measurement error. Worth noting - it did not happen anywhere on the study's "
        f"14-design grid."
    )
else:
    st.warning(
        f"**Not broken, and not better.** The two-stage procedure sits at {cell.gated_error:.4f} "
        f"and Welch at {cell.welch_error:.4f} - within {half:.4f} of each other, which is the "
        f"measurement error at this replicate count. The pretest bought nothing here; it just "
        f"did not cost anything either."
    )

left, right = st.columns(2)

with left:
    st.subheader("What each procedure's error rate actually is")
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=BG)
    ax.set_facecolor(BG)
    names = ["Student always", "Welch always", "pretest, then pick"]
    vals = [cell.student_error, cell.welch_error, cell.gated_error]
    ax.barh(names, vals, color=[STUDENT, WELCH, GATED], height=0.55)
    ax.axvline(alpha, color=INK, ls="--", lw=1.8)
    for i, v in enumerate(vals):
        ax.text(max(v, 1e-5) * 1.1, i, f"{v:.4f}", va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(max(min(vals) * 0.4, 1e-5), 2.5)
    ax.set_xlabel("false-positive rate under a TRUE null (log scale)", color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, lw=0.7, axis="x")
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    st.pyplot(fig)
    st.caption(
        "The dashed line is what you asked for. Note that a bar far to the LEFT is also a "
        "failure - that is a test so conservative it has stopped detecting things."
    )

with right:
    st.subheader("The mechanism: what the pretest hands to Student's")
    fig2, ax2 = plt.subplots(figsize=(6.4, 3.4), facecolor=BG)
    ax2.set_facecolor(BG)
    if cell.share_passed >= 0.01:
        vals2 = [cell.student_error, cell.student_error_given_pass]
        ax2.barh(["on ALL samples", "on the ones it CLEARED"], vals2, color=[PRETEST, STUDENT], height=0.5)
        ax2.axvline(alpha, color=INK, ls="--", lw=1.8)
        for i, v in enumerate(vals2):
            ax2.text(max(v, 1e-5) * 1.1, i, f"{v:.4f}", va="center", fontsize=9, color=INK)
        ax2.set_xscale("log")
        ax2.set_xlim(max(min(vals2) * 0.4, 1e-5), 2.5)
    else:
        ax2.text(0.5, 0.5, f"the pretest clears only {cell.share_passed:.1%} of samples here,\n"
                           "so there is no passing subset to look at",
                 ha="center", va="center", fontsize=10, color=MUTED, transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
    ax2.set_xlabel("Student's false-positive rate (log scale)", color=MUTED, fontsize=9)
    ax2.grid(True, color=GRID, lw=0.7, axis="x")
    ax2.set_axisbelow(True)
    for s in ax2.spines.values():
        s.set_color(GRID)
    st.pyplot(fig2)
    if cell.share_passed >= 0.01 and cell.student_error > 0:
        ratio = cell.student_error_given_pass / cell.student_error
        st.caption(
            f"{ratio:.2f}x. The pretest and Student's pooled standard error read the same sample "
            f"variances, so 'it passed' selects the samples where the pooled estimate flatters "
            f"itself. Above 1.00x means the gate is concentrating the danger, not filtering it."
        )

st.markdown("---")
st.subheader("Is this design one where Student's needed fixing at all?")
d1, d2 = st.columns(2)
verdict = {
    "ok": "Student's is already inside its band here - there was nothing for the pretest to fix.",
    "inflated": "Student's is INFLATED here, so there is a real problem. The question is whether "
                "the pretest is the thing that solves it.",
    "conservative": "Student's is CONSERVATIVE here. That reads as safety in a false-positive "
                    "audit, and it is paid for in detections you will never see.",
}[cell.student_type_one_direction]
d1.metric("Student's direction under the null", cell.student_type_one_direction)
d2.metric("Welch holds its rate?", "yes" if cell.welch_controls_type_one else "no")
st.caption(verdict)
