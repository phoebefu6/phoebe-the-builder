"""Streamlit front end: describe your design, and it measures what your t-test will actually do.

The point of the UI is that you do not have to trust the tables in the README. Set your own n,
your own variance gap and your own distribution, and the Monte-Carlo runs live.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import ttests as tt

INK, GRID, BG = "#1f2733", "#dfe4ea", "#ffffff"
STUDENT, WELCH, ZED = "#c8562b", "#2f6fdb", "#7a4fa3"
COLOUR = {"student": STUDENT, "welch": WELCH, "z-shortcut": ZED}

st.set_page_config(page_title="Which t-test?", layout="wide")
st.title("Which t-test?")
st.caption(
    "Student's, Welch's, and the normal-curve shortcut, run on YOUR design under a null that is "
    "exactly true. Whatever rejection rate comes back is the false-positive rate you are buying."
)

with st.sidebar:
    st.header("Your design")
    n1 = st.number_input("Group 1 size", min_value=2, max_value=5000, value=50, step=1, key="n1")
    n2 = st.number_input("Group 2 size", min_value=2, max_value=5000, value=10, step=1, key="n2")
    sd1 = st.number_input("Group 1 sd", min_value=0.01, max_value=100.0, value=1.0, step=0.1, key="sd1")
    sd2 = st.number_input("Group 2 sd", min_value=0.01, max_value=100.0, value=3.0, step=0.1, key="sd2")
    dist = st.selectbox("Population shape", tt.DISTRIBUTIONS, index=0, key="dist")
    alpha = st.select_slider("alpha", options=[0.01, 0.025, 0.05, 0.10], value=0.05, key="alpha")
    reps = st.select_slider("Monte-Carlo replicates", options=[5_000, 20_000, 40_000], value=20_000, key="reps")
    shift = st.slider("True difference in means (0 = null true)", 0.0, 1.5, 0.0, 0.1, key="shift")

cell = tt.Cell(n1=int(n1), n2=int(n2), sd1=float(sd1), sd2=float(sd2), dist=dist, shift=float(shift))
rates = tt.simulate(cell, reps=int(reps), alpha=float(alpha), seed=0)
null_true = shift == 0.0

st.subheader(f"{cell.label()} - {cell.balance}")
cols = st.columns(3)
for col, name in zip(cols, tt.TESTS):
    rate = rates[name]
    lo, hi = tt.wilson(rate * reps, int(reps))
    if null_true:
        delta = tt.verdict(rate, int(reps), float(alpha))
        col.metric(name, f"{rate:.4f}", delta, delta_color="off")
        col.caption(f"95% CI [{lo:.4f}, {hi:.4f}] - nominal is {alpha}")
    else:
        col.metric(name, f"{rate:.4f}", "power")
        col.caption(f"95% CI [{lo:.4f}, {hi:.4f}]")

if null_true:
    worst = max(tt.TESTS, key=lambda t: rates[t])
    if rates[worst] / alpha > 1.5:
        st.error(
            f"**{worst}** is rejecting a true null {rates[worst] / alpha:.1f}x as often as it "
            f"claims ({rates[worst]:.4f} against a nominal {alpha}). On this design that test is "
            "not a 5% test, whatever the p-value says."
        )
    else:
        st.success(f"All three are within 1.5x of nominal on this design (worst: {rates[worst]:.4f}).")
else:
    st.info("A true difference is set, so these are POWER, not error rates. Set the slider to 0 "
            "to measure the false-positive rate instead.")

fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=BG)
ax.set_facecolor(BG)
ax.grid(True, color=GRID, lw=0.7, axis="x")
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_color(GRID)
names = list(tt.TESTS)
ax.barh(range(len(names)), [rates[n] for n in names], color=[COLOUR[n] for n in names], height=0.55)
for i, n in enumerate(names):
    lo, hi = tt.wilson(rates[n] * reps, int(reps))
    ax.plot([lo, hi], [i, i], color=INK, lw=1.6)
    ax.text(hi + 0.004, i, f"{rates[n]:.4f}", va="center", color=INK, fontsize=9)
if null_true:
    ax.axvline(float(alpha), color=INK, ls="--", lw=1.8)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel("rejection rate (power if a true difference is set, Type I error if not)")
ax.set_xlim(0, max(max(rates.values()) * 1.25, float(alpha) * 1.6))
fig.tight_layout()
st.pyplot(fig)

st.divider()
st.markdown(
    """
**How to read this.** Under a true null the rejection rate IS the Type I error, so a bar past
the dashed line is a test that lies about its own alpha. Three things this app will show you:

1. Set `n1 = n2` and slide the variance gap as far as you like - Student's barely moves. The
   textbook warning about unequal variances is only half the condition.
2. Set `n1 = 50, n2 = 10, sd = 1/3` - Student's runs near 0.29. Reverse the group sizes and it
   collapses to 0.0008, which is a test with almost no power left.
3. Set the shape to `lognormal` with unequal n - now **Welch** is the inflated one. It fixes
   unequal variance, not skew.

The `z-shortcut` is Welch's statistic read off the normal curve, which is what a hand-rolled
significance helper usually does. It is above nominal on every design here.
"""
)
