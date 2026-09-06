"""Set a quarterly target twelve defensible ways and watch them disagree.

The controls change the world, not the analysis: how noisy the metric is,
how fast it really grows, what the board asked for, how fast hiring is
planned. The twelve methods stay the same, and so does the point -- the
number that comes out of a planning meeting is mostly a property of which
sentence was said in it.
"""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import targets as T

st.set_page_config(page_title="Target Setter", layout="wide")
st.title("A target is a method plus a claim about the future")
st.caption(
    "Twelve target-setting methods, one metric history, and the gap between "
    "them measured against what actually happened."
)

# --------------------------------------------------------------------------
# Controls. These set module constants before anything is computed, so the
# whole engine -- history, methods, backtest -- sees one consistent world.
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("The world")
    T.G = st.slider("real trend growth, per month", 0.0, 0.03, 0.0125, 0.0005,
                    format="%.4f")
    T.SIGMA = st.slider("month-to-month noise (sigma)", 0.02, 0.30, 0.12, 0.01)
    seas = st.slider("seasonal amplitude", 0.0, 2.0, 1.0, 0.1)
    base_seas = np.array([0.92, 0.95, 1.04, 1.02, 0.99, 0.88,
                          0.83, 0.87, 1.06, 1.12, 1.14, 1.18])
    T.SEASONAL = 1.0 + (base_seas - 1.0) * seas

    st.header("The room")
    T.BOARD_MULTIPLE = st.slider("board multiple on last year", 1.0, 1.8,
                                 1.40, 0.05)
    T.MARKET_GROWTH_Q = st.slider("published category growth, per quarter",
                                  0.0, 0.10, 0.030, 0.005)
    T.HEADCOUNT_STEP = st.slider("heads added per step", 0.0, 3.0, 1.0, 0.5)
    T.HEADCOUNT_STEP_MONTHS = st.slider("months per hiring step", 2, 12, 6)

    st.header("The run")
    seed = st.number_input("history seed", 1, 99_999, T.SEED)
    origin = st.slider("set the target at month", T.MIN_HISTORY,
                       T.N_MONTHS - T.HORIZON, 120)
    n_paths = st.select_slider("re-runs for the hit rates",
                               [50, 100, 200, 400], value=100)
    bonus = st.number_input("threshold bonus if the target is met",
                            0, 1_000_000, 100_000, step=10_000)

series = T.make_history(seed=int(seed))
tg = T.targets_at(series, int(origin))
truth_mean, _ = T.truth_quarter(int(origin))
last_q = float(series[origin - T.HORIZON : origin].sum())
actual = float(series[origin : origin + T.HORIZON].sum())
lo_pi, hi_pi = T.prediction_interval(series, int(origin), 0.80)

order = sorted(tg, key=lambda k: tg[k])
spread = tg[order[-1]] / tg[order[0]]

c1, c2, c3, c4 = st.columns(4)
c1.metric("highest / lowest target", f"{spread:.2f}x")
c2.metric("spread, as % of last quarter",
          f"{(tg[order[-1]] - tg[order[0]]) / last_q:.1%}")
c3.metric("what the quarter actually did", f"{actual / last_q - 1:+.1%}")
c4.metric("80% interval width", f"{(hi_pi - lo_pi) / ((hi_pi + lo_pi) / 2):.1%}")

if (tg[order[-1]] - tg[order[0]]) / last_q > abs(actual / last_q - 1):
    st.warning(
        "The twelve methods disagree by more than the metric moved. The "
        "choice of method is a larger number than the thing being targeted."
    )

# --------------------------------------------------------------------------

st.subheader("The twelve targets")

rows = []
for name in order:
    v = tg[name]
    rows.append({
        "method": name,
        "provenance": T.PROVENANCE[name],
        "target": round(v),
        "vs last quarter": f"{v / last_q - 1:+.1%}",
        "vs the truth": round(v / truth_mean, 3),
        "inside 80% interval": "yes" if lo_pi <= v <= hi_pi else "no",
        "met?": "met" if actual >= v else "missed",
    })
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

pairs = list(itertools.combinations(tg.values(), 2))
inside = sum(1 for a, b in pairs if abs(a - b) < (hi_pi - lo_pi))
st.caption(
    f"{inside} of the {len(pairs)} pairwise disagreements are smaller than "
    f"the 80% prediction interval ({hi_pi - lo_pi:,.0f}). Those arguments "
    "are inside the noise of the forecast they are arguing about."
)

# --------------------------------------------------------------------------

st.subheader(f"Re-run the same {T.N_MONTHS // 12} years {n_paths} times")
st.caption(
    "One history gives one hit rate. These are the hit rates the same "
    "methods get across independent draws of the same process -- the spread "
    "is how much of a hit rate is luck."
)

with st.spinner("backtesting"):
    mp = T.multipath(int(n_paths), 90_000 + int(seed))

stats_rows = []
for name in sorted(T.METHODS, key=lambda n: -mp[n]["hit_rate"].mean()):
    h = mp[name]["hit_rate"]
    stats_rows.append({
        "method": name,
        "ambition": round(float(mp[name]["ambition"].mean()), 3),
        "hit rate": round(float(h.mean()), 3),
        "sd": round(float(h.std()), 3),
        "p05": round(float(np.quantile(h, 0.05)), 3),
        "p95": round(float(np.quantile(h, 0.95)), 3),
        "E[bonus]": round(float(h.mean()) * bonus),
    })
st.dataframe(pd.DataFrame(stats_rows), width="stretch", hide_index=True)

best, worst = stats_rows[0], stats_rows[-1]
st.info(
    f"**{best['method']}** is met {best['hit rate']:.0%} of the time and "
    f"**{worst['method']}** {worst['hit rate']:.0%}, on identical work. "
    f"Under this bonus that gap is worth "
    f"{best['E[bonus]'] - worst['E[bonus]']:,.0f} per period, paid for "
    "choosing which sentence to say in the meeting."
)

# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
lo_m = max(0, origin - 30)
axes[0].plot(np.arange(lo_m, origin), series[lo_m:origin], color="#141414",
             lw=1.1)
axes[0].plot(np.arange(origin, origin + T.HORIZON),
             series[origin : origin + T.HORIZON], color="#141414", lw=1.1,
             ls=":")
x = origin + np.array([0.0, T.HORIZON - 1])
axes[0].fill_between(x, lo_pi / T.HORIZON, hi_pi / T.HORIZON,
                     color="#d98324", alpha=0.15)
for name in order:
    met = actual >= tg[name]
    axes[0].plot(x, [tg[name] / T.HORIZON] * 2,
                 color="#4b7f52" if met else "#c0392b", lw=2.0)
axes[0].axhline(actual / T.HORIZON, color="#141414", lw=0.8, ls="--")
axes[0].set_title("green = met, red = missed; band = 80% interval",
                  loc="left", fontsize=9)
axes[0].set_ylabel("per month")

names = list(T.METHODS)
amb = [float(mp[n]["ambition"].mean()) for n in names]
hit = [float(mp[n]["hit_rate"].mean()) for n in names]
axes[1].scatter(amb, hit, s=30, color="#4a7c8c")
for n, a, h in zip(names, amb, hit):
    axes[1].annotate(n, (a, h), fontsize=6, xytext=(4, 0),
                     textcoords="offset points", color="#8a8a8a")
axes[1].axhline(0.5, color="#8a8a8a", lw=0.7, ls=":")
axes[1].set_xlabel("ambition (target / truth)")
axes[1].set_ylabel("hit rate")
axes[1].set_title("the hit rate is a property of the method", loc="left",
                  fontsize=9)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
st.pyplot(fig)

st.markdown(
    """
---
**What to take away.** A target that arrives as a single number has already
thrown away the three things needed to defend it: which method produced it,
what claim about the future it encodes, and how wide the interval around
that claim is. Write those down and the hit rate stops being a grade.

Run `python evidence.py` for the full audit, or `pytest test_targets.py` for
the assertions behind every number.
"""
)
