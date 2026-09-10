"""Backtest protocol explorer: pick a protocol, see the number it reports and the number it means."""

from __future__ import annotations

import backtest as bt
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="Forecast backtest", layout="wide")
st.title("The forecast was excellent in-sample")
st.caption(
    "A backtest returns an ESTIMATE of a forecaster's future error, and the protocol decides "
    "how noisy that estimate is. Here the process is synthetic, so the quantity every protocol "
    "claims to report can be computed by re-drawing the future - and compared with what the "
    "protocol actually hands you."
)

with st.sidebar:
    st.header("The series")
    world = st.radio("what the process does", ["stable", "level break", "trend break"], key="world")
    n = st.select_slider("length", [120, 180, 240, 360], value=180, key="n")
    sigma = st.slider("noise sd", 1.0, 16.0, 4.0, 0.5, key="sigma")
    rho = st.slider("noise autocorrelation", 0.0, 0.95, 0.0, 0.05, key="rho")
    st.header("The forecaster")
    model_name = st.selectbox("model", list(bt.MODELS), index=4, key="model")
    h = st.slider("horizon h", 1, 24, 12, key="h")
    st.header("The protocol")
    k = st.slider("origins k", 1, 40, 20, key="k")
    step = st.slider("step between origins", 1, 24, 1, key="step")
    window = st.radio("training window", ["expanding", "sliding"], key="window")
    train_len = st.slider("sliding window length", 24, 160, 72, 4, key="train_len")
    seed = st.number_input("seed", 0, 9999, 3, key="seed")

kw = {"sigma": sigma, "rho": rho}
if world == "level break":
    kw |= {"break_at": int(n * 0.7), "break_shift": 25.0}
elif world == "trend break":
    kw |= {"break_at": int(n * 0.7), "break_trend": 0.60}

rng = np.random.default_rng(int(seed))
y = bt.make_series(rng, int(n), **kw)
model = bt.MODELS[model_name]
tl = None if window == "expanding" else int(train_len)
# a sliding window must be FULLY available at the earliest origin, otherwise the early folds
# quietly train on fewer points than the window you asked for and the comparison is not one
min_train = max(36, int(n * 0.5), tl or 0)

E = bt.rolling_origin(y, model, int(h), int(k), min_train, step=int(step),
                      window=window, train_len=tl)
ors = bt.origins(len(y), int(h), int(k), min_train, int(step))
if len(ors) == 0:
    st.error(
        f"NO ORIGINS: this series has {len(y)} points, the horizon takes {int(h)} of them and "
        f"every fold needs {min_train} points of training. Shorten the horizon, shorten the "
        f"sliding window, or lengthen the series."
    )
    st.stop()

per_origin = np.array([bt.rmse(r[~np.isnan(r)]) for r in E])
split_rmse = bt.rmse(bt.single_split(y, model, int(h)))
rolling_rmse = bt.rmse(E[~np.isnan(E)])

# the estimand: re-draw this exact process many times and score the model honestly
truth = bt.true_h_step_error(model, int(h), int(n) - int(h), 300, np.random.default_rng(999), **kw)

c1, c2, c3, c4 = st.columns(4)
c1.metric("single split", f"{split_rmse:.3f}", f"{(split_rmse / truth - 1) * 100:+.1f}% vs truth")
c2.metric(f"rolling ({len(ors)} origins)", f"{rolling_rmse:.3f}",
          f"{(rolling_rmse / truth - 1) * 100:+.1f}% vs truth")
c3.metric("the honest number", f"{truth:.3f}", "300 redrawn futures")
c4.metric("spread across origins", f"{per_origin.min():.2f} - {per_origin.max():.2f}",
          f"sd {per_origin.std(ddof=1) if len(per_origin) > 1 else 0:.3f}")

left, right = st.columns([3, 2])
with left:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), height_ratios=[2, 1])
    ax1.plot(y, color="#2f6fdb", lw=1.2)
    for o in ors:
        ax1.axvline(o, color="#c8562b", lw=0.6, alpha=0.5)
    ax1.axvline(len(y) - int(h), color="#1f2733", lw=2.0, ls="--")
    ax1.set_title("the series; red = a backtest origin, black = the single split")
    ax1.grid(True, color="#dfe4ea", lw=0.6)
    ax2.plot(ors, per_origin, color="#c8562b", marker="o", ms=3, lw=1.2)
    ax2.axhline(truth, color="#1f2733", ls="--", lw=1.5)
    ax2.set_title("what you would have reported, origin by origin (dashed = the honest number)")
    ax2.set_xlabel("origin")
    ax2.grid(True, color="#dfe4ea", lw=0.6)
    fig.tight_layout()
    st.pyplot(fig)

with right:
    st.subheader("What this protocol is worth")
    overlap = max(0, int(h) - int(step))
    st.write(
        f"Origins: **{len(ors)}** of the {int(k)} asked for. Each test window is {int(h)} points "
        f"and consecutive origins are {int(step)} apart, so neighbouring folds share "
        f"**{overlap} of {int(h)}** test points."
    )
    if len(per_origin) > 1:
        se_reported = per_origin.std(ddof=1) / np.sqrt(len(per_origin))
        st.write(f"The standard error you would publish is **{se_reported:.4f}** "
                 f"(sd across folds / sqrt(k)). Section 4 of the evidence run measures what it is "
                 f"really worth: at full overlap, k_eff is about **1** fold regardless of k.")
    st.write(
        f"Best origin available to report: **{per_origin.min():.3f}** "
        f"({per_origin.min() / truth * 100:.0f}% of the honest number). Worst: "
        f"**{per_origin.max():.3f}** ({per_origin.max() / truth * 100:.0f}%). Nothing in the data "
        f"says which one you picked."
    )

    st.subheader("Per-horizon error")
    per_h = np.sqrt(np.nanmean(E**2, axis=0))
    fig2, ax = plt.subplots(figsize=(5, 2.6))
    ax.bar(np.arange(1, len(per_h) + 1), per_h, color="#2f6fdb")
    ax.set_xlabel("horizon")
    ax.set_ylabel("RMSE")
    ax.grid(True, color="#dfe4ea", lw=0.6, axis="y")
    fig2.tight_layout()
    st.pyplot(fig2)
    st.caption("A mean over horizons is a mean over different estimands; the ranking of two "
               "models can reverse between h=1 and h=12.")

st.divider()
st.subheader("Best-of-M: what selection does to the number you publish")
m = st.slider("candidate models M (all equally good by construction)", 2, 20, 8, key="M")
reps = st.select_slider("replications", [100, 200, 400], value=200, key="reps")
u = np.random.default_rng(21).normal(size=(int(m), int(h)))
u = u / np.linalg.norm(u, axis=1, keepdims=True) * np.sqrt(int(h))


def _cand(j: int):
    def f(yy, hh, period=bt.PERIOD):
        return bt.f_trend_season(yy, hh, period) + 2.0 * u[j, :hh]
    return f


rng2 = np.random.default_rng(22)
rep_win, fut_win, field, mat = [], [], [], []
for _ in range(int(reps)):
    yy = bt.make_series(rng2, int(n) + int(h), **kw)
    sp = np.array([bt.rmse(bt.single_split(yy[:int(n)], _cand(j), int(h))) for j in range(int(m))])
    mat.append(sp)
    j = int(np.argmin(sp))
    rep_win.append(sp[j])
    field.append(sp.mean())
    a, p = bt.forecast_at(yy, _cand(j), int(n), int(h))
    fut_win.append(bt.rmse(a - p))
mat = np.array(mat)
cor = np.corrcoef(mat.T)
rho_bar = float((cor.sum() - int(m)) / (int(m) * (int(m) - 1))) if int(m) > 1 else 0.0
pred = float(np.mean(field) + bt.expected_min_of_m(int(m), max(0.0, rho_bar))
             * mat.std(axis=0, ddof=1).mean())

d1, d2, d3, d4 = st.columns(4)
d1.metric("winner's backtest number", f"{np.mean(rep_win):.3f}")
d2.metric("winner's actual next window", f"{np.mean(fut_win):.3f}",
          f"{(np.mean(fut_win) / np.mean(rep_win) - 1) * 100:+.1f}% optimism")
d3.metric("picking at random", f"{np.mean(field):.3f}")
d4.metric(f"predicted: E[min of {int(m)}] at rho={rho_bar:.2f}", f"{pred:.3f}",
          f"{pred - np.mean(rep_win):+.3f} vs measured")
st.caption(
    "The candidates are built to be equally good, so every point of 'improvement' the winner "
    "shows is the minimum operator. Because they wrap the same base fit they are correlated, "
    "and the correlation shrinks the winner's curse by exactly sqrt(1 - rho)."
)
