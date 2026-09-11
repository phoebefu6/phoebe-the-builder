"""Interval explorer: build a band, then measure whether its claim holds where it will be used."""

from __future__ import annotations

import intervals as iv
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="Prediction intervals", layout="wide")
st.title("The point forecast is useless")
st.caption(
    "A 95% band is a claim about a frequency - at a horizon, in a regime. The process here is "
    "synthetic, so the frequency can be measured by redrawing the future instead of assumed. "
    "Pointwise coverage, path coverage and conditional coverage are three different questions "
    "about the same band, and they do not agree."
)

with st.sidebar:
    st.header("The world")
    world = st.radio("what the noise does", ["constant", "two volatility regimes",
                                             "autocorrelated"], key="world")
    rho = st.slider("autocorrelation rho", 0.0, 0.95, 0.7, 0.05, key="rho")
    vol_ratio = st.slider("regime variance ratio", 1.0, 25.0, 9.0, 1.0, key="vol_ratio")
    sigma = st.slider("noise sd", 1.0, 12.0, 4.0, 0.5, key="sigma")
    st.header("The forecast")
    n_train = st.select_slider("training points", [36, 60, 120, 168, 240], value=168, key="n_train")
    h = st.slider("horizon h", 1, 24, 12, key="h")
    st.header("The band")
    method_name = st.selectbox("method", list(iv.METHODS), index=2, key="method")
    conf = st.select_slider("nominal level", [0.80, 0.90, 0.95, 0.99], value=0.95, key="conf")
    reps = st.select_slider("futures to redraw", [200, 400, 800], value=400, key="reps")
    seed = st.number_input("seed", 0, 9999, 5, key="seed")

kw = {"sigma": sigma}
if world == "two volatility regimes":
    kw |= {"vol_period": 24, "vol_ratio": vol_ratio}
elif world == "autocorrelated":
    kw |= {"rho": rho}

method = iv.METHODS[method_name]
if method_name in ("rolling h-step quantile", "split conformal") and n_train < 120:
    st.warning(
        f"'{method_name}' calibrates on rolling origins inside the training window. At "
        f"{n_train} training points there are too few origins, and it falls back to the "
        f"gaussian band - a real property of the method, not a bug in this app."
    )

r = iv.coverage_run(method, int(h), int(n_train), int(reps), np.random.default_rng(int(seed)),
                    float(conf), **kw)
point = float(r["hits"].mean())
path = float(r["hits"].all(axis=1).mean())

c1, c2, c3, c4 = st.columns(4)
c1.metric("pointwise coverage", f"{point:.3f}", f"{point - conf:+.3f} vs nominal")
c2.metric("whole-path coverage", f"{path:.3f}", f"nominal^h = {conf ** int(h):.3f}")
c3.metric("mean width", f"{r['width'].mean():.2f}")
k_eff = float(np.log(path) / np.log(point)) if 0 < path < 1 and point < 1 else float("nan")
c4.metric("independent horizons (k_eff)", f"{k_eff:.2f}" if np.isfinite(k_eff) else "n/a",
          f"of {int(h)}")

left, right = st.columns([3, 2])
with left:
    y, sd = iv.make_series(np.random.default_rng(int(seed) + 1), int(n_train) + int(h), **kw)
    lo, hi = method(y[:int(n_train)], int(h), float(conf))
    fut = np.arange(int(n_train), int(n_train) + int(h))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    show = max(0, int(n_train) - 80)
    ax.plot(np.arange(show, int(n_train)), y[show:int(n_train)], color="#6b7684", lw=1.2,
            label="history")
    ax.fill_between(fut, lo, hi, color="#2f6fdb", alpha=0.18, label=f"{conf:.0%} band")
    ax.plot(fut, y[int(n_train):], color="#c8562b", lw=1.6, marker="o", ms=3, label="what happened")
    inside = (y[int(n_train):] >= lo) & (y[int(n_train):] <= hi)
    ax.scatter(fut[~inside], y[int(n_train):][~inside], color="#c8562b", s=90, zorder=5,
               facecolors="none", linewidths=1.8)
    ax.set_title(f"one draw: {int(inside.sum())} of {int(h)} points inside")
    ax.grid(True, color="#dfe4ea", lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 2.6))
    ax2.plot(np.arange(1, int(h) + 1), r["hits"].mean(axis=0), color="#2f6fdb", marker="o", lw=1.6)
    ax2.axhline(conf, color="#1f2733", ls="--", lw=1.5)
    ax2.set_xlabel("horizon")
    ax2.set_ylabel("coverage")
    ax2.set_title("coverage by horizon (dashed = the claim)")
    ax2.grid(True, color="#dfe4ea", lw=0.6)
    fig2.tight_layout()
    st.pyplot(fig2)

with right:
    st.subheader("Conditional coverage")
    lo_sd = r["sd"] < np.sqrt(r["sd"].min() * r["sd"].max())
    if lo_sd.any() and (~lo_sd).any():
        q, v = float(r["hits"][lo_sd].mean()), float(r["hits"][~lo_sd].mean())
        st.write(
            f"In the quiet half of the points the band covers **{q:.3f}**; in the volatile half, "
            f"**{v:.3f}**. The marginal figure - the one every report quotes - is **{point:.3f}**, "
            f"and it is the average of two numbers {abs(q - v):.3f} apart."
        )
    else:
        st.write(
            f"This world has one noise level, so conditional and marginal coverage are the same "
            f"question ({point:.3f}). Switch to two volatility regimes to separate them."
        )

    st.subheader("Can you even check it?")
    m = st.select_slider("test points available", [20, 50, 100, 250, 1000, 5000], value=100,
                         key="m")
    st.write(
        f"Power to catch a band that really covers {conf - 0.05:.0%} instead of {conf:.0%}:\n\n"
        f"- if those {int(m)} points were independent: **"
        f"{iv.coverage_check_power(conf - 0.05, float(conf), int(m)):.3f}**\n"
        f"- at Day 172's measured k_eff for rolling windows stepped by 1 (1/20): **"
        f"{iv.coverage_check_power(conf - 0.05, float(conf), int(m), 1 / 20):.3f}**"
    )
    st.caption("A validation with no power is why a bad band survives a backtest.")

    st.subheader("Scored properly")
    rng = np.random.default_rng(int(seed) + 2)
    rows = []
    for name, mm in iv.METHODS.items():
        sc, cv, wd = [], [], []
        for _ in range(120):
            yy, _s = iv.make_series(rng, int(n_train) + int(h), **kw)
            a = yy[int(n_train):]
            l2, h2 = mm(yy[:int(n_train)], int(h), float(conf))
            sc.append(float(np.mean(iv.interval_score(l2, h2, a, float(conf)))))
            cv.append(float(np.mean((a >= l2) & (a <= h2))))
            wd.append(float(np.mean(h2 - l2)))
        rows.append((name, np.mean(cv), np.mean(wd), np.mean(sc)))
    rows.sort(key=lambda t: t[3])
    st.dataframe(
        {"method": [r_[0] for r_ in rows],
         "coverage": [round(r_[1], 3) for r_ in rows],
         "width": [round(r_[2], 2) for r_ in rows],
         "interval score": [round(r_[3], 2) for r_ in rows]},
        hide_index=True, use_container_width=True,
    )
    st.caption("Coverage can always be bought by widening. The interval score charges for width "
               "and misses in the same units, so it is the column to rank on.")
