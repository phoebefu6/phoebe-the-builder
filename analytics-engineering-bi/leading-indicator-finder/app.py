"""Rank ten candidate leading indicators four ways, then check the ranking."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import leadlag as L

st.set_page_config(page_title="Leading Indicator Finder", layout="wide")
st.title("A lead is a claim about a lag")
st.caption(
    "Ten candidates against one revenue series whose true leads and causal gains "
    "are known. Pick the horizon you actually need warning over."
)

with st.sidebar:
    st.header("The world")
    horizon = st.slider("Warning needed (months ahead)", 1, 6, 3)
    T = st.select_slider("History (months)", [60, 96, 120, 180, 240], value=240)
    sd_web = st.slider("Sensor noise on web_sessions", 0.3, 9.0, 0.3, 0.1)
    seed = st.number_input("Seed", value=20260902, step=1)
    reveal = st.checkbox("Reveal the causal gains", value=False)
    st.markdown(
        "---\n**How to read it.** `r@lag` is the usual scan. `|CCF| peak` takes the "
        "biggest correlation at any lag, including negative ones. `OOS gain` is the "
        "rolling-origin RMSE improvement over revenue's own history at the horizon "
        "chosen, with the lag re-fit on each training window."
    )

w = L.World(T=int(T), sd_web=float(sd_web))
d = L.simulate(w, seed=int(seed))
y = d["revenue"]

rows = []
for c in L.CANDIDATES:
    r1, l1 = L.rank_pearson_lead(d[c], y)
    r2, l2 = L.rank_pearson_abs_sym(d[c], y)
    r3, l3 = L.rank_prewhitened(d[c], y)
    _, gp = L.granger_f(d[c], y)
    g = L.oos_gain(y, d[c], horizon, min_train=max(60, int(T) - 96))
    useful = g["dm_p"] < 0.05 and g["gain_pct"] > 0
    if useful and w.gain[c] > 0:
        verdict = "watch AND pull"
    elif useful:
        verdict = "watch, cannot pull"
    elif r1 > 0.30:
        verdict = "correlated, no lead value"
    else:
        verdict = "drop"
    rows.append({
        "metric": c, "r@lag": round(r1, 3), "lag": l1,
        "|CCF| peak": round(r2, 3), "peak lag": l2,
        "prewhitened": round(r3, 3), "Granger p": float(f"{gp:.3g}"),
        "OOS gain %": round(g["gain_pct"], 2), "DM p": round(g["dm_p"], 3),
        "verdict": verdict,
        "true lead": w.true_lead[c], "dY/dX": w.gain[c],
    })
df = pd.DataFrame(rows)

hidden = ["true lead", "dY/dX"]
shown = df if reveal else df.drop(columns=hidden)

c1, c2, c3 = st.columns(3)
top_abs = df.loc[df["|CCF| peak"].abs().idxmax()]
top_oos = df.loc[df["OOS gain %"].idxmax()]
kept = df[df["verdict"].str.startswith("watch")]
peak_note = f"r={top_abs['|CCF| peak']:+.3f} at lag {top_abs['peak lag']}"
c1.metric("|CCF| peak names", top_abs["metric"], peak_note)
c2.metric(f"Best at h={horizon}", top_oos["metric"], f"{top_oos['OOS gain %']:+.2f}% vs own history")
c3.metric("Survive the backtest", f"{len(kept)} of 10",
          f"{int((kept['dY/dX'] > 0).sum())} can also be pulled")

st.dataframe(
    shown.style.background_gradient(subset=["OOS gain %"], cmap="Greens")
    .format({"Granger p": "{:.3g}"}),
    hide_index=True,
)

if not reveal:
    st.info(
        "Two of the ten can be moved and the rest can only be watched. Nothing in "
        "the table above says which - tick **Reveal the causal gains** in the sidebar."
    )
else:
    zero = df[(df["OOS gain %"] > 5) & (df["dY/dX"] == 0)]["metric"].tolist()
    if zero:
        st.warning(
            "Forecasts well, cannot be pushed: " + ", ".join(zero) +
            ". A leading-indicator scan is a forecasting result, never a plan."
        )

left, right = st.columns(2)
with left:
    st.subheader("Cross-correlation, both signs")
    pick = st.multiselect("Candidates", L.CANDIDATES,
                          default=["activations", "support_tickets"])
    lags = list(range(-12, 13))
    if pick:
        prof = pd.DataFrame(
            {c: [L.lagged_corr(d[c], y, k) for k in lags] for c in pick},
            index=pd.Index(lags, name="lag (months)"),
        )
        st.line_chart(prof, height=300)
        st.caption(
            "Everything left of zero FOLLOWS revenue. A scan that takes the biggest "
            "absolute correlation reads the sign off and reports a follower as a leader."
        )

with right:
    st.subheader("The horizon changes the answer")
    grid = []
    for h in range(1, 7):
        for c in w.informative:
            g = L.oos_gain(y, d[c], h, min_train=max(60, int(T) - 96))
            grid.append({"horizon": h, "metric": c, "gain": g["gain_pct"]})
    gd = pd.DataFrame(grid).pivot(index="horizon", columns="metric", values="gain")
    st.line_chart(gd, height=300)
    st.caption(
        "Only the four informative candidates are shown. A metric whose lead is "
        "shorter than the horizon cannot be read early enough to help, however "
        "strongly it correlates."
    )

st.markdown(
    """
### What this build measured

- The `|CCF|` peak crowns **support_tickets**, which *follows* revenue - and it
  outscores every real indicator, because revenue is persistent.
- Change the horizon from 1 month to 3 and the shortlist of real indicators
  **reverses** (Spearman -0.80 over the four).
- In simulated worlds containing nothing at all, a 10 x 12
  correlation scan reports a leading indicator **100%** of the time. Bonferroni
  alone leaves 67%; a Bartlett effective-sample correction alone leaves 90%.
  Granger over the same candidates is already calibrated (0.445, against 0.401
  for a perfect 5% test over 10 tries) and needs only Bonferroni.
- The best forecaster at the horizon that matters has a causal gain of **0.000**.

Full numbers: `python evidence.py`. Assertions: `python -m pytest test_leadlag.py -q`.
"""
)
