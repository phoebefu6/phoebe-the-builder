"""Goodhart detector -- point a set of detectors at a proxy that became a target.

Move the exploit until it pays, watch the KPI improve and the outcome fall, and
see which detectors notice and how late.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import goodhart as G  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Goodhart detector", layout="wide")
st.title("A proxy metric is a bet that a correlation survives being optimised")
st.caption(
    "One latent driver, one outcome, one proxy, one exploit. The proxy is a good "
    "proxy until somebody is paid for it."
)

with st.sidebar:
    st.header("The world")
    regime = st.radio(
        "What the target says",
        ["threshold", "continuous"],
        format_func=lambda r: "Clear the line (a quota)" if r == "threshold" else "Make it go up",
    )
    gamma = st.slider("gamma -- how much the exploit moves the proxy", 0.0, 2.0, 1.10, 0.05)
    kappa = st.slider("kappa -- quality bought by honest effort", 0.1, 1.2, 0.60, 0.05)
    scruple = st.slider("median scruple (lower = more people game)", 0.8, 3.5, 1.80, 0.05)
    n_agents = st.select_slider("agents", [75, 150, 300, 600, 1200], value=600)
    alpha = st.select_slider("alpha", [0.01, 0.05, 0.10], value=0.05)

world = replace(
    G.World(), gamma=gamma, kappa=kappa, scruple_median=scruple, n_agents=n_agents
)
panel = G.simulate(world, regime=regime)
dec = G.decompose(world, panel)

edge = world.exploit_edge
if edge <= 0:
    st.success(
        f"**Nobody games this.** The exploit buys {edge:+.2f} proxy points per unit of "
        f"diverted effort, so honest work is the cheaper way to move the number. "
        "Goodhart's law is not a law about metrics, it is a statement about which "
        "of two paths is cheaper."
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("corr(proxy, outcome) before", f"{world.rho_clean:.3f}")
c2.metric("proxy moved", f"{dec['proxy_delta']:+.4f}")
c3.metric("outcome moved", f"{dec['outcome_delta']:+.4f}",
          delta=f"true {dec['outcome_delta_true']:+.4f}", delta_color="off")
c4.metric("effort diverted", f"{100*dec['diverted_share']:.0f}%")

left, right = st.columns([3, 2])

with left:
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    t = np.arange(panel.n_periods)
    ax.axvline(panel.t_target - 0.5, color="#c0392b", ls="--", lw=1.1)
    ax.plot(t, panel.proxy.mean(1), color="#4a7c8c", lw=2, marker="o", ms=3, label="proxy")
    ax.plot(t, panel.outcome.mean(1), color="#c0392b", lw=2, marker="s", ms=3, label="outcome")
    ax.set_xlabel("period")
    ax.set_ylabel("mean level")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, clear_figure=True)

    if panel.threshold is not None:
        fig2, ax2 = plt.subplots(figsize=(7.4, 2.9))
        bins = np.linspace(panel.pre(panel.proxy).min(), panel.pre(panel.proxy).max(), 60)
        ax2.hist(panel.pre(panel.proxy), bins=bins, alpha=0.55, color="#8a8a8a",
                 density=True, label="before the target")
        ax2.hist(panel.post(panel.proxy), bins=bins, alpha=0.55, color="#4b7f52",
                 density=True, label="after")
        ax2.axvline(panel.threshold, color="#c0392b", lw=1.4)
        ax2.text(panel.threshold, ax2.get_ylim()[1] * 0.94, " the line",
                 color="#c0392b", fontsize=8)
        ax2.set_xlabel("proxy value")
        ax2.legend(frameon=False, fontsize=8)
        ax2.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig2, clear_figure=True)
        st.caption(
            "Excess mass just above the line is what `bunching` measures. It needs no "
            "outcome, which is why it is the only detector available in time."
        )

with right:
    rows = []
    for name, v in G.run_all(panel).items():
        rows.append({
            "detector": name,
            "sees": "outcome" if v.needs_outcome else "proxy only",
            "statistic": None if np.isnan(v.stat) else round(v.stat, 4),
            "p": v.pvalue,
            "fires": "FIRES" if v.fires(alpha) else "-",
        })
    df = pd.DataFrame(rows).sort_values("p")
    st.dataframe(
        df.style.format({"p": "{:.2e}"}), width="stretch", hide_index=True
    )
    st.markdown(
        f"""
**Exchange rate.** Every proxy point cost **{dec['exchange_rate']:.2f}** outcome points.

**What the detectors cannot tell you.** A correlation drop of this size is also
what you get from having *chosen* this proxy out of a handful of candidates on a
short history, with nobody gaming anything. Before reading a drop as evidence,
ask how many observations the metric was picked on.

**No detector here works on the KPI alone.** Each one needs a second series the
target did not control: the outcome, a sibling metric nobody was told to move,
or the shape of the distribution around a line.
"""
    )

st.divider()
st.caption(
    "Day 162 - goodhart-detector. Full argument, power curves and the winner's-curse "
    "result: `python evidence.py`."
)
