"""Six panels, all of them rebuilt from evidence.py rather than typed in."""

from __future__ import annotations

import contextlib
import io
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import evidence as E  # noqa: E402
import leadlag as L  # noqa: E402

INK = "#16222e"
MUTE = "#8b9aa7"
GOOD = "#1f7a5c"
BAD = "#b3402f"
WARN = "#c98a1a"
COOL = "#2b6ca3"
GRID = "#dfe5ea"
W = L.World()


def _style(ax, title: str, sub: str = "") -> None:
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=30)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=8.6, color=MUTE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTE, labelsize=8.4, length=0)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def panel_funnel(ax, res: Dict[str, object]) -> None:
    d = res["s1"]["data"]
    stages = [("activations", W.gain["activations"]), ("signups", W.gain["signups"]),
              ("web_sessions", 0.0)]
    leads = [W.true_lead[c] for c, _ in stages]
    corrs = [L.lagged_corr(d[c], d["revenue"], W.true_lead[c]) for c, _ in stages]
    gains = [g for _, g in stages]
    ax.plot(leads, corrs, "-o", color=COOL, lw=2.2, ms=9, zorder=3, label="correlation with revenue")
    ax.plot(leads, gains, "-s", color=BAD, lw=2.2, ms=8, zorder=3, label="revenue moved per unit pushed")
    for lead, c, name in zip(leads, corrs, [s[0] for s in stages]):
        ax.annotate(name, (lead, c), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=8.2, color=INK)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("months of warning", fontsize=8.8, color=MUTE)
    ax.set_ylim(-0.05, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax, "Warning time is bought with signal, and with leverage",
           "in a funnel the earliest stage is the weakest and the least pushable")


def panel_ccf(ax, res: Dict[str, object]) -> None:
    d = res["s1"]["data"]
    lags = range(-12, 13)
    for name, col in (("activations", COOL), ("support_tickets", BAD)):
        prof = [L.lagged_corr(d[name], d["revenue"], k) for k in lags]
        ax.plot(list(lags), prof, "-o", ms=3.4, lw=1.9, color=col, label=name)
        k = int(np.argmax(np.abs(prof)))
        ax.plot([list(lags)[k]], [prof[k]], "o", ms=10, mfc="none", mec=col, mew=2)
    ax.axvline(0, color=MUTE, lw=1, ls=":")
    ax.axhline(0, color=GRID, lw=1)
    ax.text(-11.5, 0.88, "x follows revenue", fontsize=8, color=BAD)
    ax.text(3.2, 0.88, "x leads revenue", fontsize=8, color=COOL)
    ax.set_xlabel("lag (months)", fontsize=8.8, color=MUTE)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, "The peak of the CCF is not a lead",
           "support tickets peak at lag -1 and outscore every real indicator")


def panel_horizon(ax, res: Dict[str, object]) -> None:
    oos = res["s3"]["oos"]
    order = sorted(L.CANDIDATES, key=lambda c: -oos[3][c]["gain_pct"])
    x = np.arange(len(order))
    g1 = [oos[1][c]["gain_pct"] for c in order]
    g3 = [oos[3][c]["gain_pct"] for c in order]
    ax.bar(x - 0.2, g1, 0.4, color=MUTE, label="horizon 1 month")
    ax.bar(x + 0.2, g3, 0.4, color=GOOD, label="horizon 3 months")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in order], fontsize=7.4,
                       rotation=32, ha="right")
    ax.set_ylabel("RMSE improvement vs revenue's own history (%)", fontsize=8.4, color=MUTE)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax, "Change the horizon and the shortlist reverses",
           f"Spearman over the four informative candidates: {res['s3']['rho_informative']:+.2f}")


def panel_actionable(ax, res: Dict[str, object]) -> None:
    oos, eff = res["s3"]["oos"], res["s4"]["effect"]
    offsets = {"web_sessions": (-6, 14), "awareness_index": (-10, -20),
               "signups": (8, 6), "activations": (8, 6), "marketing_spend": (8, 10)}
    for c in L.CANDIDATES:
        g, a = oos[3][c]["gain_pct"], eff[c]
        col = GOOD if a > 0 else (COOL if c in W.informative else MUTE)
        ax.scatter([g], [a], s=118, color=col, zorder=3, edgecolor="white", lw=1.2)
        if c in offsets:
            ax.annotate(c.replace("_", " "), (g, a), textcoords="offset points",
                        xytext=offsets[c], fontsize=8, color=INK,
                        ha="center" if c == "web_sessions" else "left")
    ax.axhline(0, color=BAD, lw=1.2, ls="--")
    ax.set_xlim(-8, 34)
    ax.set_ylim(-0.14, 0.93)
    ax.text(-7.2, -0.10, "on this line there is nothing downstream to push",
            fontsize=8, color=BAD)
    ax.set_xlabel("out-of-sample gain at the horizon needed (%)", fontsize=8.8, color=MUTE)
    ax.set_ylabel("revenue per unit pushed", fontsize=8.8, color=MUTE)
    _style(ax, "The best forecaster has a causal gain of zero",
           "forecasting rank and leverage are independent properties")


def panel_null(ax, res: Dict[str, object]) -> None:
    rates = res["s5"]["rates"]["ar1"]
    keys = list(L.FLAG_LABEL)
    vals = [rates[k] for k in keys]
    cols = [BAD if v > 0.5 else (WARN if v > 0.10 else GOOD) for v in vals]
    ax.barh(np.arange(len(keys)), vals, color=cols)
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([L.FLAG_LABEL[k] for k in keys], fontsize=7.6)
    ax.invert_yaxis()
    ax.axvline(0.05, color=INK, lw=1.4, ls="--")
    ax.text(0.075, -0.62, "nominal 0.05", fontsize=8, color=INK)
    for i, v in enumerate(vals):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 1.16)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("share of empty worlds yielding a 'leading indicator'",
                  fontsize=8.8, color=MUTE)
    _style(ax, f"A world with nothing in it, scanned {len(keys)} ways",
           f"{E.NULL_REPS} worlds; revenue persistent and seasonal, candidates independent")


def panel_lag(ax, res: Dict[str, object]) -> None:
    sweep = res["s6"]["sweep"]
    r = [s["r"] for s in sweep]
    ex = [s["exact"] for s in sweep]
    ax.plot(r, ex, "-o", color=COOL, lw=2.2, ms=7, zorder=3)
    ax.axhline(1 / 12, color=MUTE, lw=1.2, ls=":")
    ax.text(0.35, 1 / 12 + 0.03, "guessing on a 12-lag grid", fontsize=8, color=MUTE)
    for s in sweep:
        if s["sd"] in (0.3, 1.5, 9.0):
            ax.annotate(f"{s['exact']:.0%}", (s["r"], s["exact"]),
                        textcoords="offset points", xytext=(-4, 12), fontsize=8,
                        color=INK, ha="right")
    ax.set_xlabel("peak correlation actually observed", fontsize=8.8, color=MUTE)
    ax.set_ylabel("share of runs recovering the true lag", fontsize=8.4, color=MUTE)
    ax.set_ylim(0, 1)
    _style(ax, "Whether the lag is readable is a question about strength",
           f"five years of monthly history, {E.LAG_REPS} re-runs per point")


def main() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = E.run_all()
    fig, axes = plt.subplots(3, 2, figsize=(15.2, 15.6))
    fig.patch.set_facecolor("white")
    panel_funnel(axes[0][0], res)
    panel_ccf(axes[0][1], res)
    panel_horizon(axes[1][0], res)
    panel_actionable(axes[1][1], res)
    panel_null(axes[2][0], res)
    panel_lag(axes[2][1], res)
    fig.suptitle("A lead is a claim about a lag, and a lag has to be estimated",
                 fontsize=15.5, fontweight="bold", color=INK, x=0.055, ha="left", y=0.985)
    fig.text(0.055, 0.962,
             "Day 163 - leading-indicator-finder. One known funnel, ten candidates, "
             "four rankers and a measured null.",
             fontsize=9.6, color=MUTE, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.952])
    fig.subplots_adjust(hspace=0.52, wspace=0.24)
    for ext in ("png", "svg"):
        fig.savefig(f"lead_lag_audit.{ext}", dpi=200 if ext == "png" else None,
                    facecolor="white")
    print("wrote lead_lag_audit.png / .svg")


if __name__ == "__main__":
    main()
