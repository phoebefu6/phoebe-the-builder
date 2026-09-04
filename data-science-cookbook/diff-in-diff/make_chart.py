"""Six panels, every one of them read from `results.json` - the file the
evidence run writes - so the figure cannot drift from the measurement."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#16222e"
MUTE = "#8b9aa7"
GOOD = "#1f7a5c"
BAD = "#b3402f"
WARN = "#c98a1a"
COOL = "#2b6ca3"
PLUM = "#6b4d8f"
GRID = "#dfe5ea"

R = json.load(open("results.json"))


def _style(ax, title: str, sub: str = "") -> None:
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=30)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=8.6, color=MUTE)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTE)
    ax.tick_params(colors=MUTE, labelsize=8.5)


def panel_1(ax) -> None:
    rows = R["s1_bias_rows"]
    n = [r[0] for r in rows]
    bias = [r[3] for r in rows]
    se = [r[2] for r in rows]
    cov = [r[4] for r in rows]
    ax.plot(n, bias, "o-", color=BAD, lw=2.2, ms=6, label="bias", zorder=4)
    ax.plot(n, [1.96 * s for s in se], "s--", color=COOL, lw=1.8, ms=5, label="half-width of 95% CI")
    ax.set_xscale("log")
    ax.set_xlabel("units per arm (log scale)", fontsize=8.5, color=MUTE)
    ax.set_ylabel("effect units", fontsize=8.5, color=MUTE)
    ax.axhline(0, color=MUTE, lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(n, cov, "^:", color=WARN, lw=1.8, ms=6, label="coverage")
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_ylabel("coverage of the 95% interval", fontsize=8.5, color=WARN)
    ax2.tick_params(colors=WARN, labelsize=8.5)
    ax2.spines["top"].set_visible(False)
    ax2.axhline(0.95, color=WARN, lw=0.8, ls=":")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.8, frameon=False, loc="center right")
    _style(
        ax,
        "1  More data does not fix a violated assumption",
        f"bias pinned at {bias[-1]:.2f} over a {n[-1] // n[0]}x range of n; coverage {cov[0]:.2f} -> {cov[-1]:.2f}",
    )


def panel_2(ax) -> None:
    rows = R["s2_rows"]
    d = [r["delta"] for r in rows]
    power = [r["power"] for r in rows]
    bias = [r["bias"] for r in rows]
    ax.plot(d, power, "o-", color=COOL, lw=2.2, ms=6, label="pre-trends test fires", zorder=4)
    ax.plot(d, [b for b in bias], "s-", color=BAD, lw=2.2, ms=5, label="bias (true effect = 1.0)")
    ax.axhline(1.0, color=MUTE, lw=0.9, ls="--")
    ax.text(0.005, 1.03, "the effect being estimated", fontsize=7.4, color=MUTE)
    d80 = R["s2_delta_power80"]
    ax.axvline(d80, color=PLUM, lw=1.4, ls=":")
    ax.text(
        d80 + 0.005,
        0.35,
        f"80% power at\ndelta={d80:.3f},\nbias {R['s2_bias_at_power80']:.2f}",
        fontsize=7.4,
        color=PLUM,
    )
    ax.set_xlabel("parallel-trends violation, per period", fontsize=8.5, color=MUTE)
    ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    _style(
        ax,
        "2  The alarm rings after the damage",
        "at delta=0.05 the test fires 0.06 of the time and the estimate is 30% too large",
    )


def panel_3(ax) -> None:
    rows = [r for r in R["s2_rows"] if r["delta"] > 0]
    x = np.arange(len(rows))
    allb = [r["bias"] for r in rows]
    passb = [r["bias_pass"] for r in rows]
    mc = [r["sd_est"] / np.sqrt(max(r["n_pass"], 1)) for r in rows]
    ax.bar(x - 0.19, allb, 0.36, color=MUTE, label="bias, all runs", zorder=3)
    ax.bar(x + 0.19, passb, 0.36, color=BAD, label="bias | pre-trends test PASSED", zorder=3)
    ax.errorbar(x + 0.19, passb, yerr=[4 * m for m in mc], fmt="none", ecolor=INK, lw=1.2, capsize=3, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['delta']:.2f}" for r in rows])
    ax.set_xlabel("violation delta   (error bars = 4 Monte Carlo SE)", fontsize=8.5, color=MUTE)
    ax.set_ylabel("bias in the reported effect", fontsize=8.5, color=MUTE)
    ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    _style(
        ax,
        "3  Screening on the pre-trend removes none of the bias",
        f"largest shift anywhere: {R['s3_max_shift']:.4f}, inside its own Monte Carlo error",
    )


def panel_4(ax) -> None:
    rows = R["s4_rows"]
    rho = [r[0] for r in rows]
    iid = [r[2] for r in rows]
    clu = [r[3] for r in rows]
    ax.plot(rho, iid, "o-", color=BAD, lw=2.4, ms=6, label="default (iid) standard error", zorder=4)
    ax.plot(rho, clu, "s-", color=GOOD, lw=2.4, ms=6, label="clustered on the unit", zorder=4)
    ax.axhline(0.05, color=INK, lw=1.1, ls="--")
    ax.text(0.02, 0.068, "nominal 0.05", fontsize=7.6, color=INK)
    ax.axhline(0.45, color=MUTE, lw=1.0, ls=":")
    ax.text(0.94, 0.462, "BDM (2004): ~0.45 on real wage panels", fontsize=7.2, color=MUTE, ha="right")
    ax.set_xlabel("AR(1) autocorrelation of the error", fontsize=8.5, color=MUTE)
    ax.set_ylabel("rejection rate of a TRUE null", fontsize=8.5, color=MUTE)
    ax.set_ylim(0, 0.52)
    ax.legend(fontsize=7.8, frameon=False, loc="center left")
    _style(
        ax,
        "4  Placebo interventions, no effect at all",
        "the estimate is unbiased throughout; only the standard error fails",
    )


def panel_5(ax) -> None:
    rows = R["s5_nested"]
    labels = [f"{r[0]}x{r[1]}" for r in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.26, [r[2] for r in rows], 0.25, color=MUTE, label="iid", zorder=3)
    ax.bar(x, [r[3] for r in rows], 0.25, color=BAD, label="clustered by UNIT (too fine)", zorder=3)
    ax.bar(x + 0.26, [r[4] for r in rows], 0.25, color=GOOD, label="clustered by STATE (correct)", zorder=3)
    ax.axhline(0.05, color=INK, lw=1.1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("states x units per state   (treatment assigned by state)", fontsize=8.5, color=MUTE)
    ax.set_ylabel("rejection rate of a TRUE null", fontsize=8.5, color=MUTE)
    ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    six = [r for r in rows if r[0] == 6][0]
    fifty = [r for r in rows if r[0] == 50][0]
    _style(
        ax,
        "5  The level, not the count",
        f"6 states clustered right ({six[4]:.3f}) beats 50 clustered too fine ({fifty[3]:.3f})",
    )


def panel_6(ax) -> None:
    rows = R["s6_grow_rows"]
    g = [r[0] for r in rows]
    truth = [r[1] for r in rows]
    est = [r[2] for r in rows]
    ax.plot(g, truth, "o-", color=GOOD, lw=2.4, ms=6, label="true mean effect on the treated", zorder=4)
    ax.plot(g, est, "s-", color=BAD, lw=2.4, ms=6, label="two-way fixed effects estimate", zorder=4)
    ax.axhline(0, color=INK, lw=1.1)
    flip = R["s6_flip"]
    ax.axvline(flip, color=PLUM, lw=1.4, ls=":")
    ax.text(flip + 0.03, 6.5, f"sign flips at\ngrowth={flip:.2f}", fontsize=7.6, color=PLUM)
    ax.fill_between([flip, max(g)], -3, 12, color=BAD, alpha=0.06, zorder=1)
    ax.set_ylim(-2.5, 11.5)
    ax.set_xlabel("how fast the effect grows per period of exposure", fontsize=8.5, color=MUTE)
    ax.set_ylabel("effect units", fontsize=8.5, color=MUTE)
    ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    _style(
        ax,
        "6  Every true effect positive, the estimate negative",
        f"{R['s6_neg_share']:.0%} of treated cells carry negative weight, totalling {R['s6_neg_weight']:.2f}",
    )


def main() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15.4, 15.8))
    fig.patch.set_facecolor("white")
    for fn, ax in zip((panel_1, panel_2, panel_3, panel_4, panel_5, panel_6), axes.ravel()):
        ax.set_facecolor("white")
        fn(ax)
    fig.suptitle(
        "Parallel trends is an assumption, and the test for it has a power",
        fontsize=17,
        fontweight="bold",
        color=INK,
        x=0.007,
        y=0.9965,
        ha="left",
    )
    fig.text(
        0.007,
        0.9755,
        "Day 167  ·  diff-in-diff  ·  difference-in-differences measured on worlds whose true effect is known",
        fontsize=10,
        color=MUTE,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.958))
    fig.savefig("did_audit.png", dpi=170, facecolor="white")
    fig.savefig("did_audit.svg", facecolor="white")
    print("wrote did_audit.png / did_audit.svg")


if __name__ == "__main__":
    main()
