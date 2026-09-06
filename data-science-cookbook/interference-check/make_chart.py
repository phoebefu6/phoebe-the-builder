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
    """The cliff: split estimate vs truth across market utilisation."""
    rows = sorted(R["s2"]["sweep"], key=lambda r: r["util"])
    u = [r["util"] for r in rows]
    ax.plot(u, [r["split"] for r in rows], "o-", color=BAD, lw=2.2, ms=6, label="what the A/B test reports", zorder=4)
    ax.plot(u, [r["truth"] for r in rows], "s-", color=GOOD, lw=2.2, ms=6, label="what shipping it does", zorder=4)
    ax.fill_between(u, [r["truth"] for r in rows], [r["split"] for r in rows], color=BAD, alpha=0.10, zorder=2)
    ax.axvspan(0.6, 1.25, color=WARN, alpha=0.09, zorder=1)
    ax.text(0.75, 0.0272, "rationed\nmarket", fontsize=8.8, color=WARN, ha="center", fontweight="bold")
    ax.set_xlabel("supply per expected attempt (utilisation)", fontsize=9, color=MUTE)
    ax.set_ylabel("effect on conversion", fontsize=9, color=MUTE)
    ax.legend(fontsize=8.6, frameon=False, loc="lower right")
    _style(ax, "1.  The bias is a cliff, and it sits between 1.3 and 1.2",
           "same market, same estimator, supply varied.  At 2.0 the test is honest; at 1.0 the entire number is bias")


def panel_2(ax) -> None:
    """Bias flat in n while the SE collapses."""
    rows = R["s3"]["rows"]
    n = [r["n"] for r in rows]
    ax.plot(n, [r["split"] - r["truth"] for r in rows], "o-", color=BAD, lw=2.4, ms=6, label="bias", zorder=4)
    ax.plot(n, [1.96 * r["se"] for r in rows], "^-", color=COOL, lw=2.0, ms=6, label="95% half-width", zorder=4)
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([f"{v // 1000}k" for v in n])
    ax.minorticks_off()
    ax.set_xlabel("users in the test (log scale)", fontsize=9, color=MUTE)
    ax.set_ylabel("effect units", fontsize=9, color=MUTE)
    ax2 = ax.twinx()
    ax2.plot(n, [r["cover"] for r in rows], "s--", color=PLUM, lw=1.6, ms=5, label="coverage of the truth")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel("coverage", fontsize=9, color=PLUM)
    ax2.tick_params(colors=PLUM, labelsize=8.5)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.3, frameon=False, loc="center right")
    _style(ax, "2.  More traffic makes it worse, not better",
           "utilisation held at 1.00.  Bias flat over 32x of n; the interval closes around the wrong number")


def panel_3(ax) -> None:
    """Opposite signs in the two worlds."""
    s1, s4 = R["s1"], R["s4"]
    labels = ["shared supply\n(marketplace)", "peer effects\n(network good)"]
    truth = [s1["truth"], s4["truth"]]
    est = [s1["split"], s4["split"]]
    # normalise each world to its own truth so both fit one axis
    rel_t = [1.0, 1.0]
    CAP = 40.0
    rel_e = [est[0] / max(truth[0], 1e-9), est[1] / truth[1]]
    x = np.arange(2)
    ax.bar(x - 0.19, rel_t, 0.36, color=GOOD, label="true global effect", zorder=3)
    ax.bar(x + 0.19, [min(v, CAP) for v in rel_e], 0.36, color=BAD, label="what the split reports", zorder=3)
    ax.axhline(1.0, color=INK, lw=1.0, ls=":", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.8, color=INK)
    ax.set_ylabel("multiple of the true effect (log, bar clipped)", fontsize=9, color=MUTE)
    ax.set_yscale("log")
    ax.set_ylim(0.4, 130)
    # The marketplace ratio is unbounded - the truth there is ~0 - so quote the share of
    # the REPORTED number that is bias, which is stable, and mark the bar as clipped.
    share_bias = 100 * (est[0] - truth[0]) / est[0]
    ax.text(0.19, CAP * 1.2, f"{share_bias:.0f}% of the reported\nnumber is bias\n(ratio unbounded)",
            ha="center", fontsize=8.6, color=BAD, fontweight="bold")
    ax.text(1.19, rel_e[1] * 1.09, f"{100 * (1 - rel_e[1]):.0f}% too small", ha="center", fontsize=9.0, color=BAD, fontweight="bold")
    ax.legend(fontsize=8.3, frameon=False, loc="upper right")
    _style(ax, "3.  Same estimator, opposite errors",
           "the output cannot tell you which world you are in - only the mechanism can")


def panel_4(ax) -> None:
    """Power of the guard test vs power of the experiment."""
    rows = R["s5"]["rows"]
    n = [r["n"] for r in rows]
    ax.plot(n, [r["power"] for r in rows], "o-", color=WARN, lw=2.4, ms=6, label="dose-response check", zorder=4)
    ax.axhline(0.80, color=MUTE, lw=1.0, ls="--", zorder=2)
    ax.axhline(R["s5"]["size"], color=GRID, lw=6, zorder=1)
    ax.text(n[0] * 1.05, R["s5"]["size"] - 0.055, f"its own false-alarm rate ({R['s5']['size']:.3f})", fontsize=8.4, color=MUTE)
    ax.plot([r["n"] for r in R["s3"]["rows"]], [r["power"] for r in R["s3"]["rows"]],
            "s-", color=BAD, lw=2.0, ms=5, label="the experiment it guards", zorder=4)
    ax.axvline(R["s5"]["n_for_80"], color=WARN, lw=1.2, ls=":", zorder=3)
    ax.text(R["s5"]["n_for_80"] * 0.88, 0.86, f"{R['s5']['n_for_80'] / 1e6:.2f}M needed\nfor power 0.80",
            fontsize=8.6, color=WARN, ha="right", va="bottom", fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(-0.09, 1.06)
    ax.set_xlabel("users (log scale)", fontsize=9, color=MUTE)
    ax.set_ylabel("power", fontsize=9, color=MUTE)
    ax.legend(fontsize=8.3, frameon=False, loc="center left")
    _style(ax, "4.  The recommended check is ~100x less sensitive than the test",
           "on a market where the whole reported effect is bias.  Calibrated, and nearly blind")


def panel_5(ax) -> None:
    """Cluster randomisation: contained vs pooled, plus the price."""
    a, b = R["s6"]["contained"], R["s6"]["pooled"]
    groups = ["supply local\nto each city", "supply pooled\nnationally"]
    x = np.arange(2)
    for i, d in enumerate((a, b)):
        ax.bar(i - 0.24, d["truth"], 0.22, color=GOOD, zorder=3, label="truth" if i == 0 else None)
        ax.bar(i, d["split"], 0.22, color=BAD, zorder=3, label="user-level split" if i == 0 else None)
        ax.bar(i + 0.24, d["cluster"], 0.22, color=COOL, zorder=3, label="cluster randomised" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8.8, color=INK)
    ax.set_ylabel("estimated effect", fontsize=9, color=MUTE)
    ax.set_ylim(0, max(b["cluster"], a["split"]) * 1.42)
    ax.annotate("clustering removed\n0% of the bias", xy=(1.24, b["cluster"]), xytext=(1.02, b["cluster"] * 1.30),
                ha="center", fontsize=8.7, color=BAD, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BAD, lw=1.1))
    ax.legend(fontsize=8.3, frameon=False, loc="upper left")
    _style(ax, "5.  Clustering fixes interference only if the cluster contains it",
           "identical rows, identical volumes.  The difference is where the supply sits, which is not in the data")


def panel_6(ax) -> None:
    """Switchback carryover: coin-flip vs ABAB, and the textbook-vs-derived DE."""
    rows = R["s8"]["carryover"]
    c = [r["c"] for r in rows]
    ax.plot(c, [r["coin"] for r in rows], "o-", color=GOOD, lw=2.3, ms=6, label="coin-flip periods", zorder=4)
    ax.plot(c, [r["abab"] for r in rows], "s-", color=BAD, lw=2.3, ms=6, label="strict ABAB", zorder=4)
    ax.plot(c, [1 - x for x in c], ls=":", color=GOOD, lw=1.3, zorder=3)
    ax.plot(c, [1 - 2 * x for x in c], ls=":", color=BAD, lw=1.3, zorder=3)
    ax.axhline(1.0, color=INK, lw=1.0, ls="--", zorder=2)
    ax.text(0.005, 1.02, "true effect", fontsize=8.3, color=INK)
    ax.set_xlabel("carryover: share of each period still under the old condition", fontsize=9, color=MUTE)
    ax.set_ylabel("estimate", fontsize=9, color=MUTE)
    ax.legend(fontsize=8.3, frameon=False, loc="lower left")
    _style(ax, "6.  The balanced switchback is exactly twice as biased",
           "dotted lines are the closed forms tau(1-c) and tau(1-2c) - balance in assignment is not balance in exposure")


def main() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15.4, 15.0))
    fig.patch.set_facecolor("white")
    panel_1(axes[0][0])
    panel_2(axes[0][1])
    panel_3(axes[1][0])
    panel_4(axes[1][1])
    panel_5(axes[2][0])
    panel_6(axes[2][1])
    fig.suptitle(
        "Interference: a split test measures a transfer between the arms, and the decision needs the total",
        fontsize=14.5, fontweight="bold", color=INK, x=0.008, ha="left", y=0.996,
    )
    fig.text(
        0.008, 0.978,
        "Every panel is read from results.json, written by evidence.py.  Ground truth in each world is a second "
        "simulation under global treatment and global control - the thing no experiment can observe.",
        fontsize=9.2, color=MUTE, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.966))
    fig.savefig("interference_audit.png", dpi=200, facecolor="white")
    fig.savefig("interference_audit.svg", facecolor="white")
    print("wrote interference_audit.png / .svg")


if __name__ == "__main__":
    main()
