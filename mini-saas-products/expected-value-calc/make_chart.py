"""Six panels, every number computed live from evcalc.

    python make_chart.py  ->  ev_audit.png (300 DPI) + .svg
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import evcalc as E

INK, MUTED, GRIDC, PAPER = "#1d1a17", "#8a8178", "#e3ddd5", "#faf7f2"
ACCENT, COOL, WARM, GREEN = "#c8553d", "#2f6f8f", "#e0a458", "#4f7942"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "axes.edgecolor": GRIDC,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9.5, "axes.titleweight": "bold",
})


def strip(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=0)


def k(x, _pos=None):
    return f"{x / 1000:,.0f}k"


def panel_distributions(ax):
    """The spreadsheet number against the distribution it came from."""
    sims = E.simulate()
    naive = E.naive_point_estimate()
    ev = E.true_expected_value()
    for opt, colour in (("build", ACCENT), ("buy", COOL)):
        ax.hist(sims[opt], bins=140, alpha=0.5, color=colour, label=opt, density=True)
        ax.axvline(ev[opt], color=colour, lw=1.8)
        ax.axvline(naive[opt], color=colour, lw=1.2, ls=":")
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_xlim(-600_000, 900_000)
    ax.xaxis.set_major_formatter(k)
    ax.set_yticks([])
    ax.set_xlabel("three-year net value")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    p = E.probability_of_the_point_estimate()
    strip(ax, keep=("bottom",))
    ax.set_title("1  ·  solid = expected value, dotted = the typed estimate\n"
                 f"the typed number happens {p['build']:.1%} of the time for build, "
                 f"{p['buy']:.1%} for buy", loc="left")


def panel_wins(ax):
    """Higher expected value, lower chance of winning."""
    ev = E.true_expected_value()
    p_win = {"build": E.beats("build", "buy"), "buy": E.beats("buy", "build")}
    opts = ["build", "buy"]
    x = np.arange(2)
    ax2 = ax.twinx()
    ax.bar(x - 0.2, [ev[o] for o in opts], 0.38, color=COOL, label="expected value")
    ax2.bar(x + 0.2, [p_win[o] for o in opts], 0.38, color=ACCENT,
            label="P(beats the other)")
    ax2.axhline(0.5, color=INK, lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(opts)
    ax.yaxis.set_major_formatter(k)
    ax.set_ylabel("expected value", color=COOL)
    ax2.set_ylabel("P(beats the other)", color=ACCENT)
    ax2.set_ylim(0, 1)
    for i, o in enumerate(opts):
        ax2.text(i + 0.2, p_win[o] + 0.02, f"{p_win[o]:.1%}", ha="center",
                 fontsize=8, color=ACCENT, fontweight="bold")
    strip(ax, keep=("bottom",))
    for s in ("top", "left"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(length=0)
    ax.set_title("2  ·  build has the higher expected value\n"
                 "and wins less than half the time", loc="left")


def panel_tornado(ax):
    """Which input actually decides it."""
    rows = E.tornado()[::-1]
    y = np.arange(len(rows))
    for i, (name, lo, hi, _swing) in enumerate(rows):
        left, right = sorted((lo, hi))
        ax.barh(i, right - left, left=left, height=0.62,
                color=COOL if right - left > 200_000 else GRIDC)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.xaxis.set_major_formatter(k)
    ax.set_xlabel("build minus buy, when this input moves P10 to P90")
    strip(ax, keep=("bottom",))
    top, bottom = E.tornado()[0], E.tornado()[-1]
    ax.set_title(f"3  ·  {top[0]} swings it {top[3] / bottom[3]:.0f}x more than "
                 f"{bottom[0]}\nthe rate gets debated; adoption decides it", loc="left")


def panel_switching(ax):
    """What would have to be true."""
    names = [i.name for i in E.INPUTS]
    sp = E.switching_points()
    for i, name in enumerate(names):
        inp = E.INPUTS_BY_NAME[name]
        span = inp.high - inp.low
        ax.plot([0, 1], [i, i], color=GRIDC, lw=7, solid_capstyle="butt")
        mid_x = (inp.mid - inp.low) / span
        ax.plot([mid_x], [i], "o", color=INK, ms=7, zorder=3)
        if sp[name] is not None:
            sw = (sp[name] - inp.low) / span
            ax.plot([sw], [i], "|", color=ACCENT, ms=22, mew=3, zorder=4)
            ax.annotate(f"{sp[name]:,.1f}", (sw, i), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=7.5,
                        color=ACCENT, fontweight="bold")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["P10", "P90"], fontsize=8)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.6, len(names) - 0.4)
    strip(ax, keep=())
    ax.set_title("4  ·  black = the typed estimate, red = where the answer flips\n"
                 "two flip within a rounding error of the estimate itself", loc="left")


def panel_ergodic(ax):
    """Positive expected value, negative growth."""
    rng = np.random.default_rng(E.RNG_SEED + 11)
    rounds = 250
    for f, colour, lw in ((1.0, ACCENT, 1.0), (E.kelly_fraction(), GREEN, 1.0)):
        wins = rng.random((60, rounds)) < E.P_UP
        mult = np.where(wins, 1 + f * (E.UP - 1), 1 - f * (1 - E.DOWN))
        paths = np.cumprod(mult, axis=1)
        ax.plot(paths.T, color=colour, alpha=0.16, lw=lw)
        ax.plot(np.median(paths, axis=0), color=colour, lw=2.4,
                label=f"stake {f:.0%}  (median)")
    ax.plot(np.cumprod(np.full(rounds, E.ensemble_growth())), color=INK, lw=1.6,
            ls="--", label="what the average does")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e6)
    ax.set_xlabel("rounds")
    ax.set_ylabel("wealth, multiple of stake (log)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    strip(ax)
    ax.set_title("5  ·  the average rises and the runs fall\n"
                 f"ensemble {E.ensemble_growth():.3f} per round, "
                 f"a single run {E.time_average_growth():.3f}", loc="left")


def panel_information(ax):
    """What it is worth to find out first."""
    info = {k_: v for k_, v in E.information_value().items() if not k_.startswith("_")}
    order = sorted(info, key=lambda n: info[n])
    y = np.arange(len(order))
    ax.barh(y, [info[n] for n in order], color=GREEN, height=0.62)
    evpi = E.evpi()["evpi"]
    ax.axvline(evpi, color=INK, lw=1.4, ls="--")
    ax.text(evpi, len(order) - 0.35, f"  EVPI {evpi:,.0f}", fontsize=7.5, color=INK,
            va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8.5)
    for i, n in enumerate(order):
        ax.text(info[n] + evpi * 0.015, i, f"{info[n]:,.0f}", va="center",
                fontsize=7.5, color=MUTED)
    ax.xaxis.set_major_formatter(k)
    ax.set_xlabel("most a study of this input could be worth")
    ax.set_xlim(0, evpi * 1.15)
    strip(ax, keep=("bottom",))
    ax.set_title("6  ·  learning adoption is worth 64x learning the hourly rate\n"
                 "and the parts sum to more than the whole", loc="left")


def main() -> None:
    fig = plt.figure(figsize=(15.5, 14.5))
    gs = fig.add_gridspec(3, 2, hspace=0.46, wspace=0.26,
                          left=0.06, right=0.955, top=0.925, bottom=0.05)
    panel_distributions(fig.add_subplot(gs[0, 0]))
    panel_wins(fig.add_subplot(gs[0, 1]))
    panel_tornado(fig.add_subplot(gs[1, 0]))
    panel_switching(fig.add_subplot(gs[1, 1]))
    panel_ergodic(fig.add_subplot(gs[2, 0]))
    panel_information(fig.add_subplot(gs[2, 1]))

    c = E.ranking_conflict()
    fig.suptitle("Expected value is a number. It is not a decision.",
                 x=0.06, y=0.975, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.06, 0.951,
             f"Day 157 · build vs buy vs defer, four uncertain inputs · "
             f"build leads on expected value by "
             f"{c['ev']['build'] - c['ev']['buy']:,.0f} and beats buy "
             f"{c['pairs'][('build', 'buy')]:.1%} of the time",
             ha="left", fontsize=9, color=MUTED)

    fig.savefig("ev_audit.png", dpi=300, facecolor=PAPER)
    fig.savefig("ev_audit.svg", facecolor=PAPER)
    print("wrote ev_audit.png and ev_audit.svg")


if __name__ == "__main__":
    main()
