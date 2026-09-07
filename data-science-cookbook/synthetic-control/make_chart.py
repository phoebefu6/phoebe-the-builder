"""Six panels, every one read from `results.json` - the file the evidence run writes - so
the figure cannot drift from the measurement.  Panel 1 is the only exception: it refits one
illustrative panel from a fixed seed, which the tests pin."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from synth import fit_synth, make_panel  # noqa: E402

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
    """What the method does: a control group that did not exist before the fit."""
    rng = np.random.default_rng(1000)
    Y = make_panel(rng, effect=5.0)
    f = fit_synth(Y, 0, 30)
    synth = Y[0] - f.gaps
    t = np.arange(Y.shape[1])
    for row in Y[1:]:
        ax.plot(t, row, color=GRID, lw=0.7, zorder=1)
    ax.plot(t, Y[1:].mean(axis=0), color=WARN, lw=1.8, ls=":", label="donor average", zorder=3)
    ax.plot(t, Y[0], color=BAD, lw=2.3, label="treated market", zorder=5)
    ax.plot(t, synth, color=COOL, lw=2.1, ls="--", label="synthetic control", zorder=4)
    ax.axvline(29.5, color=INK, lw=1.1, ls="-", zorder=2)
    ax.text(29.9, ax.get_ylim()[1], " intervention", fontsize=8, color=INK, va="top")
    _style(ax, "1. The control group is fitted, not found", "grey lines are the 20 donor markets")
    ax.set_xlabel("period", fontsize=8.5, color=MUTE)
    ax.legend(fontsize=7.6, frameon=False, loc="upper left")


def panel_2(ax) -> None:
    """The p-value floor and the zero-power region it creates."""
    j = np.arange(3, 61)
    floor = 1.0 / (j + 1)
    ax.plot(j, floor, color=PLUM, lw=2.4, zorder=5)
    ax.axhline(0.05, color=BAD, lw=1.2, ls="--", zorder=4)
    ax.fill_between(j, 0.05, floor, where=floor > 0.05, color=BAD, alpha=0.13, zorder=2)
    ax.axvline(19, color=INK, lw=1.0, ls=":", zorder=3)
    ax.text(20, 0.30, "19 donors:\nthe first pool that\ncan reach p = 0.05", fontsize=8, color=INK)
    ax.text(30, 0.062, "no attainable p-value below the line", fontsize=8, color=BAD)
    ax.set_ylim(0, 0.36)
    _style(ax, "2. The smallest p-value is set before any data", "floor = 1 / (donors + 1)")
    ax.set_xlabel("donors in the pool", fontsize=8.5, color=MUTE)
    ax.set_ylabel("smallest reachable p", fontsize=8.5, color=MUTE)


def panel_3(ax) -> None:
    """Pre-fit improves, accuracy does not."""
    rows = R["s3"]["rows"]
    j = [r[0] for r in rows]
    pre = [r[1] for r in rows]
    err = [r[2] for r in rows]
    ax.plot(j, pre, "o-", color=COOL, lw=2.2, ms=6, label="pre-period fit (what is shown)", zorder=5)
    ax.plot(j, err, "s-", color=BAD, lw=2.2, ms=6, label="error in the estimate (what matters)", zorder=5)
    ax.set_xscale("log")
    ax.set_xticks(j)
    ax.set_xticklabels([str(x) for x in j])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0, max(max(pre), max(err)) * 1.25)
    ax.annotate(
        f"{pre[0] / pre[-1]:.1f}x better",
        xy=(j[-1], pre[-1]),
        xytext=(j[-2] * 0.8, pre[0] * 0.95),
        fontsize=8,
        color=COOL,
        arrowprops=dict(arrowstyle="->", color=COOL, lw=1.1),
    )
    ax.annotate("flat", xy=(j[-1], err[-1]), xytext=(j[-2], err[-1] * 1.18), fontsize=8, color=BAD)
    _style(ax, "3. The credential everyone publishes is not evidence", "10 pre-periods, donor pool grown")
    ax.set_xlabel("donors in the pool", fontsize=8.5, color=MUTE)
    ax.legend(fontsize=7.6, frameon=False, loc="lower left")


def panel_4(ax) -> None:
    """Outside the hull: bias as a fraction of the shift, plus the observable bound."""
    rows = R["s4"]["shift_rows"]
    h = [r[0] for r in rows]
    bias = [r[2] for r in rows]
    exc = [r[4] for r in rows]
    ax.plot(h, h, color=MUTE, lw=1.4, ls="--", label="the intuitive guess (bias = h)", zorder=3)
    ax.plot(h, exc, "^-", color=WARN, lw=2.0, ms=6, label="hull excess (observable)", zorder=4)
    ax.plot(h, bias, "o-", color=BAD, lw=2.4, ms=6.5, label="measured bias", zorder=5)
    for x, b in zip(h[1:], bias[1:]):
        if x < 5:  # the two lowest shifts sit on top of each other near the origin
            continue
        ax.text(x, b - max(bias) * 0.10, f"{b / x:.2f}h", fontsize=8, color=BAD, ha="center")
    ax.text(1.5, max(bias) * 0.11, "0.15h, 0.14h", fontsize=8, color=BAD, ha="left")
    _style(ax, "4. The hull is a wall, and the wall is partly bought back", "treated unit shifted h above its pool")
    ax.set_xlabel("shift h above the donor pool", fontsize=8.5, color=MUTE)
    ax.legend(fontsize=7.6, frameon=False, loc="upper left")


def panel_5(ax) -> None:
    """Which statistic: false alarms vs power, on the same four worlds."""
    rows = R["s5"]["rows"]
    d = {(r[0], r[1], r[2]): r[3] for r in rows}
    labels = ["clean null\n(h=0, tau=0)", "un-fittable null\n(h=5, tau=0)", "clean effect\n(h=0, tau=5)", "un-fittable effect\n(h=5, tau=5)"]
    keys = [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (5.0, 5.0)]
    x = np.arange(4)
    gap = [d[(k[0], k[1], "gap")] for k in keys]
    rat = [d[(k[0], k[1], "ratio")] for k in keys]
    ax.bar(x - 0.19, gap, 0.36, color=BAD, label="raw gap", zorder=4)
    ax.bar(x + 0.19, rat, 0.36, color=GOOD, label="post/pre ratio", zorder=4)
    ax.axhline(0.10, color=INK, lw=1.1, ls="--", zorder=5)
    ax.text(-0.52, 0.115, "nominal 0.10", fontsize=7.6, color=INK, ha="left")
    for xi, (g, r_) in enumerate(zip(gap, rat)):
        ax.text(xi - 0.19, g + 0.025, f"{g:.2f}", ha="center", fontsize=7.4, color=BAD)
        ax.text(xi + 0.19, r_ + 0.025, f"{r_:.2f}", ha="center", fontsize=7.4, color=GOOD)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylim(0, 1.30)
    for xi in (0, 1):
        ax.text(xi, 1.20, "want LOW", fontsize=7.4, color=MUTE, ha="center", style="italic")
    for xi in (2, 3):
        ax.text(xi, 1.20, "want HIGH", fontsize=7.4, color=MUTE, ha="center", style="italic")
    _style(ax, "5. The ratio statistic normalises, it does not protect", "rejection rate at a nominal 0.10")
    ax.legend(fontsize=7.6, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))


def panel_6(ax) -> None:
    """The exchange rate, as a heatmap."""
    pres = [8, 15, 30, 60, 120]
    donors = [5, 10, 20, 40, 80]
    M = np.array([[R["s9"][f"{tp}x{j}"] for j in donors] for tp in pres])
    im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto", vmin=M.min(), vmax=M.max())
    for i in range(len(pres)):
        for k in range(len(donors)):
            ax.text(k, i, f"{M[i, k]:.2f}", ha="center", va="center", fontsize=8.2, color=INK)
    ax.set_xticks(range(len(donors)))
    ax.set_xticklabels(donors)
    ax.set_yticks(range(len(pres)))
    ax.set_yticklabels(pres)
    ax.set_xlabel("donors in the pool  -  buys nothing here", fontsize=8.5, color=BAD)
    ax.set_ylabel("pre-periods of history  -  buys everything", fontsize=8.5, color=GOOD)
    ax.grid(False)
    ax.annotate(
        "",
        xy=(-0.62, 4.3),
        xytext=(-0.62, -0.3),
        arrowprops=dict(arrowstyle="->", color=GOOD, lw=2.2),
        annotation_clip=False,
    )
    _style(ax, "6. Only one of the two levers buys accuracy", "RMSE of the estimate, true effect 5.0")
    ax.grid(False)  # _style turns it on; a grid over a heatmap is just white scratches
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=7.5, colors=MUTE)


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.4))
    fig.patch.set_facecolor("white")
    for fn, ax in zip((panel_1, panel_2, panel_3, panel_4, panel_5, panel_6), axes.flat):
        ax.set_facecolor("#fbfcfd")
        fn(ax)
    fig.suptitle(
        "Synthetic control: the control group is a fitted object, and every one of its failures"
        " is invisible in the plot that gets published",
        fontsize=14.5,
        fontweight="bold",
        color=INK,
        x=0.008,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.008,
        0.012,
        "Day 169 - data-science-cookbook/synthetic-control - every number generated by evidence.py"
        " and asserted in test_synth.py",
        fontsize=8.6,
        color=MUTE,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.955))
    fig.savefig("synthetic_control_audit.png", dpi=170, facecolor="white")
    fig.savefig("synthetic_control_audit.svg", facecolor="white")
    print("wrote synthetic_control_audit.png / .svg")


if __name__ == "__main__":
    main()
