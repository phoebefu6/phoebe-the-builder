"""Six panels, every number computed live from declog.

    python make_chart.py  ->  decision_audit.png (300 DPI) + .svg
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import declog as D
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRIDC, PAPER = "#1d1a17", "#8a8178", "#e3ddd5", "#faf7f2"
ACCENT, COOL, WARM, GREEN = "#c8553d", "#2f6f8f", "#e0a458", "#4f7942"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "axes.edgecolor": GRIDC,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9.5, "axes.titleweight": "bold",
})

RULE_NAMES = [r.name for r in D.RULES]
F_NAMES = [f.name for f in D.FORECASTERS]


def strip(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=0)


def panel_propriety(ax):
    """What an optimising forecaster reports, against what they believe."""
    beliefs = np.linspace(0.02, 0.98, 97)
    ax.plot([0, 1], [0, 1], color=GRIDC, lw=6, zorder=0, label="honest (report = belief)")
    styles = {"brier": (COOL, "-"), "log": (GREEN, "--"), "spherical": (INK, ":"),
              "absolute": (ACCENT, "-"), "threshold_01": (WARM, "-."),
              "confidence_points": ("#7d4f9e", "-")}
    for r in D.RULES:
        ys = [D.optimal_report(r, float(p)) for p in beliefs]
        c, ls = styles[r.name]
        proper = D.propriety(r.name)[0]
        ax.plot(beliefs, ys, color=c, ls=ls, lw=2.2 if not proper else 1.4,
                label=f"{r.name}{'' if proper else '  (improper)'}")
    ax.set_xlabel("what the forecaster believes")
    ax.set_ylabel("what they should report to score best")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.04, 1.04)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    strip(ax)
    ax.set_title("1  ·  three rules pay your team to lie\n"
                 "anything off the grey diagonal is a rule rewarding a false report",
                 loc="left")


def panel_rankings(ax):
    """Same forecasters, six rules, different winners."""
    table = D.score_table()
    ys = np.arange(len(F_NAMES))
    for i, r in enumerate(D.RULES):
        order = D.ranking(r.name)
        pos = [order.index(f) for f in F_NAMES]
        proper = D.propriety(r.name)[0]
        ax.plot([i] * len(F_NAMES), pos, "o", ms=0)
        for f, p in zip(F_NAMES, pos):
            ax.text(i, p, f[:12], ha="center", va="center", fontsize=6,
                    color="white" if p == 0 else INK,
                    bbox=dict(boxstyle="round,pad=0.28", lw=0,
                              fc=(COOL if proper else ACCENT) if p == 0 else GRIDC))
    ax.set_xticks(range(len(D.RULES)))
    ax.set_xticklabels([f"{r.name}\n{'proper' if D.propriety(r.name)[0] else 'IMPROPER'}"
                        for r in D.RULES], fontsize=6.5)
    ax.set_yticks(ys)
    ordinals = ["1st", "2nd", "3rd", "4th", "5th", "6th"]
    ax.set_yticklabels(ordinals[: len(ys)], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, len(D.RULES) - 0.5)
    strip(ax, keep=())
    ax.set_title("2  ·  the rule does not measure the winner, it picks one\n"
                 "'average error' crowns the overconfident forecaster", loc="left")
    _ = table


def panel_murphy(ax):
    """Brier = reliability - resolution + uncertainty."""
    d = D.decompositions()
    order = sorted(F_NAMES, key=lambda f: d[f]["brier"])
    y = np.arange(len(order))
    rel = [d[f]["reliability"] for f in order]
    res = [d[f]["resolution"] for f in order]
    ax.barh(y + 0.19, rel, height=0.36, color=ACCENT, label="reliability (miscalibration)")
    ax.barh(y - 0.19, res, height=0.36, color=COOL, label="resolution (information)")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("lower is better  ←  reliability          resolution  →  higher is better")
    for i, f in enumerate(order):
        ax.text(max(rel[i], res[i]) + 0.002, i, f"Brier {d[f]['brier']:.3f}",
                va="center", fontsize=6.5, color=MUTED)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    strip(ax, keep=("bottom",))
    ax.set_title("3  ·  base_rate is perfectly calibrated and worthless\n"
                 "zero reliability error, zero resolution, worst score", loc="left")


def panel_reliability(ax):
    """The calibration curve for three shapes."""
    ax.plot([0, 1], [0, 1], color=GRIDC, lw=6, zorder=0)
    for name, colour in (("calibrated", COOL), ("overconfident", ACCENT),
                         ("underconfident", GREEN), ("base_rate", WARM)):
        xs, ys, ns = D.reliability_curve(name)
        ax.plot(xs, ys, "o-", color=colour, ms=4, lw=1.6, label=name)
    ax.set_xlabel("said it was this likely")
    ax.set_ylabel("it happened this often")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    strip(ax)
    ax.set_title("4  ·  the default human shape is flatter than the diagonal\n"
                 "says 90%, happens 75%; says 10%, happens 25%", loc="left")


def panel_resulting(ax):
    """Outcome-based review of a mixed portfolio."""
    p = D.resulting_portfolio()
    right = p["n"] - p["misjudged"]
    parts = [right, p["good_called_bad"], p["bad_called_good"]]
    labels = [f"verdict correct\n{right}", f"good decision\npunished\n{p['good_called_bad']}",
              f"bad decision\nrewarded\n{p['bad_called_good']}"]
    colours = [GRIDC, ACCENT, WARM]
    left = 0
    for v, lab, c in zip(parts, labels, colours):
        ax.barh([0], [v], left=left, color=c, height=0.5)
        ax.text(left + v / 2, 0, lab, ha="center", va="center", fontsize=7.5,
                color=INK if c is GRIDC else "white")
        left += v
    ax.set_xlim(0, p["n"])
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel(f"{p['n']} decisions, reviewed only on how they turned out")
    strip(ax, keep=("bottom",))
    ax.set_title(f"5  ·  outcome-based review is wrong {p['misjudged_rate']:.0%} of the time\n"
                 "which is why the record has to hold the prediction", loc="left")


def panel_power(ax):
    """How many decisions to separate two forecasters."""
    m = D.power_matrix()
    pairs = sorted(m.items(), key=lambda kv: kv[1])
    labels = [f"{a[:11]} vs {b[:11]}" for (a, b), _ in pairs]
    vals = [v for _, v in pairs]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=[COOL if v <= 260 else ACCENT for v in vals], height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.axvline(260, color=INK, lw=1.2, ls="--")
    ax.text(260, len(vals) - 0.2, "  260 = one a week for five years",
            fontsize=6.5, color=INK, va="top")
    ax.set_xlabel("decisions needed to tell the pair apart (log scale, 80% power)")
    strip(ax, keep=("bottom",))
    reach = sum(1 for v in vals if v <= 260)
    ax.set_title(f"6  ·  {reach} of {len(vals)} comparisons are reachable in five years\n"
                 "ranking your people is what the log supports least", loc="left")


def main() -> None:
    fig = plt.figure(figsize=(15.5, 14.5))
    gs = fig.add_gridspec(3, 2, hspace=0.46, wspace=0.26,
                          left=0.06, right=0.965, top=0.925, bottom=0.05)
    panel_propriety(fig.add_subplot(gs[0, 0]))
    panel_rankings(fig.add_subplot(gs[0, 1]))
    panel_murphy(fig.add_subplot(gs[1, 0]))
    panel_reliability(fig.add_subplot(gs[1, 1]))
    panel_resulting(fig.add_subplot(gs[2, 0]))
    panel_power(fig.add_subplot(gs[2, 1]))

    rep = D.resolvability_report()
    fig.suptitle("A decision log is an instrument, and an instrument has a scoring rule",
                 x=0.06, y=0.975, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.06, 0.951,
             f"Day 155 · {rep['n']} records, {rep['resolvable']} of them scoreable · "
             f"{len(D.RULES)} scoring rules, 3 of which pay your team to misreport · "
             f"{len(D.FORECASTERS)} forecasters over 4,000 events",
             ha="left", fontsize=9, color=MUTED)

    fig.savefig("decision_audit.png", dpi=300, facecolor=PAPER)
    fig.savefig("decision_audit.svg", facecolor=PAPER)
    print("wrote decision_audit.png and decision_audit.svg")


if __name__ == "__main__":
    main()
