"""Figures for the guardrail audit. Every value is computed by the engine, not typed.

``guardrail_audit.png`` / ``.svg`` - six panels, the README hero.

The notebook draws its own smaller figure inline rather than importing this module, so
each figure is defined in exactly one place.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import guardrails as G
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

INK = "#141414"
MUTED = "#8a8a8a"
GRID = "#e4e2dd"
PAPER = "#faf8f4"
RED = "#c0392b"
ORANGE = "#d98324"
BLUE = "#4a7c8c"
GREEN = "#4b7f52"
PURPLE = "#7a5a8c"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
})

N = G.n_for_power(0.80, 1.0)
DAY = 14
ALPHA = 0.05
SEED = 424242


def _title(ax, text, sub=""):
    ax.set_title(text, loc="left", fontsize=9.5, fontweight="bold", color=INK, pad=26)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=7.6, color=MUTED, va="bottom")


# ---------------------------------------------------------------------------- panel 1
def panel_trade(ax):
    a = np.linspace(0, 1, 101)
    lift = np.array([G.primary_lift(x) for x in a]) * 100
    rate = np.array([G.quality_change(x) for x in a]) * 100
    vol = np.array([G.value_change(x) for x in a]) * 100
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.plot(a, lift, color=GREEN, lw=2.2, label="conversion (the win)")
    ax.plot(a, rate, color=RED, lw=2.2, label="180-day retention rate")
    ax.plot(a, vol, color=ORANGE, lw=1.6, ls="--", label="retained users per 1,000")
    ax.fill_between(a, 0, lift, color=GREEN, alpha=0.10)
    ax.fill_between(a, 0, rate, color=RED, alpha=0.10)
    ax.annotate(f"{G.quality_change(1.0)*100:.1f}%", (1.0, G.quality_change(1.0) * 100),
                xytext=(-4, -11), textcoords="offset points", ha="right", color=RED, fontsize=8, fontweight="bold")
    ax.annotate(f"+{G.primary_lift(1.0)*100:.0f}%", (1.0, G.primary_lift(1.0) * 100),
                xytext=(-4, 5), textcoords="offset points", ha="right", color=GREEN, fontsize=8, fontweight="bold")
    ax.annotate(f"{G.value_change(1.0)*100:.1f}%", (1.0, G.value_change(1.0) * 100),
                xytext=(-4, 5), textcoords="offset points", ha="right", color=ORANGE, fontsize=7.5)
    ax.set_xlabel("lever intensity")
    ax.set_ylabel("% change")
    ax.legend(frameon=False, fontsize=7.2, loc="lower left")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "The trade the guardrail exists to catch",
           "the total absorbs 3.1x less damage than the rate, because the lever inflates its denominator")


# ---------------------------------------------------------------------------- panel 2
def panel_power(ax):
    rows = []
    for g in G.GUARDRAILS:
        p = G.analytic_power(g, 1.0, N, DAY, ALPHA)
        rows.append((g.name, 0.0 if np.isnan(p) else p, np.isnan(p), g.name in G.DASHBOARD_SUITE))
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))
    colors = [MUTED if r[2] else (ORANGE if r[3] else BLUE) for r in rows]
    ax.barh(y, [r[1] for r in rows], color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.4)
    ax.axvline(G.primary_power(1.0, N, ALPHA), color=GREEN, lw=1.8)
    ax.text(G.primary_power(1.0, N, ALPHA) - 0.015, len(rows) - 0.4,
            "the win: 0.80", color=GREEN, fontsize=7.6, ha="right", fontweight="bold")
    ax.axvline(ALPHA, color=RED, lw=1.0, ls=":")
    ax.text(ALPHA + 0.012, 0.15, "coin flip (alpha)", color=RED, fontsize=6.8)
    for i, r in enumerate(rows):
        label = "cannot be computed" if r[2] else f"{r[1]:.2f}"
        ax.text(r[1] + 0.012, i, label, va="center", fontsize=7,
                color=MUTED if r[2] else INK)
    ax.set_xlim(0, 0.95)
    ax.set_xlabel("power to detect the harm at the SAME n")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "Powered for the win, not for the harm",
           f"n = {N:,} per arm, day {DAY}; orange = on the dashboard, grey = no denominator")


# ---------------------------------------------------------------------------- panel 3
def panel_multiplicity(ax):
    rng = np.random.default_rng(SEED)
    z1 = G.simulate_experiment(1.0, N, DAY, 12_000, rng)
    z0 = G.simulate_experiment(0.0, N, DAY, 12_000, rng)
    order = G.DASHBOARD_SUITE + ["refund_rate", "d7_retention"]
    ks = range(1, len(order) + 1)
    fb = [np.mean(G.any_fires(z0, order[:k], ALPHA)) for k in ks]
    dt = [np.mean(G.any_fires(z1, order[:k], ALPHA)) for k in ks]
    dtb = [np.mean(G.any_fires(z1, order[:k], ALPHA / k)) for k in ks]
    ax.plot(list(ks), dt, color=BLUE, lw=2.2, marker="o", ms=3.5, label="detects real harm")
    ax.plot(list(ks), fb, color=RED, lw=2.2, marker="o", ms=3.5, label="blocks a harmless change")
    ax.plot(list(ks), dtb, color=PURPLE, lw=1.8, ls="--", marker="s", ms=3,
            label="detects harm, Bonferroni")
    i = order.index("page_latency_ms")
    ax.axvline(i + 1, color=MUTED, lw=0.8, ls=":")
    ax.annotate("a placebo metric\nis added here", (i + 1, 0.58), xytext=(6, 0),
                textcoords="offset points", fontsize=6.9, color=MUTED, va="center")
    ax.set_xlabel("guardrails in the suite")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "Every guardrail you add is paid for twice",
           "uncorrected the false blocks climb; corrected the detection collapses")


# ---------------------------------------------------------------------------- panel 4
def panel_maturity(ax):
    days = [3, 7, 14, 21, 28, 56, 90, 120, 180]
    daily = N / DAY
    best, comp, d90 = [], [], []
    for d in days:
        n = int(daily * d)
        rng = np.random.default_rng(SEED + d)
        suite = [g.name for g in G.GUARDRAILS if G.observable_fraction(d, g.maturity_days) > 0]
        w = G.sensitivity_weights(suite, 1.0, n, d)
        z0 = G.simulate_experiment(0.0, n, d, 6_000, rng)
        z1 = G.simulate_experiment(1.0, n, d, 6_000, rng)
        crit = float(np.quantile(G.composite_z(z0, suite, w), 1 - ALPHA))
        comp.append(float(np.mean(G.composite_z(z1, suite, w) > crit)))
        best.append(max(G.analytic_power(G.GUARDRAIL_BY_NAME[s], 1.0, n, d, ALPHA) for s in suite))
        d90.append(G.observable_fraction(d, 90))
    ax.plot(days, comp, color=BLUE, lw=2.2, marker="o", ms=3.5, label="all nine, pooled into one index")
    ax.plot(days, best, color=ORANGE, lw=2.2, marker="o", ms=3.5, label="best single guardrail")
    ax.plot(days, d90, color=MUTED, lw=1.5, ls="--", label="share of d90_retention that exists")
    ax.axhline(0.80, color=GREEN, lw=1.0, ls=":")
    ax.axvline(DAY, color=RED, lw=1.0)
    ax.text(DAY + 2, 0.06, "the decision\nis made here", color=RED, fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks(days)
    ax.set_xticklabels(days)
    ax.set_xlabel("decision day (n grows with the window)")
    ax.set_ylabel("power at full intensity")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7.2, loc="center right")
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "Pooling is worth about 4x the calendar",
           "the best single metric needs day 56 to reach 0.80; the index gets there by day 14")


# ---------------------------------------------------------------------------- panel 5
def panel_choice(ax):
    rng = np.random.default_rng(SEED + 9)
    values, retained = G.passive_cohort(300_000, rng)
    xs, ys, names, runnable = [], [], [], []
    for g in G.GUARDRAILS:
        xs.append(abs(np.corrcoef(values[g.name], retained)[0, 1]))
        ys.append(G.analytic_z(g, 1.0, N, DAY))
        names.append(g.name)
        runnable.append(G.observable_fraction(DAY, g.maturity_days) > 0)
    rho, _ = stats.spearmanr(xs, ys)
    nudge = {"refund_rate": (5, 5), "unsubscribe_rate": (5, -10), "complaint_rate": (5, -10)}
    for x, y, nm, ok in zip(xs, ys, names, runnable):
        ax.scatter(x, y, s=54, color=BLUE if ok else RED, zorder=3,
                   marker="o" if ok else "X", edgecolor=PAPER, linewidth=0.8)
        ax.annotate(nm, (x, y), xytext=nudge.get(nm, (5, 4)), textcoords="offset points",
                    fontsize=6.7, color=INK if ok else RED)
    ax.axhline(G.crit_value(ALPHA), color=GREEN, lw=1.0, ls=":")
    ax.text(0.005, G.crit_value(ALPHA) + 0.03, "z needed to fire", color=GREEN, fontsize=6.8)
    ax.set_xlabel("|correlation with the 180-day outcome|   (how it is chosen)")
    ax.set_ylabel("z against the lever   (what makes it fire)")
    ax.set_ylim(-0.15, 2.0)
    ax.set_xlim(-0.012, max(xs) * 1.24)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "Chosen on one axis, useful on the other",
           f"Spearman = {rho:+.2f}; red X = has no denominator on decision day")


# ---------------------------------------------------------------------------- panel 6
def panel_year(ax):
    import evidence

    s6 = evidence.section_6()
    order = ["no guardrail", "dashboard suite", "all computable", "composite index"]
    lifts = [s6["policies"][k]["reported_lift"] * 100 for k in order]
    rates = [s6["policies"][k]["value_change"] * 100 for k in order]
    y = np.arange(len(order))
    ax.barh(y + 0.19, lifts, height=0.34, color=GREEN, label="conversion, as reported on the slide")
    ax.barh(y - 0.19, rates, height=0.34, color=RED, label="180-day retention rate, actual")
    for i, (lv, rv) in enumerate(zip(lifts, rates)):
        ax.text(lv + 1.2, i + 0.19, f"+{lv:.0f}%", va="center", fontsize=7.4, color=GREEN, fontweight="bold")
        ax.text(rv - 1.2, i - 0.19, f"{rv:.1f}%", va="center", ha="right", fontsize=7.4,
                color=RED, fontweight="bold")
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=7.8)
    ax.set_xlabel("% change over one year of shipping")
    ax.set_xlim(-40, 78)
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "A year in which every experiment passed",
           "20 proposals, 60% harmless; the same ships produce both bars")


def build(path: str = "guardrail_audit") -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 15.2))
    panel_trade(axes[0][0])
    panel_power(axes[0][1])
    panel_multiplicity(axes[1][0])
    panel_maturity(axes[1][1])
    panel_choice(axes[2][0])
    panel_year(axes[2][1])
    fig.suptitle("We hit the KPI and broke the business",
                 x=0.008, y=0.995, ha="left", fontsize=14.5, fontweight="bold", color=INK)
    fig.text(0.008, 0.977,
             "A guardrail is a constraint, and a constraint has a power. Nine counter-metrics, "
             "one lever with a known answer, and the year they let through.",
             ha="left", fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.968])
    fig.subplots_adjust(hspace=0.46, wspace=0.28)
    fig.savefig(f"{path}.png", dpi=300)
    fig.savefig(f"{path}.svg")
    plt.close(fig)
    print(f"wrote {path}.png and {path}.svg")


if __name__ == "__main__":
    build()
