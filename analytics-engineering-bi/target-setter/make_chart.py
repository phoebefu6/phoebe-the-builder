"""Figures for the target audit. Every value is read from the engine.

``target_audit.png`` / ``.svg`` - six panels, the README hero.

The notebook draws its own two-panel figure inline rather than importing this
module, so there is exactly one place each figure is defined.
"""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import targets as T

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

ORIGIN_REF = 120
SERIES = T.make_history()
N_CHART_PATHS = 300

# Colour by what kind of claim the target is, not by method name.
KIND = {
    "seasonal_naive": "history",
    "last_quarter": "history",
    "run_rate": "history",
    "capacity": "resourcing",
    "trend_seasonal_median": "forecast",
    "trend_seasonal": "forecast",
    "trend_ols": "forecast",
    "benchmark": "outside",
    "yoy_growth": "forecast",
    "split_difference": "wanting",
    "top_down": "wanting",
    "stretch_best_ever": "wanting",
}
KIND_COLOR = {
    "history": BLUE,
    "forecast": GREEN,
    "resourcing": PURPLE,
    "outside": MUTED,
    "wanting": RED,
}


def _mp():
    return T.multipath(N_CHART_PATHS, 31_000)


def _spread(values: np.ndarray, min_gap: float) -> np.ndarray:
    """Push label positions apart so a dense cluster stays readable.

    A greedy two-pass spacer: sort, push up where two labels are closer than
    ``min_gap``, then push back down from the top so the block stays centred
    on the values it labels.
    """
    order = np.argsort(values)
    out = values.astype(float).copy()
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        if out[j] - out[i] < min_gap:
            out[j] = out[i] + min_gap
    for k in range(len(order) - 2, -1, -1):
        i, j = order[k], order[k + 1]
        if out[j] - out[i] < min_gap:
            out[i] = out[j] - min_gap
    return out


def panel_fan(ax) -> None:
    """The series, and the twelve targets fanning out of one origin."""
    lo_m = ORIGIN_REF - 30
    t = np.arange(lo_m, ORIGIN_REF)
    ax.plot(t, SERIES[lo_m:ORIGIN_REF], color=INK, lw=1.2, zorder=3)
    ax.plot(
        np.arange(ORIGIN_REF, ORIGIN_REF + T.HORIZON),
        SERIES[ORIGIN_REF : ORIGIN_REF + T.HORIZON],
        color=INK, lw=1.2, ls=":", zorder=3,
    )
    tg = T.targets_at(SERIES, ORIGIN_REF)
    # A target is for the quarter; show it as the flat monthly average.
    x = ORIGIN_REF + np.array([0.0, T.HORIZON - 1])
    names = sorted(tg, key=lambda k: tg[k])
    ys = np.array([tg[n] / T.HORIZON for n in names])
    span = ys.max() - ys.min()
    labels = _spread(ys, span * 0.085)
    for name, y, ly in zip(names, ys, labels):
        ax.plot(x, [y] * 2, color=KIND_COLOR[KIND[name]],
                lw=2.0, solid_capstyle="butt", alpha=0.9)
        ax.plot([x[1], x[1] + 1.6], [y, ly], color=KIND_COLOR[KIND[name]],
                lw=0.6, alpha=0.6)
        ax.annotate(name, (x[1] + 1.9, ly), fontsize=6.4,
                    va="center", color=KIND_COLOR[KIND[name]])
    actual = SERIES[ORIGIN_REF : ORIGIN_REF + T.HORIZON].sum() / T.HORIZON
    ax.axhline(actual, color=INK, lw=0.8, ls="--", alpha=0.5)
    ax.annotate("actual", (lo_m + 0.5, actual), fontsize=6.5, va="bottom",
                color=INK)
    lo, hi = T.prediction_interval(SERIES, ORIGIN_REF, 0.80)
    ax.fill_between(x, lo / T.HORIZON, hi / T.HORIZON, color=ORANGE,
                    alpha=0.13, zorder=0)
    ax.set_xlim(lo_m, ORIGIN_REF + 13)
    ax.set_title("Twelve defensible targets for the same quarter\n"
                 "shaded band = 80% prediction interval", loc="left",
                 fontsize=9, color=INK)
    ax.set_ylabel("signups / month")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def panel_ambition_vs_hit(ax, mp) -> None:
    names = list(T.METHODS)
    amb = np.array([mp[n]["ambition"].mean() for n in names])
    hit = np.array([mp[n]["hit_rate"].mean() for n in names])
    inversions = [
        (i, j)
        for i, j in itertools.combinations(range(len(names)), 2)
        if (amb[i] - amb[j]) * (hit[i] - hit[j]) > 0
    ]
    for i, j in inversions:
        ax.plot(amb[[i, j]], hit[[i, j]], color=ORANGE, lw=1.0,
                alpha=0.7, zorder=1)
    labels = _spread(hit, 0.045)
    for i, n in enumerate(names):
        ax.scatter(amb[i], hit[i], s=34, color=KIND_COLOR[KIND[n]], zorder=3)
        ax.plot([amb[i], amb[i] + 0.006], [hit[i], labels[i]],
                color=MUTED, lw=0.5, alpha=0.6, zorder=2)
        ax.annotate(n, (amb[i] + 0.008, labels[i]), fontsize=6.2,
                    va="center", color=MUTED)
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=":")
    ax.set_xlim(0.83, 1.30)
    ax.set_xlabel("ambition  (target / truth)")
    ax.set_ylabel("hit rate")
    ax.set_title("The hit rate is a property of the method, not the team\n"
                 f"orange = {len(inversions)} pairs where the HARDER target "
                 "is met more often",
                 loc="left", fontsize=9, color=INK)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def panel_reproducibility(ax, mp) -> None:
    order = sorted(T.METHODS, key=lambda n: mp[n]["hit_rate"].std())
    data = [mp[n]["hit_rate"] for n in order]
    bp = ax.boxplot(data, vert=False, widths=0.62, showfliers=False,
                    patch_artist=True, medianprops={"color": INK, "lw": 1.1})
    for patch, n in zip(bp["boxes"], order):
        patch.set_facecolor(KIND_COLOR[KIND[n]])
        patch.set_alpha(0.45)
        patch.set_edgecolor(MUTED)
    for el in ("whiskers", "caps"):
        for line in bp[el]:
            line.set_color(MUTED)
    ax.set_yticklabels(order, fontsize=6.4)
    ax.set_xlabel("hit rate over 11 years")
    ax.set_title(f"Re-run the same 11 years {N_CHART_PATHS} times\n"
                 "the best-specified forecasts give the least stable verdict",
                 loc="left", fontsize=9, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def panel_by_month(ax) -> None:
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ors = T.origins(SERIES)
    truth = {o: T.truth_quarter(o)[0] for o in ors}
    width = 0.38
    for k, (name, colour) in enumerate(
        (("run_rate", BLUE), ("trend_seasonal", GREEN))
    ):
        fn = T.METHODS[name]
        vals = []
        for m in range(12):
            v = [fn(SERIES[:o], o) / truth[o] for o in ors if o % 12 == m]
            vals.append(float(np.mean(v)))
        ax.bar(np.arange(12) + (k - 0.5) * width, np.array(vals) - 1.0,
               width=width, bottom=1.0, color=colour, alpha=0.85, label=name)
    ax.axhline(1.0, color=INK, lw=0.9)
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names, fontsize=6.6)
    ax.set_ylabel("target / truth")
    ax.set_title("The planning calendar is a parameter of the target\n"
                 "same history, same method, different month asked in",
                 loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=6.8, ncol=2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def panel_reconciliation(ax, mp) -> None:
    ors = T.origins(SERIES)
    ref = np.array([T.m_trend_seasonal(SERIES[:o], o) for o in ors])
    rows = [
        ("capacity", PURPLE),
        ("split_difference", ORANGE),
        ("top_down", RED),
    ]
    for i, (name, colour) in enumerate(rows):
        vals = np.array([T.METHODS[name](SERIES[:o], o) for o in ors])
        signed = float((vals / ref - 1).mean())
        ax.barh(i, signed, height=0.5, color=colour, alpha=0.85)
        hit = mp[name]["hit_rate"].mean()
        ax.annotate(f"met {hit:.0%}",
                    (signed + (0.006 if signed > 0 else -0.006), i),
                    va="center", ha="left" if signed > 0 else "right",
                    fontsize=7.2, color=INK)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlabel("signed distance from the best forecast")
    ax.set_xlim(-0.10, 0.26)
    ax.set_title("The compromise is the closest to the forecast\n"
                 "and the one that gets missed - distance is two-sided",
                 loc="left", fontsize=9, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def panel_inside_noise(ax) -> None:
    lo, hi = T.prediction_interval(SERIES, ORIGIN_REF, 0.80)
    width = hi - lo
    tg = T.targets_at(SERIES, ORIGIN_REF)
    gaps = np.array([abs(a - b)
                     for a, b in itertools.combinations(tg.values(), 2)])
    inside = int((gaps < width).sum())
    bins = np.linspace(0, max(gaps.max(), width) * 1.05, 22)
    ax.hist(gaps[gaps < width], bins=bins, color=MUTED, alpha=0.55,
            label=f"inside the interval ({inside})")
    ax.hist(gaps[gaps >= width], bins=bins, color=RED, alpha=0.75,
            label=f"larger than it ({len(gaps) - inside})")
    ax.axvline(width, color=ORANGE, lw=1.6)
    ax.annotate(f"80% interval width\n{width:,.0f}", (width, ax.get_ylim()[1]),
                xytext=(-6, -6), textcoords="offset points", ha="right",
                va="top", fontsize=6.8, color=ORANGE)
    ax.set_xlabel("gap between two methods' targets")
    ax.set_ylabel("pairs")
    ax.set_title(f"{inside} of {len(gaps)} disagreements are smaller than\n"
                 "the uncertainty everyone is arguing inside",
                 loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=6.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def hero(path: str = "target_audit") -> None:
    mp = _mp()
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 14.2))
    panel_fan(axes[0][0])
    panel_ambition_vs_hit(axes[0][1], mp)
    panel_reproducibility(axes[1][0], mp)
    panel_by_month(axes[1][1])
    panel_reconciliation(axes[2][0], mp)
    panel_inside_noise(axes[2][1])
    fig.suptitle(
        "A target is a method plus a claim about the future  -  "
        "twelve methods, one history, 300 re-runs",
        fontsize=12.5, color=INK, x=0.02, ha="left", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.982))
    fig.savefig(f"{path}.png", dpi=200)
    fig.savefig(f"{path}.svg")
    plt.close(fig)


if __name__ == "__main__":
    hero()
    print("wrote target_audit.png and target_audit.svg")
