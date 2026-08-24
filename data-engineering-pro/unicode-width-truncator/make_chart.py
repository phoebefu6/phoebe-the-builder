"""Six panels, every number computed live from uwidth.

    python make_chart.py    ->  truncation_audit.png (300 DPI) + .svg
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import uwidth as U

INK = "#1d1a17"
MUTED = "#8a8178"
GRID = "#e3ddd5"
PAPER = "#faf7f2"
ACCENT = "#c8553d"
COOL = "#2f6f8f"
WARM = "#e0a458"
GREEN = "#4f7942"

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
})


def strip(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def panel_distinct(ax):
    cases = list(U.CORPUS)
    counts = [U.verdict_for(c).distinct_outputs for c in cases]
    order = np.argsort(counts)[::-1]
    names = [cases[i].name for i in order]
    vals = [counts[i] for i in order]
    colors = [GRID if v == 1 else (ACCENT if v >= 5 else COOL) for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, height=0.72)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("distinct strings returned by 10 truncators at one n")
    ax.set_title("One string, one n, this many answers", loc="left")
    ax.axvline(1, color=MUTED, lw=0.8, ls=":")
    ax.text(1.1, len(vals) - 0.6, "only ASCII\nlands here", fontsize=6, color=MUTED, va="bottom")
    strip(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def panel_units(ax):
    names = ["emoji-family", "cjk-bio", "devanagari", "emoji-run", "url"]
    units = ["bytes", "code points", "UTF-16 units", "graphemes (regex)", "columns"]
    data = np.array([[U.unit_spread(U.CASE_BY_NAME[n])[u] for u in units] for n in names])
    x = np.arange(len(names))
    w = 0.16
    palette = [ACCENT, COOL, WARM, GREEN, MUTED]
    for i, unit in enumerate(units):
        ax.bar(x + (i - 2) * w, data[:, i], w, label=unit, color=palette[i])
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.5, rotation=12, ha="right")
    ax.set_ylabel("length")
    ax.set_title("'Length' is five different numbers", loc="left")
    ax.legend(fontsize=5.6, frameon=False, ncol=2, loc="upper right")
    strip(ax)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def panel_flags(ax):
    census = U.flag_census()
    labels = list(census)
    vals = [census[k] for k in labels]
    colors = [ACCENT if k in ("boundary-split", "identity-change") else COOL for k in labels]
    ax.barh(range(len(vals)), vals, color=colors, height=0.6)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.12, i, str(v), va="center", fontsize=7, color=INK)
    ax.set_xlabel(f"cases affected (of {len(U.CORPUS)})")
    ax.set_title("What the cuts did", loc="left")
    strip(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def panel_sinks(ax):
    t_names = [t.name for t in U.TRUNCATORS]
    s_names = [s.name for s in U.SINKS]
    grid = np.zeros((len(t_names), len(s_names)))
    for case in U.CORPUS:
        for (t, s), ok in U.sink_matrix(case).items():
            if not ok:
                grid[t_names.index(t), s_names.index(s)] += 1
    grid = grid / len(U.CORPUS) * 100
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(s_names)))
    ax.set_xticklabels([s.replace("_", "\n") for s in s_names], fontsize=5.4)
    ax.set_yticks(range(len(t_names)))
    ax.set_yticklabels(t_names, fontsize=6)
    for i in range(len(t_names)):
        for j in range(len(s_names)):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=5.4,
                    color="white" if grid[i, j] > 55 else INK)
    failed, total = U.sink_failure_rate()
    ax.set_title(f"% of cases still over the limit  ({failed}/{total} runs)", loc="left")
    strip(ax, keep=())
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02).ax.tick_params(labelsize=5.5)


def panel_overflow(ax):
    rows = [
        (case.name, cut.bytes_out, case.n)
        for case in U.CORPUS
        for cut in U.cut_all(case).values()
        if cut.overflows_own_limit
    ]
    names = [r[0] for r in rows]
    got = [r[1] for r in rows]
    lim = [r[2] for r in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.19, lim, 0.38, label="byte limit asked for", color=COOL)
    ax.bar(x + 0.19, got, 0.38, label="bytes actually returned", color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6, rotation=18, ha="right")
    ax.set_ylabel("bytes")
    ax.set_title("The byte truncator that overshoots its own limit", loc="left")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    strip(ax)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def panel_safe(ax):
    audit = U.safe_truncate_audit()
    safe_ok = sum(1 for _, _, f, d, w in audit if f and d and w)
    roster_total = roster_ok = 0
    for case in U.CORPUS:
        cuts = U.cut_all(case)
        for sink in U.SINKS:
            cut = cuts[U.choose_truncator(sink.name)]
            roster_total += 1
            if (U.fits(sink, cut.text, case.n) and cut.dangling is None and cut.well_formed):
                roster_ok += 1
    naive_total = naive_ok = 0
    for case in U.CORPUS:
        cut = U.cut_all(case)["code_points"]
        for sink in U.SINKS:
            naive_total += 1
            if (U.fits(sink, cut.text, case.n) and cut.dangling is None and cut.well_formed):
                naive_ok += 1
    labels = ["s[:n]\n(code points)", "unit-matched\nroster truncator", "safe_truncate\n(unit + cluster + marker)"]
    vals = [naive_ok / naive_total * 100, roster_ok / roster_total * 100, safe_ok / len(audit) * 100]
    colors = [ACCENT, WARM, GREEN]
    ax.bar(labels, vals, color=colors, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=8, color=INK, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of 156 runs that fit, do not dangle, stay text")
    ax.set_title("Fits the sink, keeps the meaning", loc="left")
    ax.tick_params(axis="x", labelsize=6.5)
    strip(ax)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    panel_distinct(axes[0][0])
    panel_units(axes[0][1])
    panel_flags(axes[0][2])
    panel_sinks(axes[1][0])
    panel_overflow(axes[1][1])
    panel_safe(axes[1][2])
    distinct, total = U.distinct_output_count()
    fig.suptitle(
        f"Truncate to n: {len(U.CORPUS)} strings x {len(U.TRUNCATORS)} truncators = {total} cuts, "
        f"{distinct} distinct outputs",
        x=0.008, ha="left", fontsize=13, fontweight="bold",
    )
    fig.text(0.008, 0.945,
             "The integer n does not name a length. Every layer that reads it picks its own unit.",
             ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("truncation_audit.png", dpi=300, facecolor=PAPER)
    fig.savefig("truncation_audit.svg", facecolor=PAPER)
    print("wrote truncation_audit.png / .svg")


if __name__ == "__main__":
    main()
