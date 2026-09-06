"""Two panels: how deep each job goes, and where roles overlap.

    python make_chart.py  ->  capability_map.png (300 DPI) + .svg

Replaces portfolio_burnup.png, which plotted builds against a one-a-day pace
line. That chart measured completion of a plan; these measure what is covered.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import capability as C
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRIDC, PAPER = "#1d1a17", "#8a8178", "#e3ddd5", "#faf7f2"
ACCENT, COOL, GREEN = "#c8553d", "#2f6f8f", "#4f7942"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "axes.edgecolor": GRIDC,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 10, "axes.titleweight": "bold",
})


def strip(ax, keep=("bottom",)):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=0)


def main() -> None:
    tools = C.load()
    grouped = C.by_task(tools)
    titles = C.task_titles(tools)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.0),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    order = sorted(C.TASK_ORDER, key=lambda t: len(grouped[t]))
    y = np.arange(len(order))
    ax1.barh(y, [len(grouped[t]) for t in order], color=COOL, height=0.7)
    ax1.set_yticks(y)
    ax1.set_yticklabels([titles.get(t, t) for t in order], fontsize=8.5)
    for i, t in enumerate(order):
        ax1.text(len(grouped[t]) + 0.3, i, str(len(grouped[t])),
                 va="center", fontsize=8, color=MUTED)
    ax1.set_xlabel("tools behind this job")
    strip(ax1)
    ax1.set_title("How deep each job goes\n"
                  "depth, not progress - a short bar is a few sharp tools", loc="left")

    names = [r[0] for r in C.ROLES]
    overlap = C.role_overlap(tools)
    sizes = {n: len(C.for_role(n, tools)) for n in names}
    grid = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                grid[i, j] = sizes[a]
            else:
                grid[i, j] = overlap.get((a, b), overlap.get((b, a), 0))
    im = ax2.imshow(grid, cmap="YlGnBu", aspect="auto")
    for i in range(len(names)):
        for j in range(len(names)):
            v = int(grid[i, j])
            ax2.text(j, i, str(v), ha="center", va="center", fontsize=8,
                     color="white" if v > np.nanmax(grid) * 0.55 else INK,
                     fontweight="bold" if i == j else "normal")
    ax2.set_xticks(range(len(names)))
    ax2.set_yticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=32, ha="right", fontsize=7.5)
    ax2.set_yticklabels(names, fontsize=7.5)
    cb = plt.colorbar(im, ax=ax2, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.outline.set_visible(False)
    strip(ax2, keep=())
    ax2.set_title("Where the handoffs are\n"
                  "diagonal = tools for that role, off-diagonal = shared between two",
                  loc="left")

    fig.suptitle("What this covers for a team",
                 x=0.045, y=0.972, ha="left", fontsize=15, fontweight="bold", color=INK)
    biggest = max(overlap.items(), key=lambda kv: kv[1])
    fig.text(0.045, 0.928,
             "Sourced from the generated catalog, organised by the job somebody arrived "
             f"with. Largest shared surface: {biggest[0][0]} and {biggest[0][1]}, "
             f"{biggest[1]} tools in common.",
             ha="left", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    fig.savefig("capability_map.png", dpi=300, facecolor=PAPER)
    fig.savefig("capability_map.svg", facecolor=PAPER)
    print("wrote capability_map.png and capability_map.svg")


if __name__ == "__main__":
    main()
