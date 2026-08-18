"""Figures for the percentage audit. Every value is read from the engine.

`percent_audit.png` - six panels, the README hero
`percent_demo.png`  - two panels, the notebook's chart
"""

from __future__ import annotations


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from percentages import (
    CORPUS,
    COMMITTEE,
    COUNCIL,
    METHODS,
    QUEUES,
    SEVERITY_OF,
    SHIFTS,
    audit_corpus,
    largest_remainder,
    naive_half_up,
    no_method_is_clean,
    seat_table,
)

INK = "#141414"
MUTED = "#8a8a8a"
GRID = "#e4e2dd"
PAPER = "#faf8f4"
RED = "#c0392b"
ORANGE = "#d98324"
BLUE = "#4a7c8c"
GREEN = "#4b7f52"
PURPLE = "#7a5a8c"
SEV_COLOR = {"blocking": RED, "silent": ORANGE, "advisory": BLUE}
VERDICT_COLOR = {"consistent": GREEN, "residual": ORANGE, "contested": RED, "undefined": MUTED}

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

REP = audit_corpus()
DEFINED = [t for t in CORPUS if t.is_definable()[0]]


def title(ax, n: int, text: str, sub: str = "") -> None:
    ax.set_title(f"{n}.  {text}", loc="left", fontsize=10, fontweight="bold",
                 pad=22 if sub else 6)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=7.6, color=MUTED, va="bottom")


def panel_drift(ax) -> None:
    rows = []
    for t in DEFINED:
        a = naive_half_up(t)
        rows.append((t.name, a.total - t.units, t.kind))
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows],
            color=[GREEN if r[1] == 0 else ORANGE for r in rows], height=0.66)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7, fontfamily="DejaVu Sans Mono")
    ax.set_xlabel("units the independently rounded column is off by")
    for yi, (_, d, _) in zip(y, rows):
        if d:
            ax.text(d + (0.06 if d > 0 else -0.06), yi, f"{d:+d}", va="center",
                    ha="left" if d > 0 else "right", fontsize=6.8, color=INK)
        else:
            # A zero-length bar is invisible, and "the column added up" is the
            # result worth seeing most.
            ax.plot([0], [yi], marker="o", ms=4.5, color=GREEN, lw=0)
            ax.text(0.1, yi, "added up", va="center", fontsize=6.6, color=GREEN)
    ax.set_xlim(-3.2, 2.2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    title(ax, 1, "Round every row correctly and the column still misses",
          "0 means the column added up; one unit is 0.1 of a point at 1 dp")


def panel_council(ax) -> None:
    names = list(METHODS)
    labels = list(COUNCIL.labels)
    quotas = [float(q) for q in COUNCIL.quotas()]
    x = np.arange(len(labels))
    for i, name in enumerate(names):
        al = METHODS[name](COUNCIL)
        jitter = (i - len(names) / 2) * 0.055
        ok = al.sums_to(COUNCIL)
        ax.plot(x + jitter, al.units, marker="o", ms=5, lw=0,
                mfc="none" if ok else RED, color=INK if ok else RED, zorder=3)
    ax.plot(x, quotas, marker="_", ms=26, lw=0, color=BLUE, zorder=2)
    for xi, q in zip(x, quotas):
        ax.add_patch(plt.Rectangle((xi - 0.34, np.floor(q)), 0.68, max(np.ceil(q) - np.floor(q), 0.001),
                                   facecolor=BLUE, alpha=0.10, edgecolor=BLUE, lw=0.6, zorder=0))
        ax.text(xi - 0.37, q, f"{q:.2f}", fontsize=6.6, color=BLUE, ha="right", va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("seats of 9")
    ax.set_yticks(range(0, 7))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Patch(facecolor="none", edgecolor=INK, label="one method's award"),
        Patch(facecolor=BLUE, alpha=0.2, label="floor-to-ceiling of the exact share"),
    ], loc="upper right", frameon=False, fontsize=7)
    title(ax, 2, "Nine seats, five parties, nine methods",
          "blue is awarded 3, 4 or 5 seats on the same votes; a dot outside the band left the quota")


def panel_alabama(ax) -> None:
    before = largest_remainder(COMMITTEE)
    bigger = seat_table(COMMITTEE.name, [(r.label, r.value) for r in COMMITTEE.rows],
                        COMMITTEE.units + 1)
    after = largest_remainder(bigger)
    labels = list(COMMITTEE.labels)
    x = np.arange(len(labels))
    ax.bar(x - 0.19, before.units, width=0.36, color=BLUE, label="7 seats to share")
    ax.bar(x + 0.19, after.units, width=0.36, color=ORANGE, label="8 seats to share")
    for xi, (b, a) in enumerate(zip(before.units, after.units)):
        if a < b:
            ax.annotate("", xy=(xi + 0.19, a + 0.06), xytext=(xi - 0.19, b + 0.06),
                        arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
            ax.text(xi, b + 0.35, "loses its only seat", color=RED, fontsize=7.4, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("seats")
    ax.set_ylim(0, max(after.units) + 1.4)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    title(ax, 3, "The Alabama paradox: more to share, less for one row",
          "largest remainder, headcounts 22 / 39 / 4, nothing about legal changed")


def panel_scoreboard(ax) -> None:
    board = no_method_is_clean()
    names = list(board)
    y = np.arange(len(names))
    sums = [board[n][0] for n in names]
    quota = [board[n][1] for n in names]
    alabama_hits = [board[n][2] for n in names]
    ax.barh(y - 0.24, sums, height=0.22, color=ORANGE, label="tables where the column fails to sum")
    ax.barh(y, quota, height=0.22, color=RED, label="tables with a quota violation")
    ax.barh(y + 0.24, alabama_hits, height=0.22, color=PURPLE, label="tables with the Alabama paradox")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7, fontfamily="DejaVu Sans Mono")
    ax.invert_yaxis()
    ax.set_xlabel(f"tables of the {len(DEFINED)} that have a share at all")
    ax.set_xlim(0, 11)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    title(ax, 4, "No method has an empty row",
          "Balinski-Young 1982: quota and paradox-freedom cannot both hold. Every method pays somewhere")


def panel_quota(ax) -> None:
    tables = [COUNCIL, QUEUES, SHIFTS]
    methods = ["largest_remainder", "jefferson_dhondt", "webster_sainte_lague",
               "adams", "huntington_hill"]
    colors = dict(zip(methods, [GREEN, RED, BLUE, PURPLE, ORANGE]))
    ticks, tick_labels = [], []
    pos = 0.0
    for t in tables:
        quotas = [float(q) for q in t.quotas()]
        for i, label in enumerate(t.labels):
            ticks.append(pos)
            tick_labels.append(f"{t.name.split('-')[0]}:{label}")
            for m in methods:
                al = METHODS[m](t)
                if not al.sums_to(t):
                    continue
                dev = al.units[i] - quotas[i]
                ax.plot([dev], [pos], marker="o", ms=5, color=colors[m], lw=0, alpha=0.85)
            pos += 1
        pos += 0.6
    ax.axvspan(-1, 1, color=BLUE, alpha=0.10, lw=0)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=6.4, fontfamily="DejaVu Sans Mono")
    ax.invert_yaxis()
    ax.set_xlabel("units awarded minus the exact share")
    ax.set_xlim(-2.2, 2.2)
    ax.legend(handles=[Patch(facecolor=colors[m], label=m) for m in methods],
              loc="lower right", frameon=False, fontsize=6.6)
    ax.text(0.985, 0.02, "outside the band = quota violation", transform=ax.transAxes,
            fontsize=6.8, color=BLUE, ha="right", va="bottom")
    title(ax, 5, "How far each method strays from the exact share",
          "the shaded band is floor-to-ceiling: staying inside it is the fairness a reader assumes")


def panel_findings(ax) -> None:
    rows = sorted(((c, n, SEVERITY_OF[c]) for c, n in REP.finding_counts.items()),
                  key=lambda r: (["blocking", "silent", "advisory"].index(r[2]), -r[1]))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[SEV_COLOR[r[2]] for r in rows], height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.2, fontfamily="DejaVu Sans Mono")
    ax.invert_yaxis()
    ax.set_xlabel("times the mechanism fires across the corpus")
    for yi, (_, n, _) in zip(y, rows):
        ax.text(n + 0.5, yi, str(n), va="center", fontsize=6.4, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=SEV_COLOR[k], label=k) for k in
                       ("blocking", "silent", "advisory")],
              loc="lower right", frameon=False, fontsize=7)
    title(ax, 6, "Twenty mechanisms, every one with evidence",
          "silent = the table renders, sums to 100%, and is defensibly wrong")


def build_audit() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 15.8))
    panel_drift(axes[0][0])
    panel_council(axes[0][1])
    panel_alabama(axes[1][0])
    panel_scoreboard(axes[1][1])
    panel_quota(axes[2][0])
    panel_findings(axes[2][1])
    fig.suptitle(
        "A percentage column is an apportionment, and apportionment has a proved impossibility",
        x=0.012, y=0.995, ha="left", fontsize=14, fontweight="bold",
    )
    fig.text(
        0.012, 0.9755,
        f"Day 148 - Percent Recomputer.  {REP.verdicts['consistent']} consistent, "
        f"{REP.verdicts['residual']} residual, {REP.verdicts['contested']} contested, "
        f"{REP.verdicts['undefined']} undefined across {REP.total} tables. "
        f"Nine methods, and not one of them is clean.",
        fontsize=9, color=MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.968))
    fig.savefig("percent_audit.png", dpi=170)
    print("wrote percent_audit.png")


def build_demo() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2))
    panel_council(ax1)
    panel_scoreboard(ax2)
    fig.suptitle("One table, nine defensible answers - and no method without a witness against it",
                 x=0.012, y=0.99, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("percent_demo.png", dpi=165)
    print("wrote percent_demo.png")


if __name__ == "__main__":
    build_audit()
    build_demo()
