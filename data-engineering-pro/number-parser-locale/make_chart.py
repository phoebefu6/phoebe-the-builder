"""Six panels: the whole finding in one image.

Saves eight-bit PNGs at 300 DPI (portfolio standard) plus SVG for print.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import numlocale as N

SEED = 20260821

# One fill per meaning, colour-blind safe, legend on every panel that needs one.
C_AGREE = "#2a7f62"    # same number as everyone else
C_DIFF = "#c2410c"     # a different number, silently
C_REFUSE = "#94a3b8"   # refused: loud, recoverable
C_ZERO = "#7c1d6f"     # a zero that was not in the file
C_INK = "#111827"
C_GRID = "#e5e7eb"
C_ACCENT = "#1d4ed8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#9ca3af",
    "axes.labelcolor": C_INK,
    "text.color": C_INK,
    "xtick.color": "#4b5563",
    "ytick.color": "#4b5563",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 110,
})

# 0 refused, 1 agreed-majority, 2 different value, 3 silent zero
CELL_CMAP = ListedColormap([C_REFUSE, C_AGREE, C_DIFF, C_ZERO])


def classify_cells() -> Tuple[np.ndarray, List[str], List[str]]:
    cases = N.corpus()
    table = N.read_all(cases)
    names = N.reader_names()
    grid = np.zeros((len(cases), len(names)))
    for i, c in enumerate(cases):
        row = table[c.name]
        # The modal finite reading is the reference; anything else is drift.
        counts: Dict[str, int] = {}
        for n in names:
            r = row[n]
            if r.is_finite:
                counts[str(r.value)] = counts.get(str(r.value), 0) + 1
        modal = max(counts, key=lambda k: counts[k]) if counts else None
        for j, n in enumerate(names):
            r = row[n]
            if not r.ok:
                grid[i, j] = 0
            elif r.is_finite and r.value == 0 and "silent 0" in r.note:
                grid[i, j] = 3
            elif r.value is not None and str(r.value) == modal:
                grid[i, j] = 1
            elif not r.is_finite:
                grid[i, j] = 1     # Infinity / NaN agreed across readers
            else:
                grid[i, j] = 2
    labels = [c.escaped()[:20] for c in cases]
    short = [n.replace("_strict", " S").replace("_loose", " L") for n in names]
    return grid, labels, short


def panel_matrix(ax) -> None:
    grid, labels, names = classify_cells()
    ax.imshow(grid, cmap=CELL_CMAP, vmin=0, vmax=3, aspect="auto",
              interpolation="nearest")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.5, family="DejaVu Sans Mono")
    ax.set_title("1. 35 strings x 15 readers = 525 readings", loc="left")
    ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    ax.legend(handles=[
        mpatches.Patch(color=C_AGREE, label="modal reading"),
        mpatches.Patch(color=C_DIFF, label="different number, no error"),
        mpatches.Patch(color=C_REFUSE, label="refused"),
        mpatches.Patch(color=C_ZERO, label="silent zero"),
    ], fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.145),
       ncol=4, frameon=False, handlelength=1.2, columnspacing=1.4)


def panel_ratios(ax) -> None:
    verds = [v for v in N.all_verdicts() if v.ratio is not None]
    verds.sort(key=lambda v: float(v.ratio))
    names = [v.case.escaped()[:18] for v in verds]
    vals = [float(v.ratio) for v in verds]
    colors = [C_DIFF if v >= 10 else C_ACCENT for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, height=0.72)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6, family="DejaVu Sans Mono")
    ax.set_xscale("log")
    ax.set_xlabel("largest reading / smallest reading (log)", fontsize=7)
    ax.axvline(10, color=C_INK, linestyle=":", linewidth=1)
    ax.text(11, 0.4, "10x", fontsize=6, color=C_INK)
    ax.grid(axis="x", color=C_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title("2. How far apart two conforming readings get", loc="left")
    ax.legend(handles=[
        mpatches.Patch(color=C_DIFF, label=">= 10x apart"),
        mpatches.Patch(color=C_ACCENT, label="< 10x apart"),
    ], fontsize=6, loc="lower right", frameon=False)


def panel_verdicts(ax) -> None:
    verds = N.all_verdicts()
    counts = {v: 0 for v in N.VERDICTS}
    for v in verds:
        counts[v.verdict] += 1
    keys = [k for k in N.VERDICTS if counts[k] > 0]
    vals = [counts[k] for k in keys]
    colors = {"agreed": C_AGREE, "accept-drift": C_ACCENT, "value-drift": C_DIFF,
              "magnitude-drift": C_DIFF, "silent-zero": C_ZERO,
              "sign-loss": C_ZERO, "sign-drift": C_ZERO,
              "rejected-by-all": C_REFUSE}
    ax.bar(range(len(keys)), vals, color=[colors[k] for k in keys], width=0.68)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.25, str(v), ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("strings", fontsize=7)
    ax.set_ylim(0, max(vals) + 2)
    ax.grid(axis="y", color=C_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title("3. Verdicts: 'agreed' is %d of %d" % (counts["agreed"], len(verds)),
                 loc="left")


def panel_border(ax) -> None:
    cross = N.crossings()
    locs = [loc for loc, _ in N.LOCALES]
    # 0 = all ok, 1 = some refused, 2 = SILENTLY WRONG somewhere
    grid = np.zeros((len(locs), len(locs)))
    for i, w in enumerate(locs):
        for j, r in enumerate(locs):
            cells = [c for c in cross if c.wrote == w and c.read == r]
            if any(c.status == "wrong" for c in cells):
                grid[i, j] = 2
            elif any(c.status == "error" for c in cells):
                grid[i, j] = 1
    ax.imshow(grid, cmap=ListedColormap([C_AGREE, C_REFUSE, C_DIFF]),
              vmin=0, vmax=2, aspect="equal", interpolation="nearest")
    for i, w in enumerate(locs):
        for j, r in enumerate(locs):
            cells = [c for c in cross if c.wrote == w and c.read == r]
            wrong = sum(1 for c in cells if c.status == "wrong")
            if wrong:
                ax.text(j, i, str(wrong), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(range(len(locs)))
    ax.set_xticklabels(locs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(locs)))
    ax.set_yticklabels(locs, fontsize=7)
    ax.set_xlabel("read by", fontsize=7)
    ax.set_ylabel("written by", fontsize=7)
    wrong_total = sum(1 for c in cross if c.status == "wrong")
    ax.set_title("4. Border crossing: %d of %d runs silently wrong" % (wrong_total, len(cross)),
                 loc="left")
    ax.legend(handles=[
        mpatches.Patch(color=C_AGREE, label="always correct"),
        mpatches.Patch(color=C_REFUSE, label="refused (recoverable)"),
        mpatches.Patch(color=C_DIFF, label="silently wrong (n shown)"),
    ], fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.20),
       ncol=3, frameon=False, handlelength=1.2)


def panel_diagonal(ax) -> None:
    diag = N.own_output_roundtrip()
    locs = [loc for loc, _ in N.LOCALES]
    ok_s, bad_s, ok_l, bad_l = [], [], [], []
    for loc in locs:
        cs = [c for c in diag if c.read == loc and c.strict]
        cl = [c for c in diag if c.read == loc and not c.strict]
        ok_s.append(sum(1 for c in cs if c.status == "ok"))
        bad_s.append(sum(1 for c in cs if c.status != "ok"))
        ok_l.append(sum(1 for c in cl if c.status == "ok"))
        bad_l.append(sum(1 for c in cl if c.status != "ok"))
    x = np.arange(len(locs))
    w = 0.38
    ax.bar(x - w / 2, ok_s, w, color=C_AGREE, label="strict: read own output")
    ax.bar(x - w / 2, bad_s, w, bottom=ok_s, color=C_DIFF,
           label="strict: REFUSED own output")
    ax.bar(x + w / 2, ok_l, w, color=C_AGREE, alpha=0.45)
    ax.bar(x + w / 2, bad_l, w, bottom=ok_l, color=C_DIFF, alpha=0.45)
    for i in range(len(locs)):
        if bad_s[i]:
            ax.text(x[i] - w / 2, ok_s[i] + bad_s[i] + 0.08, str(bad_s[i]),
                    ha="center", fontsize=8, fontweight="bold", color=C_DIFF)
    ax.set_xticks(x)
    ax.set_xticklabels(locs, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("amounts (of 5)", fontsize=7)
    ax.grid(axis="y", color=C_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    nbad = sum(1 for c in diag if c.status != "ok")
    ax.set_title("5. Same locale wrote and read it: %d of %d fail" % (nbad, len(diag)),
                 loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left",
              title="left bar = strict, right (faded) = loose",
              title_fontsize=6.5)


def panel_decide(ax) -> None:
    cols = [
        ("three-digit\ngroups only", ["1.234", "2.500", "3.000", "1.750"]),
        ("group count\n> 1", ["1.234.567", "89.012", "3.456"]),
        ("four-digit\ngroup", ["1.2345", "2.500"]),
        ("both\nseparators", ["1.234,56", "7.890,12"]),
        ("lakh\ngrouping", ["12,34,567", "1,23,456"]),
        ("nothing\nfits", ["1.2345,67", "9"]),
    ]
    surv = []
    colors = []
    for _, col in cols:
        d = N.decide_column(col)
        surv.append(len(d.surviving))
        colors.append({"decided": C_AGREE, "ambiguous": C_DIFF,
                       "no-locale-fits": C_REFUSE}[d.verdict])
    ax.bar(range(len(cols)), surv, color=colors, width=0.66)
    for i, (label, col) in enumerate(cols):
        d = N.decide_column(col)
        n = len({str(v) for v in d.totals.values()})
        ax.text(i, surv[i] + 0.12, "%d total%s" % (n, "" if n == 1 else "s"),
                ha="center", fontsize=6.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], fontsize=6.5)
    ax.set_ylabel("surviving locale hypotheses", fontsize=7)
    ax.set_ylim(0, 6.2)
    ax.grid(axis="y", color=C_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title("6. Which column shapes decide themselves", loc="left")
    ax.legend(handles=[
        mpatches.Patch(color=C_AGREE, label="decided"),
        mpatches.Patch(color=C_DIFF, label="ambiguous"),
        mpatches.Patch(color=C_REFUSE, label="no locale fits"),
    ], fontsize=6, frameon=False, loc="upper right")


def build(path: str = "locale_audit") -> None:
    fig = plt.figure(figsize=(15.5, 13.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.55, 1.0, 1.0],
                          hspace=0.52, wspace=0.26,
                          left=0.075, right=0.90, top=0.925, bottom=0.055)
    panel_matrix(fig.add_subplot(gs[0, 0]))
    panel_ratios(fig.add_subplot(gs[0, 1]))
    panel_verdicts(fig.add_subplot(gs[1, 0]))
    panel_border(fig.add_subplot(gs[1, 1]))
    panel_diagonal(fig.add_subplot(gs[2, 0]))
    panel_decide(fig.add_subplot(gs[2, 1]))
    fig.suptitle("A numeric string does not contain a number. A reader assigns one.",
                 fontsize=15, fontweight="bold", x=0.075, ha="left", y=0.975)
    fig.text(0.075, 0.947,
             "35 strings a pipeline actually receives, read by 5 scanners and 5 CLDR "
             "locales x 2 strictness settings. Every number is measured, not modelled.",
             fontsize=9, color="#4b5563", ha="left")
    fig.savefig("%s.png" % path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig("%s.svg" % path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote %s.png and %s.svg" % (path, path))


def build_demo(path: str = "locale_demo") -> None:
    """A single-panel version for the README hero / notebook."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    verds = [v for v in N.all_verdicts() if v.ratio is not None and float(v.ratio) >= 100]
    verds.sort(key=lambda v: float(v.ratio))
    names = [v.case.escaped()[:20] for v in verds]
    vals = [float(v.ratio) for v in verds]
    ax.barh(range(len(vals)), vals, color=C_DIFF, height=0.7)
    for i, (v, c) in enumerate(zip(vals, verds)):
        lo = min(abs(d) for d in c.distinct if d != 0)
        hi = max(abs(d) for d in c.distinct if d != 0)
        ax.text(v * 1.15, i, "%s  vs  %s" % (format(lo.normalize(), "f"),
                                             format(hi.normalize(), "f")),
                va="center", fontsize=7, family="DejaVu Sans Mono")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8, family="DejaVu Sans Mono")
    ax.set_xscale("log")
    ax.set_xlim(50, 2e9)
    ax.set_xlabel("factor between two conforming readings of the same string (log)")
    ax.grid(axis="x", color=C_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Same bytes. Two readers. Two numbers, 100x to 1,234,567x apart.",
                 loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig("%s.png" % path, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote %s.png" % path)


if __name__ == "__main__":
    build()
    build_demo()
