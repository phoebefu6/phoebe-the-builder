"""The audit figure. Four panels, all read from results.json so the chart cannot drift from the
numbers in evidence.txt and the README.

Encoding discipline: one fill per meaning, held across every panel. SHAPIRO is always the same
orange, the T-TEST always the same blue, and "correct" is never a colour - it is distance from
the dashed nominal line at 0.05.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
SHAPIRO, TTEST, GATED, WILCOXON = "#c8562b", "#2f6fdb", "#7a4fa3", "#9aa4b0"
# Cell verdicts on the map panel.
FALSE_ALARM, TRUE_POS, MISS, TRUE_NEG = "#e4a33c", "#3f8f5c", "#b03030", "#e8ecf1"

R = json.load(open("results.json"))
CELLS = R["cells"]
NS = R["meta"]["grid_n"]
DISTS = R["meta"]["distributions"]
BAND = R["meta"]["band"]
ALPHA = R["meta"]["alpha"]

BY = {d: sorted([c for c in CELLS if c["dist"] == d], key=lambda c: c["n"]) for d in DISTS}


def broken(c) -> bool:
    """Read the verdict results.json already carries - never recompute it here.

    This function used to compare the point estimate to the band itself, which made the chart
    disagree with evidence.txt on four cells: the library decides with a 99% Wilson interval at
    that cell's replicate count, and a second copy of a rule is a second answer.
    """
    return bool(c["t_is_broken"])


fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5), facecolor=BG)
for ax in axes.ravel():
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)

# ------------------------------------------------ 1. the two curves, running opposite ways
ax = axes[0, 0]
show = ["lognormal-mild", "exponential", "uniform"]
styles = ["-", "--", ":"]
for d, ls in zip(show, styles):
    rows = BY[d]
    ax.plot(NS, [c["shapiro_reject_rate"] for c in rows], ls, color=SHAPIRO, lw=2.2, marker="o", ms=4)
    ax.plot(NS, [c["t_error"] for c in rows], ls, color=TTEST, lw=2.2, marker="s", ms=4)
ax.axhline(ALPHA, color=INK, ls="--", lw=1.6)
ax.text(NS[0], ALPHA * 0.88, "nominal 0.05", color=INK, fontsize=8, ha="left", va="top")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(NS)
ax.set_xticklabels([str(n) for n in NS], fontsize=8, color=MUTED)
ax.set_ylim(5e-3, 1.6)
ax.set_xlabel("sample size n", color=MUTED, fontsize=9)
ax.set_ylabel("rejection rate (log)", color=MUTED, fontsize=9)
ax.set_title(
    "1. The gate gets surer as the test gets better",
    color=INK, fontsize=12, weight="bold", loc="left",
)
ax.legend(
    handles=[
        Patch(color=SHAPIRO, label="Shapiro-Wilk rejects normality"),
        Patch(color=TTEST, label="t-test false-positive rate"),
    ]
    + [plt.Line2D([], [], color=MUTED, ls=ls, label=d) for d, ls in zip(show, styles)],
    fontsize=8, framealpha=1, edgecolor=GRID, loc="center right",
)

# ------------------------------------------------ 2. the verdict map
ax = axes[0, 1]
grid = np.zeros((len(DISTS), len(NS), 3))
lookup = {FALSE_ALARM: 0, TRUE_POS: 1, MISS: 2, TRUE_NEG: 3}
for i, d in enumerate(DISTS):
    for j, c in enumerate(BY[d]):
        fires = c["shapiro_reject_rate"] >= 0.50
        bad = broken(c)
        col = (TRUE_POS if bad else FALSE_ALARM) if fires else (MISS if bad else TRUE_NEG)
        grid[i, j] = matplotlib.colors.to_rgb(col)
ax.imshow(grid, aspect="auto", interpolation="nearest")
ax.set_xticks(range(len(NS)))
ax.set_xticklabels([str(n) for n in NS], fontsize=8, color=MUTED)
ax.set_yticks(range(len(DISTS)))
ax.set_yticklabels(DISTS, fontsize=8, color=MUTED)
ax.grid(False)
for i, d in enumerate(DISTS):
    for j, c in enumerate(BY[d]):
        ax.text(j, i, f"{c['t_error']:.3f}", ha="center", va="center", fontsize=6.5, color=INK)
ax.set_xlabel("sample size n   (cell label = t-test's real error rate)", color=MUTED, fontsize=9)
ax.set_title(
    "2. What the gate's verdict was worth, cell by cell",
    color=INK, fontsize=12, weight="bold", loc="left",
)
ax.legend(
    handles=[
        Patch(color=FALSE_ALARM, label="fires, but the t-test was fine (false alarm)"),
        Patch(color=TRUE_POS, label="fires, t-test really is broken"),
        Patch(color=MISS, label="stays quiet, t-test is broken (miss)"),
        Patch(facecolor=TRUE_NEG, edgecolor=GRID, label="stays quiet, t-test is fine"),
    ],
    fontsize=7.5, framealpha=1, edgecolor=GRID, loc="lower left", bbox_to_anchor=(0.0, -0.42), ncol=2,
)

# ------------------------------------------------ 3. what following the gate costs
ax = axes[1, 0]
picks = [("lognormal-mild", 50), ("exponential", 50), ("lognormal-strong", 200), ("exponential", 1000)]
labels, t_v, w_v, g_v = [], [], [], []
for d, n in picks:
    c = next(x for x in BY[d] if x["n"] == n)
    labels.append(f"{d}\nn={n}")
    t_v.append(c["t_error"])
    w_v.append(c["wilcoxon_error"])
    g_v.append(c["conditional_error"])
y = np.arange(len(picks))
ax.barh(y + 0.26, t_v, height=0.25, color=TTEST, label="t-test alone")
ax.barh(y, w_v, height=0.25, color=WILCOXON, label="Wilcoxon alone")
ax.barh(y - 0.26, g_v, height=0.25, color=GATED, label="Shapiro-gated (switch if rejected)")
ax.axvline(ALPHA, color=INK, ls="--", lw=1.8)
for i in range(len(picks)):
    for off, v in ((0.26, t_v[i]), (0.0, w_v[i]), (-0.26, g_v[i])):
        ax.text(max(v, 1e-3) * 1.12, i + off, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
ax.set_xscale("log")
ax.set_xlim(2e-2, 4.0)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8, color=MUTED)
ax.set_xlabel("false-positive rate under a TRUE null (log)", color=MUTED, fontsize=9)
ax.set_title(
    "3. The swap the gate recommends is the expensive part",
    color=INK, fontsize=12, weight="bold", loc="left",
)
ax.legend(fontsize=8, framealpha=1, edgecolor=GRID, loc="lower right")

# ------------------------------------------------ 4. the defect two-sided p hides
ax = axes[1, 1]
d = "exponential"
rows = BY[d]
lo = [c["t_error_lower_tail"] for c in rows]
hi = [c["t_error_upper_tail"] for c in rows]
x = np.arange(len(NS))
ax.bar(x - 0.19, lo, width=0.36, color=TTEST, label="rejected LOW")
ax.bar(x + 0.19, hi, width=0.36, color=SHAPIRO, label="rejected HIGH")
ax.axhline(ALPHA / 2, color=INK, ls="--", lw=1.6)
ax.text(len(NS) - 1.3, ALPHA / 2 * 1.12, "0.025 each side", color=INK, fontsize=8, ha="right")
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in NS], fontsize=8, color=MUTED)
ax.set_xlabel(f"sample size n   (population: {d}, null is exactly true)", color=MUTED, fontsize=9)
ax.set_ylabel("one-sided false-positive rate", color=MUTED, fontsize=9)
ax.set_title(
    "4. A 0.05 test that only points one way",
    color=INK, fontsize=12, weight="bold", loc="left",
)
ax.legend(fontsize=8, framealpha=1, edgecolor=GRID)

fig.suptitle(
    "Shapiro-Wilk scored as a gate on the t-test: "
    f"sensitivity {R['derived']['diagnostic']['sensitivity']:.2f}, "
    f"specificity {R['derived']['diagnostic']['specificity']:.2f} over {R['derived']['diagnostic']['cells']} cells",
    color=INK, fontsize=14, weight="bold", x=0.012, ha="left", y=0.985,
)
fig.tight_layout(rect=(0, 0.04, 1, 0.955))
fig.savefig("normality_test_trap_audit.png", dpi=170, facecolor=BG)
fig.savefig("normality_test_trap_audit.svg", facecolor=BG)
print("wrote normality_test_trap_audit.png / .svg")
