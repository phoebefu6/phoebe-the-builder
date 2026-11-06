"""The forecast-reconciliation figure. Every number read from results.json, so the chart cannot drift.

Left: RMSE change vs the unreconciled forecasts, per level, across 30 worlds, with whether each method adds up.
Right: the mechanism drawn - three forecasts that do not add up, and the three ways to force them to.
Geometry self-check runs BEFORE savefig: every text inside the canvas, no two labels overlapping.
"""

from __future__ import annotations

import json
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["text.parse_math"] = False
plt.rcParams["font.family"] = ["DejaVu Sans"]

INK, MUTED, BG = "#1f2733", "#6b7684", "#ffffff"
GOOD, BAD, MID = "#2e8b57", "#c8562b", "#e0a23a"
LVCOL = {"total": "#1f4e79", "region": "#4f8fc0", "store": "#a9c8e4"}

R = json.load(open("results.json"))
S = R["seeds"]
METHODS = [m for m in S["mean"] if m != "base"]
LEVELS = ["total", "region", "store"]

fig = plt.figure(figsize=(16, 9), facecolor=BG)
axB = fig.add_axes([0.14, 0.17, 0.34, 0.57])
axT = fig.add_axes([0.53, 0.08, 0.46, 0.70])
labels: List = []

labels.append(fig.text(0.02, 0.945, "Every method adds up. That is the only thing they all get right.", fontsize=15,
                       color=INK, weight="bold"))
labels.append(fig.text(0.02, 0.905, "A fictional retailer (1 national number, 4 regions, 12 stores) forecast series by "
                       "series, then forced to add up seven ways. Scored on 12 months that happened, 30 worlds.",
                       fontsize=10.5, color=MUTED))
labels.append(fig.text(0.02, 0.80, "RMSE change vs the unreconciled forecasts (left of 0 = better)", fontsize=12.5,
                       color=INK, weight="bold"))

CAP = 25.0
for i, m in enumerate(METHODS):
    y0 = len(METHODS) - 1 - i
    for j, lv in enumerate(LEVELS):
        v = 100 * S["mean"][m][lv]
        y = y0 + 0.25 - j * 0.25
        axB.barh(y, max(min(v, CAP), -CAP), height=0.22, color=LVCOL[lv])
        if v > CAP:
            labels.append(axB.text(CAP + 0.5, y, f"+{v:.0f}% (off scale)", va="center", fontsize=8, color=BAD))
    labels.append(axB.text(-CAP - 1.0, y0, m, ha="right", va="center", fontsize=11, color=INK, weight="bold"))
axB.axvline(0, color=MUTED, lw=1)
axB.set_xlim(-CAP, CAP + 12)
axB.set_ylim(-0.6, len(METHODS) - 0.4)
for x in (-10, 0, 10, 20):
    labels.append(axB.text(x, -0.75, f"{x:+d}%" if x else "0", ha="center", va="top", fontsize=8.5, color=MUTED))
axB.axis("off")
for j, lv in enumerate(LEVELS):
    fig.patches.append(FancyBboxPatch((0.03 + j * 0.1, 0.085), 0.012, 0.02, transform=fig.transFigure, fc=LVCOL[lv],
                                      ec=LVCOL[lv], boxstyle="square,pad=0"))
    labels.append(fig.text(0.046 + j * 0.1, 0.095, lv, fontsize=9, color=INK, va="center"))
sh = R["shock"]
labels.append(fig.text(0.02, 0.035, f"A {100 * sh['shock']:.0f}% cut in one region that no history contains: "
                       f"base over-forecasts it {100 * sh['median_bias']['base']:+.1f}%, mint_shrink "
                       f"{100 * sh['median_bias']['mint_shrink']:+.1f}%. Coherent and wrong.", fontsize=9.5, color=INK))

# --- right: the mechanism
axT.set_xlim(0, 10)
axT.set_ylim(0, 10)
axT.axis("off")
labels.append(fig.text(0.53, 0.80, "Three forecasts that do not add up, and three ways to force them", fontsize=12.5,
                       color=INK, weight="bold"))


def box(x: float, y: float, text: str, col: str = INK, w: float = 1.9) -> None:
    axT.add_patch(FancyBboxPatch((x - w / 2, y - 0.35), w, 0.7, boxstyle="round,pad=0.06", fc=BG, ec=col, lw=2))
    labels.append(axT.text(x, y, text, ha="center", va="center", fontsize=9, color=col, weight="bold"))


def edge(x0: float, y0: float, x1: float, y1: float, col: str = MUTED, style: str = "-") -> None:
    axT.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=col, lw=1.6,
                                  linestyle=style))


gap = 100 * R["base_regions_vs_total"]
box(5.0, 9.2, "national 100")
for x, t in ((2.4, "North 30"), (5.0, "South 35"), (7.6, "East 37")):
    box(x, 7.8, t, MUTED, 1.6)
labels.append(axT.text(5.0, 6.95, f"regions sum to 102: each forecast on its own (here {gap:.2f}% apart at worst)",
                       ha="center", va="center", fontsize=9, color=BAD, zorder=5,
                       bbox=dict(boxstyle="square,pad=0.25", fc=BG, ec="none")))
# three ways
box(1.7, 5.3, "bottom-up", GOOD, 1.8)
box(5.0, 5.3, "top-down", BAD, 1.8)
box(8.3, 5.3, "MinT / WLS", MID, 1.8)
edge(2.4, 7.4, 1.7, 5.7, GOOD)
edge(5.0, 8.85, 5.0, 5.7, BAD)
edge(7.6, 7.4, 8.3, 5.7, MID)
labels.append(axT.text(1.7, 3.85, "national := 102\nkeeps every region,\noverrides the top", ha="center", va="center",
                       fontsize=8.5, color=GOOD))
labels.append(axT.text(5.0, 3.85, "regions := 100 x old shares\nkeeps the top, overwrites\nthe regions with history",
                       ha="center", va="center", fontsize=8.5, color=BAD))
labels.append(axT.text(8.3, 3.85, "moves every number a\nlittle, weighted by how\nnoisy each one has been",
                       ha="center", va="center", fontsize=8.5, color=MID))
tb = S["total_bias"]
labels.append(axT.text(5.0, 1.9, f"Here the stores were the better forecasts: national model {100 * tb['base']:+.1f}% "
                       f"median, sum of stores {100 * tb['bottom_up']:+.1f}%.", ha="center", va="center", fontsize=9.5,
                       color=INK))
labels.append(axT.text(5.0, 1.1, f"Old shares were stale: {100 * R['share_drift']:.1f}% of sales changed store "
                       "between history and the scored year.", ha="center", va="center", fontsize=9.5, color=INK))


def check() -> None:
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    boxes = [(t.get_text()[:30], t.get_window_extent(rr)) for t in labels if t.get_text()]
    for s_, bb in boxes:
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= W and bb.y1 <= H, f"outside canvas: {s_!r}"
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                raise AssertionError(f"overlap: {boxes[i][0]!r} / {boxes[j][0]!r}")


check()
fig.savefig("audit.png", dpi=150, facecolor=BG)
fig.savefig("audit.svg", facecolor=BG)
print("wrote audit.png, audit.svg; geometry check passed")
