"""The hierarchy-reconciliation figure. Every number read from results.json, so the chart cannot drift.

Left: where each posted dollar ends up under five roll-ups, with the result of the "regions sum to total" check.
Right: the mechanism drawn - a moved cost centre under two parents, an orphan, and an area cut off by a cycle.
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
GOOD, BAD, MID, TWICE = "#2e8b57", "#c8562b", "#e0a23a", "#7b5ea7"

R = json.load(open("results.json"))
METHODS = list(R["methods"])
SEGS = [("right", GOOD, "right region"), ("misfiled", MID, "wrong region"), ("dropped", BAD, "unplaced"),
        ("double", TWICE, "counted twice")]

fig = plt.figure(figsize=(16, 9), facecolor=BG)
axB = fig.add_axes([0.13, 0.16, 0.36, 0.58])
axT = fig.add_axes([0.53, 0.08, 0.46, 0.70])
labels: List = []

labels.append(fig.text(0.02, 0.945, "The regions sum to the total. The total is still wrong.", fontsize=15, color=INK,
                       weight="bold"))
labels.append(fig.text(0.02, 0.905, f"A fictional cost-centre hierarchy ({R['leaves']} cost centres, {R['areas']} areas, "
                       f"5 regions) extracted with the usual defects. The check 'do the regions sum to the total?' "
                       f"passes for all five roll-ups.", fontsize=10.5, color=MUTED))

# --- left: per-dollar outcome
for i, k in enumerate(METHODS):
    y = len(METHODS) - 1 - i
    a = R["methods"][k]
    x = 0.0
    for key, col, _ in SEGS:
        v = 100 * a[key]
        axB.barh(y, v, left=x, color=col, height=0.6)
        x += v
    labels.append(axB.text(-1.5, y, k, ha="right", va="center", fontsize=11, color=INK, weight="bold"))
    tie = "ties to ledger" if a["total_is_ledger"] else f"total {100 * a['total']:.1f}% of ledger"
    labels.append(axB.text(x + 1.0, y, f"{100 * a['wrong']:.1f}% wrong\n{tie}", ha="left", va="center", fontsize=8.5,
                           color=INK))
axB.set_xlim(0, 140)
axB.set_ylim(-0.7, len(METHODS) - 0.3)
axB.axis("off")
labels.append(fig.text(0.02, 0.80, "Where each posted dollar ends up", fontsize=12.5, color=INK, weight="bold"))
for j, (_, col, name) in enumerate(SEGS):
    fig.patches.append(FancyBboxPatch((0.03 + j * 0.105, 0.085), 0.012, 0.02, transform=fig.transFigure, fc=col,
                                      ec=col, boxstyle="square,pad=0"))
    labels.append(fig.text(0.046 + j * 0.105, 0.095, name, fontsize=9, color=INK, va="center"))
labels.append(fig.text(0.02, 0.035, f"The plug ties at {100 * R['plug_net']:.1f}% of the ledger while "
                       f"{100 * R['methods']['plug']['wrong']:.1f}% of dollars sit in the wrong place underneath it.",
                       fontsize=9.5, color=INK))

# --- right: the tree, drawn
axT.set_xlim(0, 10)
axT.set_ylim(0, 10)
axT.axis("off")
labels.append(fig.text(0.53, 0.80, "Three defects in one extract, and what a roll-up does with each", fontsize=12.5,
                       color=INK, weight="bold"))


def node(x: float, y: float, text: str, col: str = INK, w: float = 1.5) -> None:
    axT.add_patch(FancyBboxPatch((x - w / 2, y - 0.35), w, 0.7, boxstyle="round,pad=0.06", fc=BG, ec=col, lw=2))
    labels.append(axT.text(x, y, text, ha="center", va="center", fontsize=9, color=col, weight="bold"))


def edge(x0: float, y0: float, x1: float, y1: float, col: str = MUTED, style: str = "-", rad: float = 0.0) -> None:
    axT.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-", color=col, lw=1.6, linestyle=style,
                                  connectionstyle=f"arc3,rad={rad}"))


node(5.0, 9.2, "TOTAL")
for x, name in ((1.8, "North"), (5.0, "South"), (8.2, "East")):
    node(x, 7.7, name)
    edge(5.0, 8.85, x, 8.05)
# moved cost centre: two parent rows
node(1.0, 6.0, "A11", w=1.1)
node(3.0, 6.0, "A21", w=1.1)
edge(1.8, 7.35, 1.0, 6.35)
edge(5.0, 7.35, 3.0, 6.35)
node(2.0, 4.3, "CC moved", TWICE, w=1.7)
edge(1.0, 5.65, 2.0, 4.65, TWICE, "--")
edge(3.0, 5.65, 2.0, 4.65, TWICE)
labels.append(axT.text(2.0, 3.25, "old row never end-dated:\na walk sums it under both", ha="center", va="center",
                       fontsize=8.5, color=TWICE))
# orphan: parent retired
node(5.0, 6.0, "Z01 (retired)", BAD, w=2.0)
labels.append(axT.text(5.0, 6.75, "not in extract", ha="center", va="center", fontsize=8, color=BAD))
node(5.0, 4.3, "CC orphan", BAD, w=1.7)
edge(5.0, 4.65, 5.0, 5.65, BAD, "--")
labels.append(axT.text(5.0, 3.25, "parent was retired:\nits dollars reach no region", ha="center", va="center",
                       fontsize=8.5, color=BAD))
# cycle: area's parent keyed as its own child
node(8.2, 5.6, "A3x", BAD, w=1.1)
node(8.2, 3.9, "CC child", BAD, w=1.5)
edge(8.0, 5.25, 8.0, 4.25, BAD)
edge(8.4, 4.25, 8.4, 5.25, BAD, "--")
edge(8.2, 7.35, 8.2, 5.95, MUTED, ":")
labels.append(axT.text(9.0, 6.6, "edge lost", ha="left", va="center", fontsize=8, color=MUTED))
labels.append(axT.text(8.2, 2.8, "area's parent field keyed\nwith its own child: the\nwhole area falls off", ha="center",
                       va="center", fontsize=8.5, color=BAD))
c = R["share"]
labels.append(axT.text(5.0, 1.0, f"moved {100 * c['moved']:.1f}% of dollars   orphans {100 * c['orphan']:.1f}%   "
                       f"one cycle {100 * c['cycle']:.1f}%   unknown codes {100 * c['unknown']:.1f}%", ha="center",
                       va="center", fontsize=9.5, color=INK))


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
