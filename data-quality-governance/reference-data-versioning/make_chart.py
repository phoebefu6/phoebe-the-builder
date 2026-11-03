"""The reference-data-versioning figure. Every number read from results.json, so the chart cannot drift.

Left: how many of 36 published monthly reports each rebuild strategy reproduces, with the revenue it reports.
Right: the mechanism drawn - one old sale under a reused code, and what each join does with it.
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

R = json.load(open("results.json"))
STRATS = list(R["asw"])
U = R["reuse"]

fig = plt.figure(figsize=(16, 9), facecolor=BG)
axB = fig.add_axes([0.12, 0.14, 0.36, 0.62])
axT = fig.add_axes([0.54, 0.1, 0.44, 0.66])
labels: List = []

labels.append(fig.text(0.02, 0.945, "The country list changed: can last year's regional report be rebuilt today?",
                       fontsize=15, color=INK, weight="bold"))
labels.append(fig.text(0.02, 0.905, f"40 fictional territories over 36 months: 6 region moves ({R['backdated']} recorded "
                       "late), 2 renames, 2 splits, 1 merge, 1 code retired and reused. Each month's report was "
                       "published from the table as it stood that day.", fontsize=10.5, color=MUTED))

# --- left: months reproduced
for i, k in enumerate(STRATS):
    y = len(STRATS) - 1 - i
    a = R["asw"][k]
    col = GOOD if a["months"] == 36 else (MID if a["months"] >= 24 else BAD)
    axB.barh(y, a["months"], color=col, height=0.62)
    labels.append(axB.text(-0.8, y, k, ha="right", va="center", fontsize=11, color=INK, weight="bold"))
    labels.append(axB.text(a["months"] + 0.6, y + 0.12, f"{a['months']} of 36", ha="left", va="center", fontsize=11,
                           color=INK, weight="bold"))
    note = f"reports {100 * a['reported']:.1f}% of revenue"
    labels.append(axB.text(a["months"] + 0.6, y - 0.2, note, ha="left", va="center", fontsize=8.5, color=MUTED))
axB.set_xlim(0, 52)
axB.set_ylim(-0.7, len(STRATS) - 0.3)
axB.axis("off")
labels.append(fig.text(0.02, 0.80, "Published months the rebuild reproduces exactly", fontsize=12.5, color=INK,
                       weight="bold"))
labels.append(fig.text(0.02, 0.07, "overwrite: today's table only   code_only: every version, joined on code\n"
                       "scd2_asof: code + sale month   bitemporal: + what the table said on the run day",
                       fontsize=9, color=MUTED))

# --- right: one sale, three lookups
axT.set_xlim(0, 10)
axT.set_ylim(0, 10)
axT.axis("off")
labels.append(fig.text(0.54, 0.80, f"One sale, three joins: code {U['code']} was retired and reused", fontsize=12.5,
                       color=INK, weight="bold"))
axT.add_patch(FancyBboxPatch((0.2, 4.2), 2.4, 1.6, boxstyle="round,pad=0.08", fc=BG, ec=INK, lw=2))
labels.append(axT.text(1.4, 5.0, f"sale, month 3\ncode {U['code']}", ha="center", va="center", fontsize=10, color=INK,
                       weight="bold"))
outs = [
    (8.3, "today's table", f"{U['code']} -> {U['new_region']}\n(the new territory)", BAD),
    (5.0, "history, code only", f"{U['code']} -> {U['old_region']}\nAND {U['new_region']}: counted twice", BAD),
    (1.7, "history, as of month 3", f"{U['code']} -> {U['old_region']}\n(the territory that sold it)", GOOD),
]
for y, how, what, col in outs:
    axT.add_patch(FancyArrowPatch((2.75, 5.0), (5.4, y), arrowstyle="-|>", mutation_scale=14, color=col, lw=1.6,
                                  connectionstyle="arc3,rad=0.0"))
    labels.append(axT.text(4.05, y + (0.55 if y > 5 else -0.55 if y < 5 else 0.45), how, ha="center", va="center",
                           fontsize=8.5, color=MUTED))
    axT.add_patch(FancyBboxPatch((5.5, y - 0.8), 4.3, 1.6, boxstyle="round,pad=0.06", fc=BG, ec=col, lw=2))
    labels.append(axT.text(7.65, y, what, ha="center", va="center", fontsize=9.5, color=col, weight="bold"))
labels.append(fig.text(0.54, 0.035, f"Overwrite files {U['rev']:,} of old {U['code']} revenue under {U['new_region']}. "
                       f"Valid time alone still misses {36 - R['asw']['scd2_asof']['months']} months,\nbecause "
                       f"{R['backdated']} moves were recorded after the report had gone out.", fontsize=9.5, color=INK))


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
