"""The golden-record figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Left: field accuracy, six rule-sets x five fields, with the ceiling row (the true value is held by some record).
Right: the mechanism drawn - how a nightly CRM -> billing sync makes a stale address win a majority vote and look
fresh to 'most recent', on one timeline.
Geometry self-check runs BEFORE savefig: every text inside the canvas, no two labels overlapping.
"""

from __future__ import annotations

import json
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["text.parse_math"] = False
plt.rcParams["font.family"] = ["DejaVu Sans"]

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
GOOD, BAD, MID = "#2e8b57", "#c8562b", "#e0a23a"

R = json.load(open("results.json"))
FIELDS = ["name", "dob", "email", "phone", "address"]
RULES = list(R["acc"])

fig = plt.figure(figsize=(16, 9.2), facecolor=BG)
axM = fig.add_axes([0.11, 0.1, 0.46, 0.68])
axT = fig.add_axes([0.61, 0.16, 0.37, 0.56])
labels: List = []

labels.append(fig.text(0.02, 0.945, "Which customer record is real? Six survivorship rule-sets against the customer's details today",
                       fontsize=15, color=INK, weight="bold"))
labels.append(fig.text(0.02, 0.905, f"{R['clusters']} fictional customers held by 2-3 systems ({R['records']} records). "
                       f"Cell = share of customers whose kept value is the true one. Best rule-set gets all five fields "
                       f"right for {100 * max(R['whole'].values()):.0f}%; a perfect chooser, {100 * R['oracle_whole']:.0f}%.",
                       fontsize=10.5, color=MUTED))


def colour(v: float) -> str:
    return GOOD if v >= 0.9 else (MID if v >= 0.75 else BAD)


rows = RULES + ["ceiling"]
cols = FIELDS + ["all five"]
for i, k in enumerate(rows):
    y = len(rows) - 1 - i
    vals = [R["ceiling"][f] for f in FIELDS] + [R["oracle_whole"]] if k == "ceiling" else \
        [R["acc"][k][f] for f in FIELDS] + [R["whole"][k]]
    for j, v in enumerate(vals):
        axM.add_patch(Rectangle((j + 0.04, y + 0.06), 0.92, 0.88, color=colour(v) if k != "ceiling" else "#8a96a8",
                                alpha=0.92))
        labels.append(axM.text(j + 0.5, y + 0.5, f"{100 * v:.0f}%", ha="center", va="center", fontsize=11,
                               color="white", weight="bold"))
    labels.append(axM.text(-0.1, y + 0.5, k, ha="right", va="center", fontsize=10.5, color=INK,
                           weight="bold" if k in ("per_field", "ceiling") else "normal"))
for j, f in enumerate(cols):
    labels.append(axM.text(j + 0.5, len(rows) + 0.3, f, ha="center", va="center", fontsize=10.5, color=INK, weight="bold"))
axM.axhline(1, color=INK, lw=1.5)
axM.plot([len(FIELDS)] * 2, [0, len(rows)], color=INK, lw=1.0)
labels.append(axM.text(len(cols) / 2, -0.45, "green >= 90%   amber 75-89%   orange < 75%   grey = the true value is in "
                       "the cluster at all", ha="center", va="center", fontsize=9, color=MUTED))
axM.set_xlim(-0.05, len(cols) + 0.05)
axM.set_ylim(-0.8, len(rows) + 0.7)
axM.axis("off")

# --- the mechanism: one address, three systems, one timeline
axT.set_xlim(0, 10)
axT.set_ylim(0, 10)
axT.axis("off")
axT.plot([0.6, 9.6], [1.2, 1.2], color=INK, lw=1.4)
for x, lab in ((2.2, "day 150\nCRM keys it in"), (4.6, "day 400\ncustomer moves"), (6.6, "day 620\nweb form"),
               (8.6, "day 900\nnightly sync")):
    axT.plot([x, x], [1.05, 1.35], color=INK, lw=1.4)
    labels.append(axT.text(x, 0.85, lab, ha="center", va="top", fontsize=8.5, color=MUTED))
lanes = [("CRM", 7.9, 2.2, "57 Lantern Way", BAD), ("web", 5.4, 6.6, "134 Mill Street", GOOD),
         ("billing", 2.9, 8.6, "57 Lantern Way", BAD)]
for src, y, x, addr, col in lanes:
    labels.append(axT.text(0.15, y, src, ha="left", va="center", fontsize=10.5, color=INK, weight="bold"))
    axT.add_patch(FancyBboxPatch((x - 1.05, y - 0.55), 2.1, 1.1, boxstyle="round,pad=0.05", fc=BG, ec=col, lw=2))
    labels.append(axT.text(x, y, addr, ha="center", va="center", fontsize=8.5, color=col, weight="bold"))
axT.add_patch(FancyArrowPatch((3.3, 7.4), (7.6, 3.4), arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.3,
                              linestyle="--", connectionstyle="arc3,rad=0.3"))
labels.append(axT.text(5.6, 7.3, "sync copies CRM's old address,\nstamps it day 900", ha="center", va="center",
                       fontsize=8.5, color=MUTED))
labels.append(fig.text(0.61, 0.80, "How a sync makes a stale address win", fontsize=12.5, color=INK, weight="bold"))
s = R["stale_sync"]
labels.append(fig.text(0.61, 0.755, f"Two systems now agree on the old address, so majority keeps it ({s['majority_stale']} "
                       f"of {s['clusters']} such customers),\nand its stamp is the newest, so field-level recency keeps it too.",
                       fontsize=9.5, color=MUTED))
t = R["ts_fix"]
labels.append(fig.text(0.61, 0.05, f"Address right, 'most recent' read by: record touch {100 * t['record_ts']:.1f}%,\n"
                       f"field capture day {100 * t['field_ts']:.1f}%, capture day with lineage {100 * t['lineage_ts']:.1f}% "
                       f"(= ceiling).",
                       fontsize=9.5, color=INK))


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
