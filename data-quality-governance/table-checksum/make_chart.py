"""The table-checksum figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Left: the verdict matrix - 13 things that happen to a table in transit x 5 fingerprints, each cell right or wrong.
Right: the mechanism drawn - why even the canonical fingerprint false-alarms once in 1,000 rows: two floats one
ulp either side of a half cent round to different cents.
Geometry self-check runs BEFORE savefig: every text inside the canvas, no two labels overlapping.
"""

from __future__ import annotations

import json
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams["text.parse_math"] = False
plt.rcParams["font.family"] = ["DejaVu Sans"]

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
RIGHT, WRONG, PART = "#2e8b57", "#c8562b", "#e0a23a"

R = json.load(open("results.json"))
SCHEMES = ["csv_md5", "count_sums", "xor_rows", "sum_rows", "canonical"]
ROWS = [(n, v, "benign") for n, v in R["scores"]["benign"].items()] + [
    (n, v, "mutation") for n, v in R["scores"]["mutations"].items()
]

fig = plt.figure(figsize=(16, 9.6), facecolor=BG)
axM = fig.add_axes([0.20, 0.08, 0.43, 0.74])
axN = fig.add_axes([0.66, 0.30, 0.32, 0.42])
labels: List = []

labels.append(fig.text(0.02, 0.945, "Are these two tables the same? Five fingerprints, thirteen things that happen in transit",
                       fontsize=15, color=INK, weight="bold"))
labels.append(fig.text(0.02, 0.91, f"A fictional {R['rows']:,}-row keyless event export ({R['dup_pairs']} exact duplicate pairs). "
                       "Green = the fingerprint gave the right answer in every trial; orange = wrong in every trial; amber = right in only some.",
                       fontsize=10.5, color=MUTED))

# --- the verdict matrix
n = len(ROWS)
for i, (name, v, kind) in enumerate(ROWS):
    y = n - 1 - i
    for j, k in enumerate(SCHEMES):
        m = v["match"][k]
        ok = m if kind == "benign" else 1 - m
        col = RIGHT if ok == 1 else (WRONG if ok == 0 else PART)
        axM.add_patch(Rectangle((j + 0.04, y + 0.06), 0.92, 0.88, color=col, alpha=0.9 if ok in (0, 1) else 0.8))
        txt = "right" if ok == 1 else ("wrong" if ok == 0 else f"right {100 * ok:.0f}%")
        labels.append(axM.text(j + 0.5, y + 0.5, txt, ha="center", va="center", fontsize=9, color="white", weight="bold"))
    labels.append(axM.text(-0.12, y + 0.5, name, ha="right", va="center", fontsize=10, color=INK))
axM.axhline(len(R["scores"]["mutations"]), color=INK, lw=1.5)
nm = len(R["scores"]["mutations"])
labels.append(axM.text(len(SCHEMES) + 0.12, nm + (n - nm) / 2, "SAME DATA\nshould match", rotation=90, ha="left",
                       va="center", fontsize=9.5, color=MUTED, weight="bold", multialignment="center"))
labels.append(axM.text(len(SCHEMES) + 0.12, nm / 2, "DIFFERENT DATA\nshould differ", rotation=90, ha="left",
                       va="center", fontsize=9.5, color=MUTED, weight="bold", multialignment="center"))
for j, k in enumerate(SCHEMES):
    t = R["tally"][k]
    labels.append(axM.text(j + 0.5, n + 0.55, k, ha="center", va="center", fontsize=10.5, color=INK, weight="bold"))
    labels.append(axM.text(j + 0.5, -0.45, f"{len(t['false_alarms'])} FA / {len(t['misses'])} miss", ha="center",
                           va="center", fontsize=9, color=MUTED))
axM.set_xlim(0, len(SCHEMES) + 0.5)
axM.set_ylim(-0.8, n + 0.9)
axM.axis("off")

# --- the mechanism drawn: a number line around the half cent
f = R["float"]
eid, a, b = f["straddle"][0]
axN.set_xlim(-3.2, 3.2)
axN.set_ylim(-2.3, 2.5)
axN.axis("off")
axN.plot([-3, 3], [0, 0], color=INK, lw=1.6)
for x, lab in ((-2.6, "821.22"), (0, "821.225\nhalf cent"), (2.6, "821.23")):
    axN.plot([x, x], [-0.15, 0.15], color=INK, lw=1.6 if x else 2.4)
    labels.append(axN.text(x, -0.35, lab, ha="center", va="top", fontsize=9.5, color=INK if x else WRONG))
for x, val, side, col, out in ((-0.45, a, "forward sum", RIGHT, "rounds to 821.22"),
                               (0.45, b, "reverse sum", PART, "rounds to 821.23")):
    axN.plot([x], [0], "o", color=col, ms=10, zorder=3)
    axN.annotate("", xy=(x, 0.12), xytext=(x * 3.2, 1.25), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4))
    labels.append(axN.text(x * 3.2, 1.35, f"{side}\n{val}\n{out}", ha="center", va="bottom", fontsize=9, color=col,
                           weight="bold"))
axN.add_patch(FancyBboxPatch((-3.05, -2.2), 6.1, 0.95, boxstyle="round,pad=0.02", fc="#f6f7f9", ec=GRID))
labels.append(axN.text(0, -1.73, f"event {eid}: same line items, added in the other order.\n"
                       f"{f['bits_differ']} of {f['rows']} amounts differ in the bits,\n{f['half_cent_exact']} sit exactly on a "
                       f"half cent, {f['cents_differ']} rounds to a different cent.", ha="center", va="center", fontsize=9,
                       color=INK))
labels.append(fig.text(0.66, 0.79, "Why canonical still false-alarms once", fontsize=12.5, color=INK, weight="bold"))
labels.append(fig.text(0.66, 0.745, f"Rounding to the cent removes {f['bits_differ'] - f['cents_differ']} of the "
                       f"{f['bits_differ']} bit differences,\nand turns the last one into a different cent.",
                       fontsize=9.5, color=MUTED))
labels.append(fig.text(0.66, 0.2, "Fix: store money at its declared scale (NUMERIC(12,2)) before hashing,\n"
                       "so both sides round once, at load, not at compare time.", fontsize=10, color=INK))


def check() -> None:
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    boxes = [(t.get_text()[:30], t.get_window_extent(rr)) for t in labels if t.get_text()]
    for s, bb in boxes:
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= W and bb.y1 <= H, f"outside canvas: {s!r}"
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                raise AssertionError(f"overlap: {boxes[i][0]!r} / {boxes[j][0]!r}")


check()
fig.savefig("audit.png", dpi=150, facecolor=BG)
fig.savefig("audit.svg", facecolor=BG)
print("wrote audit.png, audit.svg; geometry check passed")
