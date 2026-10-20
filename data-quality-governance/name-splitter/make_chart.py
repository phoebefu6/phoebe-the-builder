"""The name-splitter figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Top: the mechanism drawn - names with the SAME shape and different truths, each word coloured by its true role, so
it is visible why a rule over word positions has to get some of them wrong. Bottom left: convention x splitter, the
share split exactly right. Bottom right: rows split right per splitter against the two ceilings.
Geometry self-check runs BEFORE savefig: every text inside the canvas, no two labels overlapping.
"""

from __future__ import annotations

import json
from typing import List

import matplotlib

matplotlib.use("Agg")
import explore as E
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

plt.rcParams["text.parse_math"] = False
plt.rcParams["font.family"] = ["DejaVu Sans"]

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
GIV, FAM, OTHER, BAD = "#2f6fdb", "#2e8b57", "#9aa3ad", "#c8562b"

R = json.load(open("results.json"))["main"]
NAMES = list(E.SPLITTERS)
N = R["n"]

fig = plt.figure(figsize=(16, 11.5), facecolor=BG)
axM = fig.add_axes([0.03, 0.62, 0.94, 0.27])
axH = fig.add_axes([0.17, 0.14, 0.42, 0.38])
axB = fig.add_axes([0.73, 0.14, 0.24, 0.36])
labels: List = []

# ---------------------------------------------------------------- M. same shape, different truth
ax = axM
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 12)
fig.text(
    0.03,
    0.945,
    "Why a name splitter cannot be right: the string does not say where the surname starts",
    fontsize=15,
    color=INK,
    weight="bold",
)
fig.text(
    0.03,
    0.918,
    f"{N} fictional names, 12 conventions. Best splitter {R['per'][R['best']]['both_right']}/{N}; the best ANY rule "
    f"over word positions could do is {R['ceiling']['by_shape']}/{N}.",
    fontsize=11,
    color=MUTED,
)
examples = [
    ("Thomas Edward Barlow", "middle name", "G-F"),
    ("Carlos Ruiz Delgado", "two surnames", "GFF"),
    ("Rupert Ashby Lowe", "double-barrelled", "GFF"),
    ("Chen Lin", "family first", "FG"),
    ("Chen Lin", "given first", "GF"),
]
col = {"G": GIV, "F": FAM, "-": OTHER}
y0 = 10.2
for k, (s, cap, rl) in enumerate(examples):
    y = y0 - k * 2.05
    x = 22
    labels.append(ax.text(20.5, y, cap, ha="right", va="center", fontsize=10.5, color=MUTED))
    for w, r in zip(s.split(), rl):
        wd = 1.35 * len(w) + 2.4
        ax.add_patch(
            FancyBboxPatch(
                (x, y - 0.75),
                wd,
                1.5,
                boxstyle="round,pad=0.05,rounding_size=0.4",
                fc=col[r],
                alpha=0.16,
                ec=col[r],
                lw=1.6,
            )
        )
        labels.append(
            ax.text(
                x + wd / 2, y, w, ha="center", va="center", fontsize=14, color=INK, family="DejaVu Sans Mono"
            )
        )
        x += wd + 0.9
ax.plot([21.3, 21.3], [y0 - 4 * 2.05 - 0.9, y0 + 0.9], color=GRID, lw=1)
# brackets: same shape groups
ax.annotate(
    "",
    xy=(58, y0 + 0.2),
    xytext=(58, y0 - 2 * 2.05 - 0.2),
    arrowprops=dict(arrowstyle="-", color=BAD, lw=1.6),
)
labels.append(
    ax.text(
        59.2,
        y0 - 2.05,
        f"same shape: three plain words.\nA position rule must pick ONE split;\n"
        f"{R['ceiling']['WWW'].get('G-F', 0)} rows want the middle-name split, "
        f"{R['ceiling']['WWW'].get('GFF', 0)} the surname pair.",
        va="center",
        fontsize=10,
        color=BAD,
    )
)
ax.annotate(
    "",
    xy=(58, y0 - 3 * 2.05 + 0.2),
    xytext=(58, y0 - 4 * 2.05 - 0.2),
    arrowprops=dict(arrowstyle="-", color=BAD, lw=1.6),
)
labels.append(
    ax.text(
        59.2,
        y0 - 3.5 * 2.05,
        "same string, two people.\nNothing that reads only the string\ncan get both right.",
        va="center",
        fontsize=10,
        color=BAD,
    )
)
for i, (lab, c) in enumerate(
    [("given (the greeting)", GIV), ("family (the surname)", FAM), ("neither", OTHER)]
):
    ax.add_patch(
        FancyBboxPatch((86, y0 - 0.45 - i * 1.4), 1.6, 0.9, boxstyle="round,pad=0.02", fc=c, alpha=0.3, ec=c)
    )
    labels.append(ax.text(88.4, y0 - i * 1.4, lab, va="center", fontsize=9.5, color=INK))

# ---------------------------------------------------------------- H. convention x splitter
groups = R["groups"]
M = np.array([[R["by_group"][n][g] for n in NAMES] for g in groups])
ax = axH
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("w", [BAD, "#f2c9b8", "#f4f7f5"])
ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(len(groups)):
    for j in range(len(NAMES)):
        ax.text(
            j,
            i,
            f"{M[i, j]:.0%}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="white" if M[i, j] < 0.35 else INK,
        )
ax.set_yticks(range(len(groups)), [f"{g}  ({E.SHARE[g]:.1%})" for g in groups], fontsize=9.5, color=INK)
ax.set_xticks(range(len(NAMES)), NAMES, rotation=30, ha="right", fontsize=9, color=INK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(
    "Share of each convention split exactly right   (declared share of customers in brackets)",
    loc="left",
    fontsize=10.5,
    color=INK,
    pad=10,
)

# ---------------------------------------------------------------- B. rows right vs ceilings
ax = axB
both = [R["per"][n]["both_right"] for n in NAMES]
ax.barh(range(len(NAMES)), both, color=GIV, height=0.6)
for i, v in enumerate(both):
    labels.append(ax.text(v + 0.6, i, f"{v}", va="center", fontsize=9, color=INK))
ax.set_yticks(
    range(len(NAMES)),
    [f"{n}\ngreets {R['per'][n]['greeted_wrong']:.1%} wrong" for n in NAMES],
    fontsize=9,
    color=INK,
)
ax.set_ylim(len(NAMES) - 0.5, -2.3)
for x, lab, y in [
    (R["ceiling"]["by_shape"], "shape ceiling", -1.6),
    (R["ceiling"]["by_string"], "string ceiling", -0.9),
]:
    ax.plot([x, x], [len(NAMES) - 0.5, y + 0.35], color=BAD, lw=1.3, ls="--")
    labels.append(
        ax.text(
            x + (0.6 if y > -1 else 0),
            y,
            f"{lab} {x}",
            ha="left" if y > -1 else "right",
            fontsize=8.5,
            color=BAD,
            va="center",
        )
    )
ax.set_xlim(0, 75)
labels.append(ax.set_xlabel(f"names split exactly right (of {N})", color=INK))
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
labels.append(
    ax.set_title(
        "No splitter reaches what the string allows", loc="left", fontsize=11, color=INK, weight="bold", pad=8
    )
)


def check_geometry() -> None:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    for t in axH.get_xticklabels() + axH.get_yticklabels() + axB.get_yticklabels():
        bb = t.get_window_extent(rend)
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= W and bb.y1 <= H, f"tick off canvas: {t.get_text()!r}"
    boxes = []
    for t in labels:
        bb = t.get_window_extent(rend)
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= W and bb.y1 <= H, f"off canvas: {t.get_text()!r}"
        boxes.append((t.get_text(), bb))
    for i, (ta, a) in enumerate(boxes):
        for tb, b in boxes[i + 1 :]:
            assert not (a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1), (
                f"overlap {ta!r} / {tb!r}"
            )


if __name__ == "__main__":
    check_geometry()
    fig.savefig("audit.png", dpi=150, facecolor=BG)
    fig.savefig("audit.svg", facecolor=BG)
    print("wrote audit.png, audit.svg")
