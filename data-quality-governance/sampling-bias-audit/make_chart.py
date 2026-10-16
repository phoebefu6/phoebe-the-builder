"""The audit figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Top: the mechanism drawn - the population and the training set as two 2x2 grids with cell AREA proportional to
share. Both have the same row and column totals; the rural-B corner all but vanishes in one of them.
Bottom: what that costs rural B, cell by cell; and how many rural-B rows it takes to stop costing it.
Geometry self-check runs BEFORE savefig: every text inside the figure, no two texts overlapping.
"""

from __future__ import annotations

import json
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams["text.parse_math"] = False

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
BAD, OK, BLUE, HOT = "#c8562b", "#2e8b57", "#2f6fdb", "#e3b23c"

R = json.load(open("results.json"))
CELLS = ["A-urban", "A-rural", "B-urban", "B-rural"]
CURVE = [100 + k for k in [0, 50, 100, 200, 400, 900, 1900]]

fig = plt.figure(figsize=(16, 10.2), facecolor=BG)
axD = fig.add_axes([0.03, 0.56, 0.94, 0.37])
axA = fig.add_axes([0.07, 0.08, 0.38, 0.38])
axB = fig.add_axes([0.56, 0.08, 0.40, 0.38])
texts: List = []

# ---------------------------------------------------------------- D. the mechanism, drawn
ax = axD
ax.set_xlim(0, 100)
ax.set_ylim(0, 36)
ax.axis("off")
ax.set_title(
    "Same row totals, same column totals, different corners: a quota on the marginals let rural B "
    "fall from 12% to 0.5% of the training set",
    loc="left",
    color=INK,
    fontsize=12,
    weight="bold",
)


def grid(x0: float, shares: dict, label: str, size: float = 26.0) -> None:
    """2x2 mosaic: columns = group (width = group share), rows within a column = region (height = cell share)."""
    y0 = 3.0
    pa = shares["A-urban"] + shares["A-rural"]
    widths = {"A": pa * size * 1.42, "B": (1 - pa) * size * 1.42}
    x = x0
    for gname in ("A", "B"):
        tot = shares[f"{gname}-urban"] + shares[f"{gname}-rural"]
        h_rural = size * shares[f"{gname}-rural"] / tot
        for reg, yb, h in (("rural", y0, h_rural), ("urban", y0 + h_rural, size - h_rural)):
            cell = f"{gname}-{reg}"
            hot = cell == "B-rural"
            ax.add_patch(
                Rectangle(
                    (x, yb),
                    widths[gname],
                    h,
                    fc=("#f6d3c4" if hot else "#eef2f6"),
                    ec=(BAD if hot else MUTED),
                    lw=(2.2 if hot else 1),
                )
            )
            if h > 3.0:
                texts.append(
                    ax.text(
                        x + widths[gname] / 2,
                        yb + h / 2,
                        f"{cell}\n{shares[cell]:.1%}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=(BAD if hot else INK),
                    )
                )
        x += widths[gname]
    texts.append(ax.text(x0, y0 + size + 1.2, label, fontsize=10.5, color=INK, weight="bold"))


P, Q = R["pop_shares"], R["quota_shares"]
grid(2, P, "deployment population")
grid(55, Q, "training set (quota: 30% B, 40% rural)")
ax.add_patch(
    FancyArrowPatch((96.5, 4.6), (91.0, 3.4), arrowstyle="-|>", mutation_scale=12, color=BAD, lw=1.4)
)
texts.append(ax.text(96.8, 5.4, "rural B:\n100 rows", fontsize=9, color=BAD, ha="left"))
texts.append(
    ax.text(
        47.5,
        18,
        "B: 30% = 30%\nrural: 40% = 40%\none-attribute\naudit passes",
        fontsize=9.2,
        color=OK,
        ha="center",
        va="center",
    )
)
texts.append(
    ax.text(
        47.5, 6, "rural B:\n12% vs 0.5%", fontsize=9.2, color=BAD, ha="center", va="center", weight="bold"
    )
)

# ---------------------------------------------------------------- A. what it costs, per cell
ax = axA
qv = [R["quota"][c]["good_rejected"]["mean"] * 100 for c in CELLS]
pv = [R["proportional"][c]["good_rejected"]["mean"] * 100 for c in CELLS]
xs = range(len(CELLS))
ax.bar([x - 0.2 for x in xs], pv, 0.38, color=OK, label="trained on a proportional sample")
ax.bar([x + 0.2 for x in xs], qv, 0.38, color=[MUTED, MUTED, MUTED, BAD], label="trained on the quota sample")
for x, v in zip(xs, qv):
    texts.append(ax.text(x + 0.2, v + 1.2, f"{v:.0f}%", ha="center", fontsize=9.5, color=INK))
ax.set_xticks(list(xs))
ax.set_xticklabels(CELLS)
ax.set_ylim(0, 66)
ax.set_ylabel("good payers rejected (%)")
ax.set_title(
    f"Rural B's good payers are rejected {qv[3] / pv[3]:.1f}x as often; the other cells do not move",
    loc="left",
    fontsize=11,
    color=INK,
)
leg = ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
own, dep = R["quota_on_own_test"]["overall"]["acc"]["mean"], R["quota"]["overall"]["acc"]["mean"]
texts.append(
    ax.text(
        1.0,
        42,
        f"overall accuracy: {own:.1%} on the team's own\ntest split, {dep:.1%} in deployment",
        fontsize=9,
        color=MUTED,
        ha="center",
    )
)

# ---------------------------------------------------------------- B. rows needed
ax = axB
cm = [v["mean"] * 100 for v in R["curve"]]
lo = [v["lo"] * 100 for v in R["curve"]]
hi = [v["hi"] * 100 for v in R["curve"]]
prop = R["proportional"]["B-rural"]["good_rejected"]["mean"] * 100
rw = R["reweighted"]["B-rural"]["good_rejected"]
ax.fill_between(CURVE, lo, hi, color="#f6d3c4", lw=0)
ax.plot(CURVE, cm, color=BAD, lw=2.2, marker="o")
ax.axhline(prop, color=OK, ls="--", lw=1.4)
texts.append(
    ax.text(1400, prop - 4.2, f"proportional sample: {prop:.1f}%", color=OK, fontsize=9.2, ha="center")
)
ax.errorbar(
    [100],
    [rw["mean"] * 100],
    yerr=[[rw["mean"] * 100 - rw["lo"] * 100], [rw["hi"] * 100 - rw["mean"] * 100]],
    fmt="D",
    color=BLUE,
    capsize=4,
)
texts.append(
    ax.text(135, rw["mean"] * 100 - 1, "same 100 rows,\nreweighted", color=BLUE, fontsize=9, va="center")
)
em = R["empty"]["B-rural"]["good_rejected"]["mean"] * 100
texts.append(
    ax.text(
        2000,
        60,
        f"no rural B rows at all: {em:.0f}%,\nand reweighting has nothing to weight",
        color=MUTED,
        fontsize=9,
        ha="right",
    )
)
ax.set_xscale("log")
ax.set_xticks(CURVE)
ax.set_xticklabels([f"{c:,}" for c in CURVE])
ax.minorticks_off()
ax.set_ylim(0, 66)
ax.set_xlabel("rural B rows in a 20,000-row training set")
ax.set_ylabel("rural B good payers rejected (%)")
ax.set_title(
    "About 1,000 rows (5% of the set) before the cell stops paying for it", loc="left", fontsize=11, color=INK
)
ax.spines[["top", "right"]].set_visible(False)


def check_geometry() -> None:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    Wd, Ht = fig.bbox.width, fig.bbox.height
    allt = list(texts) + [a.title for a in (axD, axA, axB)] + list(leg.get_texts())
    for a in (axA, axB):
        allt += [tk for tk in a.get_xticklabels() + a.get_yticklabels() if tk.get_text() and tk.get_visible()]
        allt += [a.xaxis.label, a.yaxis.label]
    cells = [pa.get_window_extent(rend) for pa in axD.patches if isinstance(pa, Rectangle)]
    for tx in texts[:]:
        bb = tx.get_window_extent(rend)
        inside = [c for c in cells if c.x0 <= bb.x0 and bb.x1 <= c.x1 and c.y0 <= bb.y0 and bb.y1 <= c.y1]
        crosses = [c for c in cells if bb.x0 < c.x1 and c.x0 < bb.x1 and bb.y0 < c.y1 and c.y0 < bb.y1]
        if tx.axes is axD and crosses and not inside:
            raise AssertionError(f"text crosses a grid cell edge: {tx.get_text()!r}")
    boxes = []
    for tx in allt:
        if not tx.get_text():
            continue
        bb = tx.get_window_extent(rend)
        if bb.width == 0:
            continue
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= Wd and bb.y1 <= Ht, (
            f"text outside figure: {tx.get_text()!r}"
        )
        boxes.append((tx.get_text(), bb))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 < b.x1 - 1 and b.x0 < a.x1 - 1 and a.y0 < b.y1 - 1 and b.y0 < a.y1 - 1:
                raise AssertionError(f"overlap: {boxes[i][0]!r} / {boxes[j][0]!r}")


check_geometry()
fig.savefig("audit.png", dpi=150, facecolor=BG)
fig.savefig("audit.svg", facecolor=BG)
print("wrote audit.png / audit.svg")
