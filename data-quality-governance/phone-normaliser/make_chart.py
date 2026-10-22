"""The phone-normaliser figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Top: the mechanism drawn - one string a Spanish customer typed, fanned out to the valid numbers it becomes under
different region hints, with the two that collide in this table highlighted. Bottom left: false splits and false
merges per normaliser. Bottom right: the seeded sweep, how many valid numbers a typed string can be.
Geometry self-check runs BEFORE savefig: every text inside the canvas, no two labels overlapping.
"""

from __future__ import annotations

import json
from typing import List

import matplotlib

matplotlib.use("Agg")
import explore as E
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["text.parse_math"] = False
plt.rcParams["font.family"] = ["DejaVu Sans"]

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
SPLIT, MERGE, OK, BAD = "#2f6fdb", "#c8562b", "#2e8b57", "#c8562b"

R = json.load(open("results.json"))["main"]
NAMES = list(E.NORMALISERS)
N = R["n"]
S = "612 345 678"
READ = E.readings(S)

fig = plt.figure(figsize=(16, 11.5), facecolor=BG)
axM = fig.add_axes([0.03, 0.58, 0.94, 0.31])
axP = fig.add_axes([0.25, 0.10, 0.33, 0.38])
axS = fig.add_axes([0.68, 0.10, 0.29, 0.36])
labels: List = []

fig.text(
    0.03,
    0.945,
    "Two formats, one number, two customers: a phone string without a country code is many valid numbers",
    fontsize=15,
    color=INK,
    weight="bold",
)
fig.text(
    0.03,
    0.918,
    f"{N} CRM rows, {R['true_customers']} real customers. The best normaliser gets {max(p['right'] or 0 for p in R['per'].values())}"
    f"/{N} right; only {R['ceiling']['by_string']}/{N} rows say which country they are in.",
    fontsize=11,
    color=MUTED,
)

# ---------------------------------------------------------------- M. one string, many numbers
ax = axM
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 12)
ax.add_patch(
    FancyBboxPatch((2, 5), 17, 2, boxstyle="round,pad=0.1,rounding_size=0.5", fc="#f4f6f9", ec=INK, lw=1.6)
)
labels.append(
    ax.text(10.5, 6, S, ha="center", va="center", fontsize=17, color=INK, family="DejaVu Sans Mono")
)
labels.append(
    ax.text(
        10.5,
        3.9,
        "typed by a Spanish customer\nwhose record says FR",
        ha="center",
        va="top",
        fontsize=10,
        color=MUTED,
    )
)
show = ["ES", "FR", "CH", "DE", "FI", "CL", "PL", "DZ"]
picks = [(r, e) for e, rs in READ.items() for r in rs if r in show]
picks.sort(key=lambda x: show.index(x[0]))
for i, (reg, e) in enumerate(picks):
    y = 11.0 - i * 1.4
    hot = reg in ("ES", "FR")
    colr = BAD if hot else MUTED
    ax.annotate(
        "",
        xy=(40, y),
        xytext=(19.6, 6),
        arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.6 if hot else 0.9, connectionstyle="arc3,rad=0.0"),
    )
    labels.append(
        ax.text(40.8, y, f"{reg}", va="center", fontsize=11, color=colr, weight="bold" if hot else None)
    )
    labels.append(ax.text(45, y, e, va="center", fontsize=12, color=INK, family="DejaVu Sans Mono"))
labels.append(ax.text(64, 11.0, "truth: the Spanish mobile", va="center", fontsize=10, color=OK))
labels.append(
    ax.text(
        64,
        9.6,
        "what region=country field files it as;\nthe same number as the French customer "
        "who typed 06 12 34 56 78",
        va="center",
        fontsize=10,
        color=BAD,
    )
)
labels.append(
    ax.text(
        64,
        6.0,
        f"... and {len(READ) - len(picks)} more. This string is a valid number\n"
        f"in {len(READ)} different ways across {sum(len(v) for v in READ.values())} regions "
        "(libphonenumber metadata).",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )
)
labels.append(
    ax.text(
        64,
        2.3,
        "Validity is about the shape of a numbering plan,\nnot whether the line exists. "
        "A + and a country code\nis the only part of the string that settles it.",
        va="center",
        fontsize=10.5,
        color=INK,
    )
)

# ---------------------------------------------------------------- P. splits and merges per normaliser
ax = axP
y = list(range(len(NAMES)))
sp = [len(R["per"][n]["false_splits"]) for n in NAMES]
mg = [len(R["per"][n]["false_merges"]) for n in NAMES]
ax.barh([v - 0.18 for v in y], sp, height=0.34, color=SPLIT, label="false split: one customer becomes two")
ax.barh([v + 0.18 for v in y], mg, height=0.34, color=MERGE, label="false merge: two customers become one")
for i in y:
    labels.append(ax.text(sp[i] + 0.5, i - 0.18, str(sp[i]), va="center", fontsize=9, color=INK))
    labels.append(ax.text(mg[i] + 0.5, i + 0.18, str(mg[i]), va="center", fontsize=9, color=INK))


def _tick(n: str) -> str:
    p = R["per"][n]
    right = "key only" if p["right"] is None else f"{p['right']}/{N} right, {p['refused']} refused"
    return f"{n}\n{right}, {p['customers']} customers (truth {R['true_customers']})"


ax.set_yticks(y, [_tick(n) for n in NAMES], fontsize=8.8, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, 52)
labels.append(
    ax.set_xlabel(f"row pairs (of {R['per'][NAMES[0]]['true_pairs']} true duplicate pairs)", color=INK)
)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower right", fontsize=9)
labels.append(
    ax.set_title(
        "Every normaliser errs; refusing is the only zero",
        loc="left",
        fontsize=11,
        color=INK,
        weight="bold",
        pad=8,
    )
)

# ---------------------------------------------------------------- S. sweep histogram
ax = axS
cols = [SPLIT, MERGE]
for (lab, sw), c in zip(R["sweep"].items(), cols):
    xs = sorted(int(k) for k in sw["hist"])
    ax.plot(
        xs,
        [sw["hist"][str(x)] if str(x) in sw["hist"] else sw["hist"][x] for x in xs],
        color=c,
        lw=2,
        label=f"{lab}: median {sw['median_numbers']}",
    )
top = max(max(sw["hist"].values()) for sw in R["sweep"].values())
ax.set_ylim(0, top * 1.45)
ax.axvline(1, color=OK, lw=1.3, ls="--")
labels.append(ax.text(1.8, top * 1.08, "1 = unambiguous", color=OK, fontsize=9))
ax.set_xlim(0, 56)
labels.append(ax.set_xlabel("distinct valid numbers one typed string can be", color=INK))
labels.append(ax.set_ylabel(f"strings (of {next(iter(R['sweep'].values()))['n']})", color=INK))
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
labels.append(
    ax.set_title(
        "Seeded random strings: none had one reading",
        loc="left",
        fontsize=11,
        color=INK,
        weight="bold",
        pad=8,
    )
)


def check_geometry() -> None:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    for t in axP.get_yticklabels() + axP.get_xticklabels() + axS.get_xticklabels():
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
    # legends must not sit on bars or lines (text vs artists, the Day 203 defect)
    for ax_, leg in ((axP, axP.get_legend()), (axS, axS.get_legend())):
        lb = leg.get_window_extent(rend)
        for t in labels:
            if t.axes is ax_:
                b = t.get_window_extent(rend)
                assert not (lb.x0 < b.x1 and b.x0 < lb.x1 and lb.y0 < b.y1 and b.y0 < lb.y1), (
                    f"legend on {t.get_text()!r}"
                )
    lb = axS.get_legend().get_window_extent(rend)
    for ln in axS.get_lines():
        for px, py in ln.get_transform().transform(ln.get_xydata()):
            assert not (lb.x0 <= px <= lb.x1 and lb.y0 <= py <= lb.y1), "legend covers a line point"
    lb = axP.get_legend().get_window_extent(rend)
    for p in axP.patches:
        b = p.get_window_extent(rend)
        assert not (lb.x0 < b.x1 and b.x0 < lb.x1 and lb.y0 < b.y1 and b.y0 < lb.y1), "legend covers a bar"


if __name__ == "__main__":
    check_geometry()
    fig.savefig("audit.png", dpi=150, facecolor=BG)
    fig.savefig("audit.svg", facecolor=BG)
    print("wrote audit.png, audit.svg")
