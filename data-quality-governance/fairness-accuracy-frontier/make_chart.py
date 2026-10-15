"""The frontier figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Top: the mechanism drawn - one real skill gap, one biased ruler, and the cut-off the fairness constraint moves.
Bottom: the frontier measured against both rulers; the price of parity as the ruler gets more biased; and how wide
the price looks when estimated on a validation sample.
Geometry self-check runs BEFORE savefig: every text inside the figure, no two texts overlapping, titles included.
"""

from __future__ import annotations

import json
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["text.parse_math"] = False

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
BAD, OK, BLUE, HOT = "#c8562b", "#2e8b57", "#2f6fdb", "#e3b23c"

R = json.load(open("results.json"))
FR, P, PR, SW = R["frontier"], R["points"], R["prices"], R["sweep"]
W = R["world"]

fig = plt.figure(figsize=(16, 10.6), facecolor=BG)
axD = fig.add_axes([0.03, 0.62, 0.94, 0.31])
axA = fig.add_axes([0.06, 0.08, 0.36, 0.44])
axB = fig.add_axes([0.49, 0.08, 0.22, 0.44])
axC = fig.add_axes([0.78, 0.08, 0.19, 0.44])
texts: List = []

# ---------------------------------------------------------------- D. the mechanism, drawn
ax = axD
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis("off")
ax.set_title("One real skill gap, one biased ruler: the fairness constraint moves group B's cut-off, and the "
             "price depends on which ruler scores the move", loc="left", color=INK, fontsize=12, weight="bold")


def box(x, y, w, h, label, sub, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.9", fc=BG, ec=ec, lw=2))
    texts.append(ax.text(x + w / 2, y + h - 1.6, label, ha="center", fontsize=10, color=INK, weight="bold"))
    if sub:
        texts.append(ax.text(x + w / 2, y + 1.0, sub, ha="center", fontsize=8.6, color=MUTED))


def arrow(a, b, c, lw=1.4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=c, lw=lw))


box(2, 17, 17, 7, "true skill", f"B lower by {W['skill_gap']:.2f} sd (real)", OK)
box(2, 3, 17, 7, "past decisions", f"held B {W['label_bias']:.1f} sd higher", BAD)
arrow((10.5, 16.6), (10.5, 10.4), BAD)
texts.append(ax.text(11.5, 13.2, "minus a bias", fontsize=8.6, color=BAD))
box(28, 10, 15, 7, "model", "learns the history", INK)
arrow((19.4, 6.5), (27.6, 12.5), BAD)
box(52, 10, 20, 7, "select 30%", "cut-off for A, cut-off for B", INK)
arrow((43.4, 13.5), (51.6, 13.5), MUTED)
texts.append(ax.text(62, 21.2, "fairness = lower B's cut-off,", ha="center", fontsize=9, color=BLUE, weight="bold"))
texts.append(ax.text(62, 19.0, "raise A's, total stays 30%", ha="center", fontsize=9, color=BLUE))
box(80, 18, 18, 7, "ruler 1: history", "what the team can score", BAD)
box(80, 3, 18, 7, "ruler 2: truth", "what the team wants", OK)
arrow((72.4, 15), (79.6, 21), MUTED)
arrow((72.4, 12), (79.6, 6.5), MUTED)
texts.append(ax.text(89, 15.6, "parity: -" + f"{PR['parity_hit_hist']['mean']:.1f} pts hit rate", ha="center",
                     fontsize=9, color=BAD, weight="bold"))
texts.append(ax.text(89, 0.6, "parity: +" + f"{-PR['parity_hit_true']['mean']:.1f} pts hit rate", ha="center",
                     fontsize=9, color=OK, weight="bold"))

# ---------------------------------------------------------------- A. the frontier, two rulers
ax = axA
x = FR["ratio"]
ax.plot(x, [v * 100 for v in FR["hit_hist"]], color=BAD, lw=2.4)
ax.plot(x, [v * 100 for v in FR["hit_true"]], color=OK, lw=2.4)
for k, short, dy in [("aware (group as a feature)", "aware", -1.0), ("unaware (group removed)", "unaware", -1.0),
                     ("reweighed (Kamiran-Calders)", "reweighed", -1.0)]:
    p = P[k]
    ax.scatter(p["ratio"]["mean"], p["hit_hist"]["mean"] * 100, s=40, color=INK, zorder=4)
    texts.append(ax.text(p["ratio"]["mean"], p["hit_hist"]["mean"] * 100 + dy, short, ha="center", va="top",
                         fontsize=8.6, color=INK))
for r, lab in ((0.8, "four-fifths"), (1.0, "parity")):
    ax.axvline(r, color=MUTED, ls=":", lw=1)
    texts.append(ax.text(r + 0.012, 67.6, lab, fontsize=8.6, color=MUTED))
texts.append(ax.text(0.10, 52.2, "hit rate against past decisions (ruler 1)", fontsize=9, color=BAD, weight="bold"))
texts.append(ax.text(0.40, 67.0, "against true merit (ruler 2)", fontsize=9, color=OK, weight="bold"))
texts.append(ax.text(0.40, 55.6, "dots: one cut-off for all;\nwithin 0.5 pts of the curve", fontsize=8.6, color=INK))
ax.set_xlim(0.05, 1.25)
ax.set_ylim(51, 69)
ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
ax.set_xlabel("selection ratio, rate(B) / rate(A)", color=INK)
ax.set_ylabel("hit rate of the selected 30% (%)", color=INK)
ax.set_title(f"A. Moving to parity costs {PR['parity_hit_hist']['mean']:.1f} pts on the ruler the team\n"
             f"    has and gains {-PR['parity_hit_true']['mean']:.1f} on the one it wants", loc="left",
             color=INK, fontsize=11, weight="bold")

# ---------------------------------------------------------------- B. bias sweep
ax = axB
for i, row in enumerate(SW):
    h, t = -row["parity_hit_hist"]["mean"], -row["parity_hit_true"]["mean"]
    ax.bar(i - 0.2, h, 0.38, color=BAD)
    ax.bar(i + 0.2, t, 0.38, color=OK)
    texts.append(ax.text(i - 0.2, h - 0.25, f"{h:+.1f}", ha="center", va="top", fontsize=8.4, color=BAD))
    texts.append(ax.text(i + 0.2, t + 0.25 if t >= 0 else t - 0.25, f"{t:+.1f}", ha="center",
                         va="bottom" if t >= 0 else "top", fontsize=8.4, color=OK))
texts.append(ax.text(-0.5, -6.6, "orange: vs history", fontsize=8.6, color=BAD, weight="bold"))
texts.append(ax.text(-0.5, -7.4, "green: vs truth", fontsize=8.6, color=OK, weight="bold"))
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(range(len(SW)))
ax.set_xticklabels([f"{row['label_bias']['mean']:.2f}" + ("\ncontrol" if i == 0 else "") for i, row in enumerate(SW)],
                   color=INK)
ax.set_xlim(-0.6, len(SW) - 0.4)
ax.set_ylim(-8.8, 4.6)
ax.set_xlabel("how biased the history is (label_bias, sd)", color=INK)
ax.set_ylabel("change in hit rate, unconstrained -> parity (pts)", color=INK)
ax.set_title("B. With an honest history parity\n    costs 0.4 pts; the rest is the ruler", loc="left", color=INK,
             fontsize=11, weight="bold")

# ---------------------------------------------------------------- C. the estimate on a sample
ax = axC
A = R["audit"]
for i, m in enumerate(A):
    v = A[m]
    ax.plot([i, i], [v["p05"]["mean"], v["p95"]["mean"]], color=BLUE, lw=6, solid_capstyle="butt")
    ax.scatter(i, v["p50"]["mean"], color=INK, s=50, zorder=3)
    texts.append(ax.text(i + 0.12, v["p95"]["mean"], f"{v['p95']['mean']:.1f}", va="center", fontsize=8.4, color=INK))
    texts.append(ax.text(i + 0.12, v["p05"]["mean"], f"{v['p05']['mean']:.1f}", va="center", fontsize=8.4, color=INK))
ax.set_xticks(range(len(A)))
ax.set_xticklabels([f"{int(m):,} rows" for m in A], color=INK)
ax.set_xlim(-0.5, len(A) - 0.3)
ax.set_ylim(0, 6)
ax.set_ylabel("estimated cost of four-fifths (pts)", color=INK)
small = A[min(A, key=int)]
ax.set_title(f"C. On {int(min(A, key=int)):,} rows the same\n    cost reads {small['p05']['mean']:.1f} to {small['p95']['mean']:.1f}", loc="left", color=INK, fontsize=11, weight="bold")

for a in (axA, axB, axC):
    a.set_facecolor(BG)
    for sp in a.spines.values():
        sp.set_color(GRID)
    a.tick_params(colors=INK)
    a.grid(True, axis="y", color=GRID, lw=0.7)
    a.set_axisbelow(True)


def check_geometry() -> None:
    """Every text inside the figure; no two texts overlap (titles and tick labels included)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    Wd, Ht = fig.bbox.width, fig.bbox.height
    axes = (axD, axA, axB, axC)
    allt = list(texts) + [a.title for a in axes]
    for a in (axA, axB, axC):
        allt += [tk for tk in a.get_xticklabels() + a.get_yticklabels() if tk.get_text() and tk.get_visible()]
        allt += [a.xaxis.label, a.yaxis.label]
    boxes = []
    for tx in allt:
        if not tx.get_text():
            continue
        bb = tx.get_window_extent(rend)
        if bb.width == 0:
            continue
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= Wd and bb.y1 <= Ht, f"text outside figure: {tx.get_text()!r}"
        boxes.append((tx.get_text(), bb))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 < b.x1 - 1 and b.x0 < a.x1 - 1 and a.y0 < b.y1 - 1 and b.y0 < a.y1 - 1:
                raise AssertionError(f"overlap: {boxes[i][0]!r} / {boxes[j][0]!r}")


if __name__ == "__main__":
    check_geometry()
    fig.savefig("frontier.png", dpi=150, facecolor=BG)
    fig.savefig("frontier.svg", facecolor=BG)
    print("saved frontier.png / .svg, geometry ok")
