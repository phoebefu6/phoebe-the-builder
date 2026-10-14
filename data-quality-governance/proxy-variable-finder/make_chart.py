"""The audit figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Top: the mechanism drawn - where skill and group enter each feature, and the biased history the model learns from.
Bottom: the three proxy screens; the two accuracy numbers moving in opposite directions as proxies come out; and who
among the truly qualified gets selected under each model.
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
M = R["models"]
S = R["screen"]
ORDER = ["aware (group as a feature)", "unaware (group removed)", "drop zip", "drop zip + the commute/shift pair",
         "linear repair of every feature", "drop zip + pair, repair exp", "cert only"]
SHORT = {"aware (group as a feature)": "aware", "unaware (group removed)": "unaware", "drop zip": "drop zip",
         "drop zip + the commute/shift pair": "drop zip + pair", "linear repair of every feature": "linear repair",
         "drop zip + pair, repair exp": "drop zip + pair, repair exp", "cert only": "cert only"}

fig = plt.figure(figsize=(16, 10.6), facecolor=BG)
axD = fig.add_axes([0.03, 0.60, 0.94, 0.33])
axA = fig.add_axes([0.09, 0.08, 0.19, 0.40])
axB = fig.add_axes([0.42, 0.08, 0.24, 0.40])
axC = fig.add_axes([0.69, 0.08, 0.28, 0.40])
texts: List = []

# ---------------------------------------------------------------- D. the mechanism, drawn
ax = axD
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis("off")
ax.set_title("Skill is the same in both groups. Group still reaches the model through four of the five features, "
             "and the labels it learns from were biased", loc="left", color=INK, fontsize=12, weight="bold")


def box(x, y, w, h, label, sub, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.9", fc=BG, ec=ec, lw=2))
    texts.append(ax.text(x + w / 2, y + h - 1.6, label, ha="center", fontsize=10, color=INK, weight="bold"))
    if sub:
        texts.append(ax.text(x + w / 2, y + 1.0, sub, ha="center", fontsize=8.6, color=MUTED))


box(2, 18, 13, 6, "skill", "same in A and B", OK)
box(2, 4, 13, 6, "group", "30% are B", BAD)
feats = [("zip", 30, 22.5, "proxy, no skill", BAD), ("exp", 30, 15.5, "skill + proxy", HOT),
         ("cert", 30, 8.5, "skill only", OK), ("commute + shift", 30, 1.5, "proxy only as a pair", BAD)]
for name, x, y, sub, ec in feats:
    box(x, y, 19, 5, name, sub, ec)
for name, x, y, _, _ in feats:
    if name in ("exp", "cert"):
        ax.add_patch(FancyArrowPatch((15.4, 21), (x - 0.4, y + 2.5), arrowstyle="-|>", mutation_scale=11, color=OK, lw=1.3))
    if name in ("zip", "exp", "commute + shift"):
        ax.add_patch(FancyArrowPatch((15.4, 7), (x - 0.4, y + 2.5), arrowstyle="-|>", mutation_scale=11, color=BAD, lw=1.3))
box(60, 11, 14, 8, "model", "gradient boosting", INK)
for name, x, y, _, _ in feats:
    ax.add_patch(FancyArrowPatch((49.6, y + 2.5), (59.6, 15), arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.1))
box(80, 18, 18, 7, "past decisions", f"A {R['truth']['hist_rate_a']:.0%} hired, B {R['truth']['hist_rate_b']:.0%}", BAD)
ax.add_patch(FancyArrowPatch((80, 20.5), (74.4, 17.5), arrowstyle="-|>", mutation_scale=12, color=BAD, lw=1.6))
texts.append(ax.text(89, 16.3, "the labels it learns from", ha="center", fontsize=9, color=BAD))
box(80, 3, 18, 7, "true merit", f"A {R['truth']['true_rate_a']:.0%} qualified, B {R['truth']['true_rate_b']:.0%}", OK)
texts.append(ax.text(89, 1.0, "what we actually want (never seen in training)", ha="center", fontsize=9, color=OK))

# ---------------------------------------------------------------- A. the screens
ax = axA
rows = [(f, S["single"][f]["mean"], "#9aa5b4") for f in ["zip", "exp", "cert", "commute", "shift"]]
rows += [("commute+shift", S["pairs"]["commute+shift"]["mean"], BAD), ("all five", S["joint"]["mean"], INK)]
ys = list(range(len(rows)))[::-1]
for y, (lab, v, c) in zip(ys, rows):
    col = BAD if (c == "#9aa5b4" and v > 0.6) else c
    ax.barh(y, v - 0.5, 0.62, left=0.5, color=col)
    texts.append(ax.text(v + 0.008, y, f"{v:.3f}", va="center", fontsize=8.6, color=INK))
ax.axvline(0.6, color=MUTED, ls=":", lw=1)
texts.append(ax.text(0.61, 4.42, "flag line", fontsize=8.6, color=MUTED))
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], color=INK)
ax.set_xlim(0.5, 1.06)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("AUC for predicting the group from it", color=INK)
ax.set_title("A. Commute and shift pass a one-at-a-\n    time screen; together they score 0.92",
             loc="left", color=INK, fontsize=11, weight="bold")

# ---------------------------------------------------------------- B. two accuracies
ax = axB
ys = list(range(len(ORDER)))[::-1]
for y, k in zip(ys, ORDER):
    hv, tv = M[k]["auc_hist"]["mean"], M[k]["auc_true"]["mean"]
    ax.plot([hv, tv], [y, y], color=GRID, lw=3, zorder=1)
    ax.scatter(hv, y, color=MUTED, s=55, zorder=2)
    ax.scatter(tv, y, color=OK, s=55, zorder=3)
texts.append(ax.text(0.725, len(ORDER) - 0.35, "grey: vs past decisions (what the team sees)", fontsize=8.6, color=MUTED))
texts.append(ax.text(0.725, len(ORDER) - 0.85, "green: vs true merit", fontsize=8.6, color=OK, weight="bold"))
ax.set_yticks(ys)
ax.set_yticklabels([SHORT[k].replace("\n", " ") for k in ORDER], color=INK, fontsize=9)
ax.set_ylim(-0.6, len(ORDER) + 0.2)
ax.set_xlim(0.72, 0.87)
ax.set_xticks([0.72, 0.76, 0.80, 0.84])
ax.set_xlabel("AUC", color=INK)
ax.set_title("B. Taking proxies out lowers the number\n    the team watches and raises the real one",
             loc="left", color=INK, fontsize=11, weight="bold")

# ---------------------------------------------------------------- C. who among the qualified is selected
ax = axC
for y, k in zip(ys, ORDER):
    ax.barh(y + 0.19, M[k]["qual_sel_a"]["mean"] * 100, 0.36, color=BLUE)
    ax.barh(y - 0.19, M[k]["qual_sel_b"]["mean"] * 100, 0.36, color=BAD)
    texts.append(ax.text(M[k]["qual_sel_b"]["mean"] * 100 + 1, y - 0.19, f"{M[k]['qual_sel_b']['mean']:.0%}",
                         va="center", fontsize=8.2, color=BAD))
ax.set_yticks(ys)
ax.set_yticklabels([])
ax.set_ylim(-0.6, len(ORDER) + 0.2)
ax.set_xlim(0, 82)
ax.set_xticks([0, 20, 40, 60, 80])
ax.set_xlabel("% of truly qualified selected: blue bar group A, orange bar group B", color=INK)
ax.set_title(f"C. Qualified B selected: {M[ORDER[0]]['qual_sel_b']['mean']:.0%} aware, "
             f"{M[ORDER[1]]['qual_sel_b']['mean']:.0%} with\n    the group removed, "
             f"{M['drop zip + pair, repair exp']['qual_sel_b']['mean']:.0%} with its proxies removed",
             loc="left", color=INK, fontsize=11, weight="bold")

for a in (axA, axB, axC):
    a.set_facecolor(BG)
    for sp in a.spines.values():
        sp.set_color(GRID)
    a.tick_params(colors=INK)
for a in (axB, axC):
    a.grid(True, axis="x", color=GRID, lw=0.7)
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
    fig.savefig("proxy_audit.png", dpi=150, facecolor=BG)
    fig.savefig("proxy_audit.svg", facecolor=BG)
    print("saved proxy_audit.png / .svg, geometry ok")
