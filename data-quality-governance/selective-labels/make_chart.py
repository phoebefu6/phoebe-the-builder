"""The selective-labels figure. Every bar and line read from results.json, so the chart cannot drift from evidence.txt.

Top: the mechanism drawn - 1,500 applicants from the declared world (seed 200) on the two signals the officers
used. Filled dots are the approved: the only outcomes anyone ever sees. The model is handed the horizontal axis
only, and group B's approval line sits 0.6 sd further out.
Bottom: the same two models as five estimators report them; and how big a random-explore slice must be.
Geometry self-check runs BEFORE savefig: every text inside the figure, no two texts overlapping.
"""

from __future__ import annotations

import json
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import selective as S

plt.rcParams["text.parse_math"] = False

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
BAD, OK, BLUE, HOT = "#c8562b", "#2e8b57", "#2f6fdb", "#e3b23c"

R = json.load(open("results.json"))["main"]
EST = [
    ("truth", "truth\n(all outcomes)"),
    ("labeled", "labeled\nrows only"),
    ("rejected_bad", "rejected\n= bad"),
    ("ipw", "IPW on\nx, group"),
    ("explore_0.05", "5% random\nexplore"),
]

fig = plt.figure(figsize=(16, 11), facecolor=BG)
axD = fig.add_axes([0.06, 0.55, 0.40, 0.37])
axN = fig.add_axes([0.50, 0.55, 0.47, 0.37])
axA = fig.add_axes([0.06, 0.07, 0.44, 0.36])
axB = fig.add_axes([0.58, 0.07, 0.39, 0.36])
texts: List = []

# ---------------------------------------------------------------- D. the mechanism, drawn from the world
ax = axD
d = S.draw(1_500, np.random.default_rng(200))
big = S.draw(200_000, np.random.default_rng(1))
s = big["x"] + big["u"] - S.WORLD["bar_b"] * big["g"]
t = float(np.quantile(s, 1 - S.APPROVE))
for gk, mk in ((0, "o"), (1, "^")):
    m = d["g"] == gk
    lab, rej = m & d["approved"], m & ~d["approved"]
    ax.scatter(d["x"][rej], d["u"][rej], s=12, marker=mk, facecolors="none", edgecolors="#b7bfc9", lw=0.7)
    ax.scatter(d["x"][lab], d["u"][lab], s=14, marker=mk, c=np.where(d["y"][lab] == 1, OK, BAD), lw=0)
xx = np.linspace(-3.5, 3.5, 2)
ax.plot(xx, t - xx, color=INK, lw=1.4)
ax.plot(xx, t + S.WORLD["bar_b"] - xx, color=INK, lw=1.4, ls="--")
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_xlabel("bureau score x (recorded: the model sees this)")
ax.set_ylabel("interviewer's read u (never recorded)")
ax.set_title("Outcomes exist only above the line", loc="left", fontsize=12, color=INK, weight="bold")
BOX = dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, alpha=0.92)
texts.append(
    ax.text(
        -3.3,
        -3.3,
        "filled = approved, outcome seen (green repaid, orange defaulted)\n"
        "hollow = rejected, outcome never seen\n"
        "solid line = group A cut-off; dashed = group B, 0.6 sd higher",
        fontsize=8.6,
        color=INK,
        va="bottom",
        bbox=BOX,
    )
)
texts.append(
    ax.annotate(
        "approved at a low x\n= had a high u",
        xy=(-0.95, 2.15),
        xytext=(-3.35, 1.4),
        fontsize=9,
        color=BAD,
        va="center",
        bbox=BOX,
        arrowprops=dict(arrowstyle="-|>", color=BAD, lw=1.2),
    )
)
ax.spines[["top", "right"]].set_visible(False)

# ---------------------------------------------------------------- N. the narrative numbers
ax = axN
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title(
    "What the label set does to a model trained and judged on it",
    loc="left",
    fontsize=12,
    color=INK,
    weight="bold",
)
o, lm = R["eval_full_model"], R["eval_lab_model"]
rows = [
    (
        f"{R['pop_repay']['mean']:.0%}",
        "of all applicants repay;",
        f"{R['hist_repay_labeled']['mean']:.0%} of the labeled rows do",
        INK,
    ),
    (
        f"{R['b_labeled_repay']['mean']:.0%}",
        "of labeled group B repay vs",
        f"{R['a_labeled_repay']['mean']:.0%} of A: the higher bar reads as lower risk",
        BAD,
    ),
    (
        f"{R['share_b_approved']['labeled']['mean']:.0%}",
        "of B approved by the label-trained model;",
        f"{R['share_b_approved']['oracle']['mean']:.0%} by one trained on all outcomes",
        BAD,
    ),
    (
        f"{lm['truth']['repay']['mean']:.1%}",
        "true repay among its approvals, vs",
        f"{o['truth']['repay']['mean']:.1%} for the oracle",
        INK,
    ),
    (
        f"{lm['labeled']['repay']['mean']:.1%}",
        "what the labeled rows report for it, vs",
        f"{o['labeled']['repay']['mean']:.1%}: the dashboard ranks the worse model first",
        BAD,
    ),
]
for i, (big_n, a, b, col) in enumerate(rows):
    y = 0.86 - i * 0.18
    texts.append(ax.text(0.0, y, big_n, fontsize=21, color=col, weight="bold", va="center"))
    texts.append(ax.text(0.17, y + 0.035, a, fontsize=10.5, color=INK, va="center"))
    texts.append(ax.text(0.17, y - 0.035, b, fontsize=10.5, color=MUTED, va="center"))

# ---------------------------------------------------------------- A. five estimators, two models
ax = axA
xs = np.arange(len(EST))
ov = [o[k]["repay"]["mean"] * 100 for k, _ in EST]
lv = [lm[k]["repay"]["mean"] * 100 for k, _ in EST]
ax.bar(xs - 0.2, ov, 0.38, color=OK, label="oracle model (truly better)")
ax.bar(xs + 0.2, lv, 0.38, color=HOT, label="label-trained model")
for x, a, b in zip(xs, ov, lv):
    texts.append(ax.text(x - 0.2, a + 1.0, f"{a:.0f}", ha="center", fontsize=9, color=INK))
    texts.append(ax.text(x + 0.2, b + 1.0, f"{b:.0f}", ha="center", fontsize=9, color=INK))
ax.axhline(o["truth"]["repay"]["mean"] * 100, color=OK, lw=1, ls=":")
ax.set_xticks(list(xs))
ax.set_xticklabels([n for _, n in EST], fontsize=9)
ax.set_ylim(40, 108)
ax.set_ylabel("repay rate among approvals (%)")
ax.set_title("Labeled rows rank the worse model first; IPW cannot see u", loc="left", fontsize=11, color=INK)
leg = ax.legend(frameon=False, loc="upper right", fontsize=9, ncol=2)
ax.spines[["top", "right"]].set_visible(False)

# ---------------------------------------------------------------- B. explore slice size
ax = axB
ex = [e * 100 for e in S.EXPLORE]
est = [o[f"explore_{e:g}"]["repay"]["mean"] * 100 for e in S.EXPLORE]
ci = [o[f"explore_{e:g}"]["ci_half"]["mean"] * 100 for e in S.EXPLORE]
tr = o["truth"]["repay"]["mean"] * 100
ax.errorbar(ex, est, yerr=ci, fmt="o", color=BLUE, capsize=5, lw=1.6)
ax.axhline(tr, color=OK, ls="--", lw=1.4)
ax.axhline(o["labeled"]["repay"]["mean"] * 100, color=BAD, lw=1.2)
ax.axhline(o["ipw"]["repay"]["mean"] * 100, color=MUTED, lw=1.2, ls="-.")
texts.append(ax.text(2.6, tr + 0.6, f"truth {tr:.1f}%", color=OK, fontsize=9, ha="left"))
texts.append(
    ax.text(
        10.5, o["labeled"]["repay"]["mean"] * 100 + 0.7, "labeled rows", color=BAD, fontsize=9, ha="right"
    )
)
texts.append(ax.text(10.5, o["ipw"]["repay"]["mean"] * 100 - 1.9, "IPW", color=MUTED, fontsize=9, ha="right"))
for x, y, c in zip(ex, est, ci):
    texts.append(ax.text(x + 0.25, y - c - 1.8, f"+/-{c:.1f}", color=BLUE, fontsize=9, ha="left"))
ax.set_xticks(ex)
ax.set_xticklabels([f"{e:g}%" for e in ex])
ax.set_xlim(0, 11)
ax.set_ylim(68, 98)
ax.set_xlabel("share of applicants approved at random (the explore slice)")
ax.set_ylabel("oracle repay estimate (%), 95% CI")
ax.set_title("Only random approvals measure it; 5% gets +/-2.6 pts", loc="left", fontsize=11, color=INK)
ax.spines[["top", "right"]].set_visible(False)


def check_geometry() -> None:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    Wd, Ht = fig.bbox.width, fig.bbox.height
    allt = list(texts) + [a.title for a in (axD, axN, axA, axB)] + list(leg.get_texts())
    for a in (axD, axA, axB):
        allt += [tk for tk in a.get_xticklabels() + a.get_yticklabels() if tk.get_text() and tk.get_visible()]
        allt += [a.xaxis.label, a.yaxis.label]
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
    # legend box vs bar labels: a text-vs-text check cannot see the legend frame, so test its extent too
    lb = leg.get_window_extent(rend)
    for tx in texts:
        if tx.axes is axA:
            bb = tx.get_window_extent(rend)
            assert not (bb.x0 < lb.x1 and lb.x0 < bb.x1 and bb.y0 < lb.y1 and lb.y0 < bb.y1), (
                f"legend covers {tx.get_text()!r}"
            )
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 < b.x1 - 1 and b.x0 < a.x1 - 1 and a.y0 < b.y1 - 1 and b.y0 < a.y1 - 1:
                raise AssertionError(f"overlap: {boxes[i][0]!r} / {boxes[j][0]!r}")


check_geometry()
fig.savefig("audit.png", dpi=150, facecolor=BG)
fig.savefig("audit.svg", facecolor=BG)
print("wrote audit.png / audit.svg")
