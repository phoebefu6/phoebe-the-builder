"""The audit figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Top: the routing drawn (agent confidence -> per-segment cutoff -> a human queue with a capacity and a load-dependent
error rate -> overflow back to the agent). Bottom: what one review is worth by segment; seven policies on cost; and
the same-skill model upgrade that cut escalations and raised the bill.
Geometry self-check runs BEFORE savefig: every text inside the figure, no two texts overlapping, titles included.
"""

from __future__ import annotations

import json
import math
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["text.parse_math"] = False  # dollar amounts are text, never mathtext

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
BAD, OK, BLUE, HOT = "#c8562b", "#2e8b57", "#2f6fdb", "#e3b23c"
COL = {"refund": BAD, "account": BLUE, "faq": "#8a94a3"}

R = json.load(open("results.json"))
P = {r["policy"]: r for r in R["policies"]}
B = R["base"]
SEGS = ["refund", "account", "faq"]
JOINT = P["priced + capacity (joint)"]

fig = plt.figure(figsize=(16, 10.4), facecolor=BG)
axD = fig.add_axes([0.03, 0.62, 0.94, 0.31])
axA = fig.add_axes([0.06, 0.07, 0.26, 0.45])
axB = fig.add_axes([0.47, 0.07, 0.20, 0.45])
axC = fig.add_axes([0.75, 0.07, 0.22, 0.45])
texts: List = []

# ---------------------------------------------------------------- D. the routing, drawn
ax = axD
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis("off")
ax.set_title("The routing: one cutoff per segment, a human queue with a ceiling, and overflow that goes back to the agent",
             loc="left", color=INK, fontsize=12, weight="bold")
ax.add_patch(FancyBboxPatch((1, 11), 13, 10, boxstyle="round,pad=0.4,rounding_size=1.2", fc=BG, ec=INK, lw=2))
texts.append(ax.text(7.5, 18.2, "AI agent", ha="center", fontsize=11, color=INK, weight="bold"))
texts.append(ax.text(7.5, 14.6, f"{int(B['tickets']):,} tickets/day", ha="center", fontsize=9, color=MUTED))
ys = {"refund": 22.5, "account": 16, "faq": 9.5}
for s in SEGS:
    y = ys[s]
    seg = R["segments"][s]
    ax.add_patch(FancyArrowPatch((15, 16), (24, y + 1.6), arrowstyle="-|>", mutation_scale=12, color=MUTED))
    ax.add_patch(FancyBboxPatch((24.5, y), 21, 3.4, boxstyle="round,pad=0.3,rounding_size=0.8", fc=BG, ec=COL[s], lw=2))
    cut = JOINT["thr"][s]
    rule = f"escalate if conf < {cut:.3f}  ({JOINT['esc_by_seg'][s]:.0f}/day)" if cut > 0.0 else "never escalate"
    texts.append(ax.text(25.3, y + 1.15, f"{s}  ${seg['cost']:.0f}/error   {rule}", fontsize=9, color=INK))
ax.add_patch(FancyBboxPatch((58, 13), 17, 9, boxstyle="round,pad=0.4,rounding_size=1.2", fc=BG, ec=OK, lw=2.2))
texts.append(ax.text(66.5, 19.4, "human queue", ha="center", fontsize=11, color=INK, weight="bold"))
texts.append(ax.text(66.5, 16.4, f"{int(B['capacity'])}/day, ${B['review_cost']:.0f} each", ha="center", fontsize=9,
                     color=MUTED))
texts.append(ax.text(66.5, 14.0, f"errs {B['human_err']:.0%} calm, {B['human_err_full']:.0%} when full", ha="center",
                     fontsize=9, color=OK))
ax.add_patch(FancyArrowPatch((46.5, 24.2), (57.6, 19.5), arrowstyle="-|>", mutation_scale=14, color=COL["refund"], lw=1.6))
ax.add_patch(FancyArrowPatch((75.8, 17.5), (84, 17.5), arrowstyle="-|>", mutation_scale=14, color=OK))
texts.append(ax.text(84.5, 17.0, "resolved", fontsize=10, color=INK))
ax.add_patch(FancyArrowPatch((66.5, 12.4), (12, 10.3), arrowstyle="-|>", mutation_scale=12, color=HOT, lw=1.4,
                             connectionstyle="arc3,rad=-0.18"))
texts.append(ax.text(40, 2.2, "past capacity, escalations time out and the agent acts anyway", fontsize=9, color=INK))
texts.append(ax.text(1, 27.4, f"joint optimum: {JOINT['escalated']:.0f} escalations/day, all refunds, queue "
                     f"{JOINT['utilisation']:.0%} full, ${JOINT['cost']:,.0f}/day vs ${P['never escalate']['cost']:,.0f} "
                     f"never escalating", fontsize=10, color=INK))

# ---------------------------------------------------------------- A. value of one review
ax = axA
cx = R["centers"]
for s in SEGS:
    pts = [(c, v) for c, v in zip(cx, R["value_curve"][s]) if v is not None and not (isinstance(v, float) and math.isnan(v))]
    ax.plot([c for c, _ in pts], [v for _, v in pts], color=COL[s], lw=2.2)
ax.axhline(0, color=INK, lw=1)
texts.append(ax.text(0.77, 55, "refund", fontsize=10, color=COL["refund"], weight="bold"))
texts.append(ax.text(0.62, 19, "account", fontsize=10, color=COL["account"], weight="bold"))
texts.append(ax.text(0.62, -16, "faq: never pays", fontsize=10, color=COL["faq"], weight="bold"))
ax.set_xlim(0.5, 1.0)
ax.set_ylim(-25, 80)
ax.set_yticks([-20, 0, 20, 40, 60, 80])
ax.set_xlabel("confidence the agent reports", color=INK)
ax.set_ylabel("$ saved by one review (above 0 it pays)", color=INK)
ax.set_title("A. A review is worth the error it prevents,\n    minus its cost and the human's own errors",
             loc="left", color=INK, fontsize=11, weight="bold")

# ---------------------------------------------------------------- B. seven policies
ax = axB
order = ["never escalate", "accuracy rule: conf < 0.90", "staffing rule: fill capacity", "best single threshold",
         "priced, stated confidence", "priced, calibrated", "priced + capacity (joint)"]
short = {"never escalate": "never", "accuracy rule: conf < 0.90": "conf < 0.90",
         "staffing rule: fill capacity": "fill the team", "best single threshold": "best one cutoff",
         "priced, stated confidence": "priced, stated", "priced, calibrated": "priced, calibrated",
         "priced + capacity (joint)": "priced + capacity"}
yy = list(range(len(order)))[::-1]
for y, n in zip(yy, order):
    r = P[n]
    c = OK if n == "priced + capacity (joint)" else (BAD if r["cost"] > P["best single threshold"]["cost"] else "#9aa5b4")
    ax.barh(y, r["cost"] / 1000, 0.62, color=c)
    texts.append(ax.text(r["cost"] / 1000 + 0.6, y, f"${r['cost'] / 1000:.1f}k  ({r['escalated']:.0f} esc)", va="center",
                         fontsize=8.6, color=INK))
ax.set_yticks(yy)
ax.set_yticklabels([short[n] for n in order], color=INK)
ax.set_xlim(0, 75)
ax.set_xlabel("cost per day, $ thousands", color=INK)
ax.set_title("B. Pricing each segment but ignoring\n    capacity loses to one cutoff", loc="left", color=INK,
             fontsize=11, weight="bold")

# ---------------------------------------------------------------- C. the upgrade
ax = axC
U = R["upgrade"]
lab = {"before": "v1", "after": "v2, same\ncutoffs", "refit": "v2,\nrefitted"}
xs = [0, 1, 2]
for x, k in zip(xs, ("before", "after", "refit")):
    r = U[k]
    ax.bar(x, r["cost"] / 1000, 0.55, color=BAD if k == "after" else "#9aa5b4")
    texts.append(ax.text(x, r["cost"] / 1000 + 1.0, f"${r['cost'] / 1000:.1f}k", ha="center", fontsize=9.5, color=INK,
                         weight="bold"))
    texts.append(ax.text(x, 3.0, f"{r['escalation_rate']:.1%}\nescalated", ha="center", fontsize=8.6, color=BG))
ax.set_xticks(xs)
ax.set_xticklabels([lab[k] for k in ("before", "after", "refit")], color=INK)
ax.set_ylim(0, 56)
ax.set_ylabel("cost per day, $ thousands", color=INK)
ax.set_title("C. Same skill, louder confidence:\n    escalations fall, the bill rises 27%", loc="left", color=INK,
             fontsize=11, weight="bold")

for a in (axA, axB, axC):
    a.set_facecolor(BG)
    for sp in a.spines.values():
        sp.set_color(GRID)
    a.tick_params(colors=INK)
axA.grid(True, color=GRID, lw=0.7)
axA.set_axisbelow(True)


def check_geometry() -> None:
    """Every text inside the figure; no two texts overlap (titles and tick labels included)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    allt = list(texts) + [a.title for a in (axA, axB, axC, axD)]
    for a in (axA, axB, axC):
        allt += [t for t in a.get_xticklabels() + a.get_yticklabels() if t.get_text()]
        allt += [a.xaxis.label, a.yaxis.label]
    boxes = []
    for t in allt:
        if not t.get_text():
            continue
        bb = t.get_window_extent(rend)
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= W and bb.y1 <= H, f"text outside figure: {t.get_text()!r}"
        boxes.append((t.get_text(), bb))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 < b.x1 - 1 and b.x0 < a.x1 - 1 and a.y0 < b.y1 - 1 and b.y0 < a.y1 - 1:
                raise AssertionError(f"overlap: {boxes[i][0]!r} / {boxes[j][0]!r}")


if __name__ == "__main__":
    check_geometry()
    fig.savefig("escalation_audit.png", dpi=150, facecolor=BG)
    fig.savefig("escalation_audit.svg", facecolor=BG)
    print("saved escalation_audit.png / .svg, geometry ok")
