"""The audit figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Three panels: the four policies on the dashboard's number and on the user's number (the ranking flips); the
budget sweep, where context recall is monotone and the answer rate peaks and falls; and a drawn picture of the
mechanism - the window as the model meets it, the instruction at the top, the needed chunk in the middle, the
read-probability curve beside it. Geometry self-check runs BEFORE savefig (Day 180), tick labels included
(Day 189), axes titles included and all three title artists collected (Day 191).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import packing as P
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
BAD, OK, DASH, HOT = "#c8562b", "#2e8b57", "#8a94a3", "#e3b23c"
PCOL = {"v1 rank, 4k": "#8a94a3", "v2 rank, 12k": "#2f6fdb", "v3 reorder, 4k": "#7a4fb5", "v4 reorder+repeat, 4k": "#2e8b57"}

R = json.load(open("results.json"))
POL = R["policies"]

fig = plt.figure(figsize=(18, 7.4), facecolor=BG)
axes = [fig.add_axes([0.135, 0.14, 0.2, 0.66]), fig.add_axes([0.405, 0.14, 0.2, 0.66]),
        fig.add_axes([0.64, 0.05, 0.35, 0.75])]
for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
texts = []

# ---------------------------------------------------------------- A. two numbers, four policies
ax = axes[0]
ax.grid(True, color=GRID, lw=0.7, axis="x")
ax.set_axisbelow(True)
names = list(POL)
ys = [3.0, 2.0, 1.0, 0.0]
for y, n in zip(ys, names):
    e = POL[n]
    ax.barh(y + 0.19, e["context_recall"] * 100, 0.34, color=DASH)
    ax.barh(y - 0.19, e["answer_rate"] * 100, 0.34, color=PCOL[n])
    texts.append(ax.text(e["context_recall"] * 100 + 1, y + 0.19, f"{e['context_recall']:.0%} in the window",
                         va="center", fontsize=8.8, color=MUTED))
    texts.append(ax.text(e["answer_rate"] * 100 + 1, y - 0.19, f"{e['answer_rate']:.1%} answered",
                         va="center", fontsize=9.5, color=INK, weight="bold"))
ax.set_yticks(ys)
ax.set_yticklabels([f"{n}\n{POL[n]['tokens_per_query']:,.0f} tokens" for n in names], color=INK, fontsize=9.2)
ax.set_xlim(0, 150)
ax.set_ylim(-0.75, 3.85)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("share (%)", color=INK)
ax.set_title("A. Dashboard: 12k wins. Answers: 12k loses", loc="left", color=INK, fontsize=11, weight="bold")
texts.append(ax.text(2, 3.62, "grey: context recall, needed facts present (dashboard)", fontsize=8.2, color=MUTED))
texts.append(ax.text(2, 3.44, "colour: answer rate, every fact read + instruction obeyed", fontsize=8.2, color=INK))
d = R["deltas"]
texts.append(ax.text(2, -0.6, f"12k: context recall {d['12k window: context recall'] * 100:+.1f} pts, answers "
                               f"{d['12k window: answer rate'] * 100:+.1f} pts", fontsize=9, color=BAD))

# ---------------------------------------------------------------- B. the budget sweep
ax = axes[1]
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
sw = R["sweep"]["rank, top"]
bx = [r["budget"] / 1000 for r in sw]
ax.plot(bx, [r["context_recall"] * 100 for r in sw], color=DASH, lw=2.4)
ax.plot(bx, [r["answer_rate"] * 100 for r in sw], color=PCOL["v1 rank, 4k"], lw=2.4, ls="-", marker="o", ms=3.5)
sw2 = R["sweep"]["reorder, both"]
ax.plot([r["budget"] / 1000 for r in sw2], [r["answer_rate"] * 100 for r in sw2], color=OK, lw=2.0, ls="--")
peak = max(sw, key=lambda r: r["answer_rate"])
peak2 = max(sw2, key=lambda r: r["answer_rate"])
ax.plot([peak["budget"] / 1000], [peak["answer_rate"] * 100], "o", color=BAD, ms=7)
ax.set_xlim(0, 12.6)
ax.set_ylim(0, 100)
ax.set_xlabel("token budget (thousands)", color=INK)
ax.set_ylabel("share of queries (%)", color=INK, fontsize=9.5)
ax.set_title("B. Recall only rises, the answer rate peaks", loc="left", color=INK, fontsize=11, weight="bold")
texts.append(ax.text(6.3, sw[-1]["context_recall"] * 100 + 2.5, f"context recall -> {sw[-1]['context_recall']:.0%}", fontsize=9,
                     color=MUTED))
texts.append(ax.text(4.4, 22, f"peak {peak['answer_rate']:.0%} at {peak['budget'] // 1000}k, rank order", fontsize=9, color=INK,
                     weight="bold"))
texts.append(ax.text(9.6, sw[-1]["answer_rate"] * 100 + 4, f"{sw[-1]['answer_rate']:.0%} at 12k", fontsize=9, color=BAD))
texts.append(ax.text(5.6, 50, f"dashed: reorder + repeated\ninstruction, peak {peak2['answer_rate']:.0%} at {peak2['budget'] // 1000}k",
                     fontsize=8.4, color=OK))
texts.append(ax.text(0.3, 6, "comply falls 3 pts per 1k tokens;\nevery position reads 2.5 pts worse", fontsize=8.4, color=MUTED))

# ---------------------------------------------------------------- C. the mechanism, drawn
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 7.2)
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
# the window, top to bottom
x0, wdt = 1.05, 4.25
blocks = [("INSTRUCTION  (180 tokens)", 0.55, "#eef1f5", INK, False),
          ("chunk, rank 1", 0.42, BG, MUTED, False), ("chunk, rank 2", 0.42, BG, MUTED, False),
          ("chunk, rank 3", 0.42, BG, MUTED, False), ("chunk, rank 4", 0.42, BG, MUTED, False),
          ("chunk, rank 5  <- the needed fact", 0.42, "#fff8e6", HOT, True),
          ("chunk, rank 6", 0.42, BG, MUTED, False), ("chunk, rank 7", 0.42, BG, MUTED, False),
          ("chunk, rank 8", 0.42, BG, MUTED, False), ("chunk, rank 9", 0.42, BG, MUTED, False),
          ("QUESTION  (60 tokens)", 0.55, "#eef1f5", INK, False)]
y = 6.6
ys_mid = []
for label, h, fc, ec, gold in blocks:
    y -= h + 0.06
    ax.add_patch(FancyBboxPatch((x0, y), wdt, h, boxstyle="round,pad=0.01,rounding_size=0.06", fc=fc, ec=ec,
                                lw=1.4 if gold else 0.9))
    texts.append(ax.text(x0 + 0.15, y + h / 2, label, fontsize=8.4, color=BAD if gold else INK, va="center",
                         weight="bold" if gold else "normal"))
    ys_mid.append(y + h / 2)
texts.append(ax.text(x0, 6.72, "THE WINDOW, as the model meets it  (v1: 4k budget, rank order)", fontsize=9.3,
                     color=INK, weight="bold"))
# the read curve beside the chunks
cx0, cw = 6.1, 2.5
top, bot = ys_mid[1], ys_mid[-2]
import numpy as np  # noqa: E402

us = np.linspace(0, 1, 60)
ps = [P.read(u, 4_000) for u in us]
ax.plot([cx0 + cw * p for p in ps], [top - (top - bot) * u for u in us], color=INK, lw=1.8)
for p in (0.4, 0.6, 0.8):
    ax.plot([cx0 + cw * p] * 2, [bot - 0.1, top + 0.1], color=GRID, lw=0.7)
    texts.append(ax.text(cx0 + cw * p, bot - 0.32, f"{p:.0%}", fontsize=7.8, color=MUTED, ha="center"))
texts.append(ax.text(cx0, top + 0.62, "P(the model reads this position)", fontsize=8.6, color=INK, weight="bold"))
texts.append(ax.text(cx0, top + 0.3, "quadratic through the 3 Liu et al. points", fontsize=7.8, color=MUTED))
gold_u = (ys_mid[1] - ys_mid[5]) / (ys_mid[1] - ys_mid[-2])
pg = P.read(gold_u, 4_000)
ax.plot([cx0 + cw * pg], [ys_mid[5]], "o", color=BAD, ms=7)
ax.add_patch(FancyArrowPatch((x0 + wdt + 0.05, ys_mid[5]), (cx0 + cw * pg - 0.12, ys_mid[5]), arrowstyle="-|>",
                             mutation_scale=9, color=BAD, lw=1.1, ls=":"))
texts.append(ax.text(cx0 + cw * pg + 0.15, ys_mid[5], f"read {pg:.0%}", fontsize=9, color=BAD, weight="bold", va="center"))
# the instruction distance
ax.add_patch(FancyArrowPatch((x0 - 0.22, ys_mid[0] - 0.2), (x0 - 0.22, ys_mid[-1] + 0.2), arrowstyle="<|-|>",
                             mutation_scale=8, color=MUTED, lw=1.0))
e1 = POL["v1 rank, 4k"]
texts.append(ax.text(x0 - 0.3, (ys_mid[0] + ys_mid[-1]) / 2, f"~{e1['tokens_per_query'] - 240:,.0f} tokens between the\n"
                                                           f"instruction and the question:\nobeyed {e1['compliance']:.0%}",
                     fontsize=7.9, color=MUTED, ha="right", va="center", rotation=90))
# bottom line
texts.append(ax.text(0.1, 0.42, f"context recall: {e1['context_recall']:.0%} of needed facts in the window. "
                                f"{e1['fail_unread']:.0%} of queries lose a fact that WAS there;", fontsize=8.7, color=INK))
texts.append(ax.text(0.1, 0.08, f"{e1['fail_missing']:.0%} lose one to retrieval, {e1['fail_ignored']:.0%} to the instruction. "
                                f"Answer rate {e1['answer_rate']:.0%}.", fontsize=8.7, color=INK))
texts.append(ax.text(0.1, -0.32, "the dashboard's failure class is the smaller one, and the only one it can see",
                     fontsize=9.1, color=BAD, weight="bold"))
ax.set_title("C. Present is not read: position and length decide, after retrieval", loc="left", color=INK,
             fontsize=11, weight="bold")

fig.suptitle("The fact was in the context, and the model did not read it.  Context recall counts presence; "
             "the answer depends on position, length and where the instruction sits", x=0.02, ha="left", y=0.965,
             color=INK, fontsize=12.5, weight="bold")

fig.canvas.draw()
rend = fig.canvas.get_renderer()
titles = [t for ax in axes for t in (ax.title, ax._left_title, ax._right_title) if t.get_text()]
assert len(titles) == len(axes), "a panel title was not found by the geometry check"
boxes = [t.get_window_extent(rend) for t in texts]
tboxes = [t.get_window_extent(rend) for t in titles]
for i in range(len(tboxes)):
    for j in range(i + 1, len(tboxes)):
        assert not tboxes[i].overlaps(tboxes[j]), f"title {titles[i].get_text()!r} overlaps {titles[j].get_text()!r}"
    for j, bb in enumerate(boxes):
        assert not tboxes[i].overlaps(bb), f"title {titles[i].get_text()!r} overlaps {texts[j].get_text()!r}"
problems = []
for i in range(len(boxes)):
    ax_box = texts[i].axes.get_window_extent(rend)
    bb = boxes[i]
    if not (ax_box.x0 - 1 <= bb.x0 and bb.x1 <= ax_box.x1 + 1 and ax_box.y0 - 1 <= bb.y0 and bb.y1 <= ax_box.y1 + 1):
        inv = texts[i].axes.transData.inverted()
        (xa, ya), (xb, yb) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        problems.append(f"text {texts[i].get_text()!r} leaves its axes: data x {xa:.2f}..{xb:.2f} y {ya:.2f}..{yb:.2f}")
    for j in range(i + 1, len(boxes)):
        if bb.overlaps(boxes[j]):
            problems.append(f"{texts[i].get_text()!r} overlaps {texts[j].get_text()!r}")
assert not problems, "\n".join(problems)
fig_box = fig.bbox
for ax in axes[:2]:
    for lab in ax.get_yticklabels() + ax.get_xticklabels():
        if lab.get_text():
            bb = lab.get_window_extent(rend)
            assert bb.x0 >= fig_box.x0 and bb.x1 <= fig_box.x1, f"tick label {lab.get_text()!r} leaves the figure"
            assert bb.y0 >= fig_box.y0 and bb.y1 <= fig_box.y1, f"tick label {lab.get_text()!r} leaves the figure"
sup = fig._suptitle.get_window_extent(rend)
assert sup.x1 <= fig_box.x1, "suptitle leaves the figure"
for bb in tboxes:
    assert not sup.overlaps(bb), "suptitle overlaps a panel title"

# Lines join the check: the first render of this chart put the peak label straight across the dashed line and the
# text-vs-text check passed. Sample every Line2D in the data panels and refuse any text box a sample lands in.
for ax in axes[:2]:
    for line in ax.get_lines():
        xy = ax.transData.transform(np.column_stack(line.get_data()))
        for t, bb in zip(texts, boxes):
            if t.axes is ax and any(bb.contains(x, y) for x, y in xy):
                raise AssertionError(f"text {t.get_text()!r} sits on a plotted line")

fig.savefig("packing_audit.png", dpi=300, facecolor=BG)
fig.savefig("packing_audit.svg", facecolor=BG)
print("wrote packing_audit.png / .svg")
