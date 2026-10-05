"""The audit figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Three panels: the three policies on the dashboard's number and on the user's number (the ranking flips);
answer staleness through one monthly cycle against the phase average the metric reports; and a drawn picture
of the mechanism - the index snapshot, the live source, the hot document that moved, the query that lands on
it with the same similarity either way. Geometry self-check runs BEFORE savefig (Day 180), tick labels
included (Day 189).
"""

from __future__ import annotations

import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import staleness as T
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
BAD, OK, DASH, HOT = "#c8562b", "#2e8b57", "#8a94a3", "#e3b23c"
PCOL = {"v1 monthly full": "#8a94a3", "v2a weekly full": "#2f6fdb", "v2b tiered 1/7/90": "#7a4fb5"}

R = json.load(open("results.json"))
P = R["policies"]

fig = plt.figure(figsize=(18, 7.4), facecolor=BG)
axes = [fig.add_axes([0.125, 0.14, 0.2, 0.66]), fig.add_axes([0.385, 0.14, 0.205, 0.66]),
        fig.add_axes([0.62, 0.05, 0.37, 0.75])]
for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
texts = []

# ---------------------------------------------------------------- A. two numbers, three policies
ax = axes[0]
ax.grid(True, color=GRID, lw=0.7, axis="x")
ax.set_axisbelow(True)
names = list(P)
ys = [2.0, 1.0, 0.0]
for y, n in zip(ys, names):
    e = P[n]
    ax.barh(y + 0.19, (1 - e["hash_freshness"]) * 100, 0.34, color=DASH)
    ax.barh(y - 0.19, e["answer_staleness"] * 100, 0.34, color=PCOL[n])
    texts.append(ax.text((1 - e["hash_freshness"]) * 100 + 0.5, y + 0.19, f"{1 - e['hash_freshness']:.1%} of docs",
                         va="center", fontsize=9, color=MUTED))
    texts.append(ax.text(e["answer_staleness"] * 100 + 0.5, y - 0.19, f"{e['answer_staleness']:.1%} of answers",
                         va="center", fontsize=9.5, color=INK, weight="bold"))
ax.set_yticks(ys)
ax.set_yticklabels([f"{n}\n{P[n]['reindex_per_day']:.0f} reindex/day" for n in names], color=INK, fontsize=9.5)
ax.set_xlim(0, 54)
ax.set_ylim(-0.75, 2.78)
ax.set_xlabel("stale share (%)", color=INK)
ax.set_title("A. Dashboard ranks weekly first, answers rank it last", loc="left", color=INK, fontsize=12,
             weight="bold")
texts.append(ax.text(16, 2.52, "grey: docs not fresh (the dashboard)", fontsize=8.6, color=MUTED))
texts.append(ax.text(16, 2.34, "colour: answers from a changed fact", fontsize=8.6, color=INK))
rt = R["ratios"]
texts.append(ax.text(1, -0.6, f"weekly: {rt['stale_answers_weekly_over_tiered']:.1f}x tiered's stale answers, "
                              f"{rt['cost_weekly_over_tiered']:.1f}x its cost", fontsize=9, color=BAD))

# ---------------------------------------------------------------- B. inside the cycle
ax = axes[1]
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
cyc = R["cycle"]
ax.plot([c["day"] for c in cyc], [c["answer_staleness"] * 100 for c in cyc], color=PCOL["v1 monthly full"], lw=2.4)
avg = P["v1 monthly full"]["answer_staleness"] * 100
ax.axhline(avg, color=INK, lw=1.1, ls=":")
ax.fill_between([c["day"] for c in cyc], [c["answer_staleness"] * 100 for c in cyc], avg,
                where=[c["answer_staleness"] * 100 > avg for c in cyc], color=BAD, alpha=0.12, lw=0)
ax.set_xlim(0, 30)
ax.set_ylim(0, 52)
ax.set_xlabel("days since the monthly full reindex", color=INK)
ax.set_ylabel("answers served from a changed fact (%)", color=INK, fontsize=9.5)
ax.set_title("B. The monthly average is not day 29", loc="left", color=INK, fontsize=12, weight="bold")
texts.append(ax.text(0.8, avg + 1.5, f"phase average {avg:.1f}% - what the metric reports", fontsize=9, color=INK))
texts.append(ax.text(17.5, 48.5, f"day 29: {cyc[29]['answer_staleness']:.1%}", fontsize=9.5, color=BAD,
                     weight="bold"))
texts.append(ax.text(13, 4.5, "monthly full reindex,\nv1", fontsize=9, color=PCOL["v1 monthly full"]))

# ---------------------------------------------------------------- C. the mechanism, drawn
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 7.2)
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
f = T.FACTS[2]
ax.add_patch(FancyBboxPatch((0.1, 3.9), 4.2, 2.7, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#eef1f5",
                            ec=MUTED, lw=0.9))
texts.append(ax.text(0.3, 6.3, "THE INDEX  (snapshot, day 0)", fontsize=9.5, color=MUTED, weight="bold"))
ax.add_patch(FancyBboxPatch((5.6, 3.9), 4.3, 2.7, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#fff8e6",
                            ec=HOT, lw=1.2))
texts.append(ax.text(5.8, 6.3, "THE SOURCE  (today, day 29)", fontsize=9.5, color="#8a6d3b", weight="bold"))
rows = [(T.chunk(f, False), T.chunk(f, True), BAD, True), (T.chunk(T.FACTS[3], False), T.chunk(T.FACTS[3], False),
                                                           OK, False)]
for i, (old, new, col, changed) in enumerate(rows):
    y = 5.55 - i * 0.95
    ax.add_patch(FancyBboxPatch((0.3, y - 0.3), 3.8, 0.62, boxstyle="round,pad=0.02,rounding_size=0.08", fc=BG,
                                ec=col, lw=1.2))
    texts.append(ax.text(0.42, y, "\n".join(textwrap.wrap(old, 40)), fontsize=8.2, color=INK, va="center"))
    ax.add_patch(FancyBboxPatch((5.8, y - 0.3), 3.9, 0.62, boxstyle="round,pad=0.02,rounding_size=0.08", fc=BG,
                                ec=col, lw=1.2))
    texts.append(ax.text(5.92, y, "\n".join(textwrap.wrap(new, 42)), fontsize=8.2, color=INK, va="center"))
    texts.append(ax.text(4.95, y, "CHANGED" if changed else "same", fontsize=8.5, ha="center", va="center",
                         color=col, weight="bold"))
texts.append(ax.text(0.3, 4.1, "hash equal: 1 of 2  ->  'index 50% fresh'", fontsize=8.6, color=MUTED))
texts.append(ax.text(5.8, 4.1, "pricing page: edited every 14 days", fontsize=8.6, color="#8a6d3b"))
# the query
ax.add_patch(FancyBboxPatch((0.1, 1.5), 4.2, 1.5, boxstyle="round,pad=0.02,rounding_size=0.12", fc=BG, ec=INK,
                            lw=1.1))
texts.append(ax.text(0.3, 2.68, "THE QUERY", fontsize=9.5, color=INK, weight="bold"))
texts.append(ax.text(0.3, 2.22, f'"{T.query(f)}"', fontsize=8.6, color=INK))
sg = R["similarity"]["rows"][2]
texts.append(ax.text(0.3, 1.74, f"similarity to stale chunk {sg['stale_sim']:.3f}, to fresh {sg['fresh_sim']:.3f}:",
                     fontsize=8.4, color=MUTED))
ax.add_patch(FancyArrowPatch((2.2, 3.05), (2.2, 3.85), arrowstyle="-|>", mutation_scale=10, color=BAD, lw=1.3))
texts.append(ax.text(2.35, 3.42, "retrieves the stale chunk", fontsize=8.6, color=BAD, va="center"))
# the answer
ax.add_patch(FancyBboxPatch((5.6, 1.5), 4.3, 1.5, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#fbeee9",
                            ec=BAD, lw=1.3))
texts.append(ax.text(5.8, 2.68, "THE ANSWER", fontsize=9.5, color=BAD, weight="bold"))
texts.append(ax.text(5.8, 2.22, f'"Refunds within {f[2]}." - cites [1]', fontsize=8.6, color=INK))
texts.append(ax.text(5.8, 1.74, f"the source has said {f[3]} for weeks", fontsize=8.4, color=MUTED))
ax.add_patch(FancyArrowPatch((4.35, 2.25), (5.55, 2.25), arrowstyle="-|>", mutation_scale=10, color=BAD, lw=1.3))
# bottom line
texts.append(ax.text(0.1, 0.75, "the value is not in the query, so no retrieval score moves when the value moves;",
                     fontsize=9, color=INK))
texts.append(ax.text(0.1, 0.35, f"a similarity gate that keeps every fresh chunk passes "
                                f"{R['similarity']['stale_passed_by_gate_keeping_all_fresh']:.0%} of stale ones",
                     fontsize=9, color=INK))
texts.append(ax.text(0.1, -0.2, f"{P['v1 monthly full']['answer_staleness']:.0%} of answers stale, index reads "
                                f"{P['v1 monthly full']['hash_freshness']:.0%} fresh: queries land on the docs that move",
                     fontsize=9.2, color=BAD, weight="bold"))
ax.set_title("C. The index is a snapshot, the source moved, the query cannot tell", loc="left", color=INK,
             fontsize=12, weight="bold")

fig.suptitle("It answered from last quarter's document.  Index freshness is doc-weighted; the answers are "
             "query-weighted, and the queries land on what changes", x=0.02, ha="left", y=0.965, color=INK,
             fontsize=14, weight="bold")

fig.canvas.draw()
rend = fig.canvas.get_renderer()
# Axes titles join the check: Day 191 shipped panel A's title running into panel B's, which the text-only
# check could not see. A title may leave its own axes (it sits above it) but may not overlap any other text.
titles = [ax.title for ax in axes]
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
        (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        problems.append(f"text {texts[i].get_text()!r} leaves its axes: data x {x0:.2f}..{x1:.2f} y {y0:.2f}..{y1:.2f}")
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

fig.savefig("staleness_audit.png", dpi=300, facecolor=BG)
fig.savefig("staleness_audit.svg", facecolor=BG)
print("wrote staleness_audit.png / .svg")
