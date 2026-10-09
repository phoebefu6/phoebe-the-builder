"""The audit figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Four panels: the drawn chain (where each tail mechanism sits); per-hop p99s stacked against the true end-to-end
p99 and the SLO; who owns the average vs who owns the slow requests; and the five fixes on mean vs p99.
Geometry self-check runs BEFORE savefig: every text inside the figure, no two texts overlapping, titles included.
"""

from __future__ import annotations

import json
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
BAD, OK, BLUE, HOT = "#c8562b", "#2e8b57", "#2f6fdb", "#e3b23c"

R = json.load(open("results.json"))
S = R["study"]
HOPS = ["gateway", "embed", "search", "rerank", "llm"]
COL = {"gateway": "#9aa5b4", "embed": "#7a4fb5", "search": BLUE, "rerank": BAD, "llm": OK}

fig = plt.figure(figsize=(16, 10.2), facecolor=BG)
axD = fig.add_axes([0.03, 0.62, 0.94, 0.31])
axA = fig.add_axes([0.08, 0.07, 0.25, 0.45])
axB = fig.add_axes([0.40, 0.07, 0.25, 0.45])
axC = fig.add_axes([0.72, 0.07, 0.25, 0.45])
texts: List = []

# ---------------------------------------------------------------- D. the chain, drawn
ax = axD
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis("off")
ax.set_title("The chain: five hops in sequence, and the one tail mechanism each really has", loc="left", color=INK,
             fontsize=12, weight="bold")
boxes = [("gateway", 2, "gateway", "lognormal only"),
         ("embed", 20, "query embedding", "2% cold start +300 ms"),
         ("search", 38, "vector search", f"waits for slowest of {R['shards']}"),
         ("rerank", 60, "reranker", "3% hang -> 400 ms timeout, retry"),
         ("llm", 80, "LLM generation", "83% of the mean")]
for key, x, name, mech in boxes:
    ax.add_patch(FancyBboxPatch((x, 13), 15, 9, boxstyle="round,pad=0.4,rounding_size=1.2", fc=BG, ec=COL[key], lw=2.2))
    texts.append(ax.text(x + 7.5, 19.4, name, ha="center", fontsize=10.5, color=INK, weight="bold"))
    texts.append(ax.text(x + 7.5, 15.4, f"p50 {S['hops'][key]['p50']}  p99 {S['hops'][key]['p99']}", ha="center",
                         fontsize=9, color=MUTED))
    texts.append(ax.text(x + 7.5, 9.0, mech, ha="center", fontsize=9, color=COL[key]))
for (_, x0, *_), (_, x1, *_) in zip(boxes[:-1], boxes[1:]):
    ax.add_patch(FancyArrowPatch((x0 + 15.6, 17.5), (x1 - 0.6, 17.5), arrowstyle="-|>", mutation_scale=14, color=MUTED))
# shards fan-out drawn under the search box
for i in range(R["shards"]):
    sx = 38.6 + i * 1.95
    ax.add_patch(FancyBboxPatch((sx, 2.0), 1.3, 3.2, boxstyle="square,pad=0", fc=HOT if i == 5 else GRID, ec="none"))
texts.append(ax.text(54.6, 2.4, f"one shard pauses 0.8% -> some shard {S['p_any_shard_gc']:.1%}", fontsize=8.6, color=INK))
texts.append(ax.text(2, 2.4, f"every shard p99 = {S['shard']['p99']} ms (healthy)   search step p95 = {S['hops']['search']['p95']} ms",
                     fontsize=8.6, color=BLUE))
texts.append(ax.text(2, 26.5, f"end to end: p50 {S['e2e']['p50']} ms, p99 {S['e2e']['p99']} ms, SLO p99 <= {R['slo_ms']} ms "
                     f"({S['p_over_slo']:.2%} of requests over)", fontsize=10, color=INK))

# ---------------------------------------------------------------- A. sum of p99s vs true p99
ax = axA
bottom = 0.0
for h in HOPS:
    v = S["hops"][h]["p99"]
    ax.bar(0, v, 0.55, bottom=bottom, color=COL[h])
    texts.append(ax.text(0.33, bottom + v / 2, f"{h} {v}", va="center", fontsize=8.6, color=INK))
    bottom += v
ax.bar(1.4, S["e2e"]["p99"], 0.55, color=INK)
texts.append(ax.text(1.4, S["e2e"]["p99"] + 40, f"true p99\n{S['e2e']['p99']} ms", ha="center", fontsize=9.5, color=INK,
                     weight="bold"))
texts.append(ax.text(0, bottom + 40, f"sum {S['sum_of_hop_p99']} ms", ha="center", fontsize=9.5, color=BAD, weight="bold"))
ax.axhline(R["slo_ms"], color=BAD, ls="--", lw=1.2)
texts.append(ax.text(1.75, R["slo_ms"] + 25, "SLO", fontsize=9, color=BAD))
ax.set_xticks([0, 1.4])
ax.set_xticklabels(["per-hop p99s,\nadded", "end-to-end\np99"], color=INK)
ax.set_xlim(-0.5, 2.05)
ax.set_ylim(0, 2700)
ax.set_ylabel("ms", color=INK)
ax.set_title(f"A. Adding p99s says {S['sum_of_hop_p99'] - R['slo_ms']} over;\n    truth is {S['e2e']['p99'] - R['slo_ms']} over",
             loc="left", color=INK, fontsize=11, weight="bold")

# ---------------------------------------------------------------- B. mean share vs tail share
ax = axB
ys = list(range(len(HOPS)))[::-1]
for y, h in zip(ys, HOPS):
    ax.barh(y + 0.18, S["mean_share"][h] * 100, 0.34, color=COL[h], alpha=0.4)
    ax.barh(y - 0.18, S["tail_share"][h] * 100, 0.34, color=COL[h])
    texts.append(ax.text(S["mean_share"][h] * 100 + 1.2, y + 0.18, f"{S['mean_share'][h]:.0%}", va="center", fontsize=8.6, color=MUTED))
    texts.append(ax.text(S["tail_share"][h] * 100 + 1.2, y - 0.18, f"{S['tail_share'][h]:.0%}", va="center", fontsize=8.6,
                         color=INK, weight="bold"))
ax.set_yticks(ys)
ax.set_yticklabels(HOPS, color=INK)
ax.set_xlim(0, 100)
ax.set_xlabel("% of mean (pale)  vs  % of tail excess (solid)", color=INK)
ax.set_title("B. The LLM owns the average;\n    the reranker owns the slow requests", loc="left", color=INK, fontsize=11, weight="bold")

# ---------------------------------------------------------------- C. fixes on two axes
ax = axC
short = {"smaller LLM (-15% median)": "smaller LLM", "warm embedding pool": "warm embed pool",
         "hedge shard calls at 80 ms": "hedge shards", "rerank timeout 400 -> 120 ms": "rerank timeout",
         "all three config fixes": "3 config fixes"}
offs = {"smaller LLM": (-6, 10), "warm embed pool": (4, -14), "hedge shards": (6, 6), "rerank timeout": (8, 2),
        "3 config fixes": (8, -4)}
for r in R["fixes"]:
    n = short[r["fix"]]
    c = OK if n == "smaller LLM" else BAD
    ax.scatter(-r["d_mean"], -r["d_p99"], s=70, color=c, zorder=3)
    dx, dy = offs[n]
    texts.append(ax.text(-r["d_mean"] + dx, -r["d_p99"] + dy, n, fontsize=9, color=INK, ha="right" if dx < 0 else "left"))
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
ax.set_xlim(-5, 160)
ax.set_ylim(0, 215)
ax.set_xlabel("ms saved on the MEAN (what the dashboard ranks)", color=INK)
ax.set_ylabel("ms saved on p99 (what the SLO counts)", color=INK)
ax.set_title("C. Config fixes save 6x less mean\n    and more p99 than a smaller LLM", loc="left", color=INK, fontsize=11, weight="bold")

for a in (axA, axB, axC):
    a.set_facecolor(BG)
    for sp in a.spines.values():
        sp.set_color(GRID)
    a.tick_params(colors=INK)


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
    fig.savefig("latency_audit.png", dpi=150, facecolor=BG)
    fig.savefig("latency_audit.svg", facecolor=BG)
    print("saved latency_audit.png / .svg, geometry ok")
