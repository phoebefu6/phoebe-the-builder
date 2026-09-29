"""The audit figure. Every value read from results.json, so the chart cannot drift from evidence.txt.

Three panels: how a frozen golden set's headline drifts from production month by month, split into the
part reweighting removes for free and the part only new labels remove; how often each allocation design
catches a regression confined to one small intent; and a drawn picture of the mechanism - the traffic
mix at month 0 and month 12 beside the cases the set holds. Geometry self-check runs BEFORE savefig.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID, BG, BAND = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff", "#eef1f5"
RAW, REW, GAP, WARN = "#c8562b", "#2f6fdb", "#b8860b", "#8a3b12"

R = json.load(open("results.json"))
C = R["config"]

fig = plt.figure(figsize=(18, 6.8), facecolor=BG)
axes = [fig.add_axes([0.05, 0.14, 0.26, 0.69]), fig.add_axes([0.37, 0.14, 0.26, 0.69]),
        fig.add_axes([0.75, 0.14, 0.23, 0.69])]
for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
texts = []

# ---------------------------------------------------------------- A. staleness, decomposed
ax = axes[0]
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
S = R["staleness"]
mo = [s["month"] for s in S]
raw = [100 * s["raw_bias"] for s in S]
rew = [100 * s["rew_bias"] for s in S]
ax.fill_between(mo, rew, raw, color=REW, alpha=0.14, lw=0)
ax.fill_between(mo, 0, rew, color=GAP, alpha=0.18, lw=0)
ax.plot(mo, raw, color=RAW, lw=2.4, marker="o", ms=4, label="raw pass rate on the set")
ax.plot(mo, rew, color=REW, lw=2.4, marker="o", ms=4, label="reweighted by today's traffic mix")
ax.axhline(0, color=INK, lw=0.9)
ax.set_xlim(0, 12.3)
ax.set_ylim(-0.3, 6.4)
ax.set_xlabel("months since the golden set was frozen", color=INK)
ax.set_ylabel("golden-set pass rate minus production (pts)", color=INK)
s12 = S[-1]
ax.set_title(f"A. Month 12: set says {100 * (s12['truth'] + s12['raw_bias']):.1f}%, production {100 * s12['truth']:.1f}%",
             loc="left", color=INK, fontsize=12.5, weight="bold")
texts.append(ax.text(0.3, 3.9, f"blue band - mix drift, {100 * (s12['raw_bias'] - s12['rew_bias']):.1f} pts:\nreweighting removes it free",
                     color=REW, fontsize=9, ha="left", va="bottom"))
texts.append(ax.text(11.8, 1.1, f"coverage gap {100 * s12['rew_bias']:.1f} pts:\n{s12['uncovered_share']:.0%} of traffic has\n"
                     "no cases. Only new\nlabels remove it", color=WARN, fontsize=9, ha="right", va="bottom"))
ax.legend(loc="upper left", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9)

# ---------------------------------------------------------------- B. regression detection
ax = axes[1]
ax.grid(True, color=GRID, lw=0.7, axis="y")
ax.set_axisbelow(True)
G = R["regression"]
x = np.arange(len(G))
ax.bar(x - 0.19, [g["aggregate"] for g in G], 0.36, color=REW, label="one McNemar on the whole set")
ax.bar(x + 0.19, [g["per_intent"] for g in G], 0.36, color=GAP, label="per-intent McNemar, Bonferroni")
for i, g in enumerate(G):
    texts.append(ax.text(i - 0.19, g["aggregate"] + 0.015, f"{g['aggregate']:.2f}", ha="center", fontsize=9, color=INK))
    texts.append(ax.text(i + 0.19, g["per_intent"] + 0.015, f"{g['per_intent']:.2f}", ha="center", fontsize=9, color=INK))
ax.set_xticks(x)
ax.set_xticklabels([f"{g['design']}\n{g['cases_in_regressed']} cases there" for g in G], color=INK)
ax.set_ylim(0, 1)
ax.axhline(0.8, color=MUTED, lw=0.8, ls=":")
ax.set_ylabel(f"P(regression flagged), '{C['reg_intent']}' {C['q'][8]:.2f} -> {C['q'][8] * (1 - C['reg_flip']):.2f}", color=INK)
pr = G[0]
ax.set_title(f"B. The set built for the average misses it {1 - pr['aggregate']:.0%} of the time",
             loc="left", color=INK, fontsize=12.5, weight="bold")
ax.legend(loc="upper left", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9)

# ---------------------------------------------------------------- C. the mechanism, drawn
ax = axes[2]
names = C["intents"]
w0, w12 = np.array(C["w0"]), np.array(C["w12"])
alloc = np.array(next(h for h in R["headline"] if h["design"] == "proportional")["allocation"])
y = np.arange(len(names))[::-1]
ax.barh(y + 0.2, 100 * w0, 0.38, color=MUTED, label="traffic, month 0")
ax.barh(y - 0.2, 100 * w12, 0.38, color=INK, label="traffic, month 12")
ax.set_yticks(y)
ax.set_yticklabels(names, color=INK, fontsize=9.5)
ax.set_xlim(0, 58)
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xlabel("share of traffic (%)   |   cases in the set", color=INK)
for yi, a, s in zip(y, alloc, w12):
    col = WARN if a == 0 else REW
    texts.append(ax.text(44, yi - 0.25, f"{a} cases" if a else "0 cases", color=col, fontsize=9,
                         weight="bold" if a == 0 else "normal"))
ax.axvline(42, color=GRID, lw=1)
ax.add_patch(plt.Rectangle((0, -0.7), 57.6, 2.1, fill=False, ec=WARN, lw=1.4, ls="--"))
texts.append(ax.text(12, 0.35, "launched after the\nset was frozen", color=WARN, fontsize=9, va="center"))
ax.set_title(f"C. {C['m']} cases, built once, month 0", loc="left", color=INK, fontsize=12.5, weight="bold")
ax.legend(loc="center right", bbox_to_anchor=(0.70, 0.45), frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9)

fig.suptitle("A golden set is a sample of last year's traffic.  Synthetic 12-intent traffic, m = 240, "
             "exact binomial moments; regression power by Monte Carlo", x=0.05, ha="left", y=0.965, color=INK,
             fontsize=14, weight="bold")

fig.canvas.draw()
rend = fig.canvas.get_renderer()
boxes = [t.get_window_extent(rend) for t in texts]
legends = [ax.get_legend().get_window_extent(rend) for ax in axes]
for i in range(len(boxes)):
    ax_box = texts[i].axes.get_window_extent(rend)
    bb = boxes[i]
    assert ax_box.x0 - 1 <= bb.x0 and bb.x1 <= ax_box.x1 + 1 and ax_box.y0 - 1 <= bb.y0 and bb.y1 <= ax_box.y1 + 1, \
        f"text {texts[i].get_text()!r} leaves its axes"
    for j in range(i + 1, len(boxes)):
        assert not bb.overlaps(boxes[j]), f"{texts[i].get_text()!r} overlaps {texts[j].get_text()!r}"
    for lg in legends:
        assert not bb.overlaps(lg), f"{texts[i].get_text()!r} overlaps a legend"

fig.savefig("golden_audit.png", dpi=300, facecolor=BG)
fig.savefig("golden_audit.svg", facecolor=BG)
print("wrote golden_audit.png / .svg")
