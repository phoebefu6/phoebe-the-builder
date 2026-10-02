"""The audit figure. Every value read from results.json, so the chart cannot drift from evidence.txt.

Three panels: which gate fires under each migration (exact); the net-zero sweep, where McNemar gets
quieter as churn grows; and a drawn picture of the mechanism - the 400 cases by intent, with the cases
the concentrated migration broke and fixed marked. The geometry self-check runs BEFORE savefig (Day 180).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import migrate as M
import numpy as np

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
GATE_COL = {"score delta": "#8a94a3", "mcnemar": "#2f6fdb", "tost": "#8a6d3b", "per-intent": "#e3b23c",
            "churn vs rerun": "#2e8b57"}
PASS, FAIL, FLAKY, BROKE, FIXED = "#dfe7f1", "#56657a", "#f0d58c", "#c8562b", "#2e8b57"

R = json.load(open("results.json"))
SC = {r["scenario"]: r for r in R["scenarios"]}

fig = plt.figure(figsize=(18, 6.8), facecolor=BG)
axes = [fig.add_axes([0.05, 0.16, 0.30, 0.66]), fig.add_axes([0.40, 0.16, 0.26, 0.66]),
        fig.add_axes([0.71, 0.16, 0.27, 0.66])]
for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
texts = []

# ---------------------------------------------------------------- A. which gate fires
ax = axes[0]
ax.grid(True, color=GRID, lw=0.7, axis="y")
ax.set_axisbelow(True)
w = 0.16
for j, g in enumerate(M.GATES):
    vals = [SC[s]["gates"][g] for s in M.SCENARIOS]
    xs = np.arange(len(M.SCENARIOS)) + (j - 2) * w
    ax.bar(xs, vals, w * 0.92, color=GATE_COL[g], label=g + (" (P certified)" if g == "tost" else ""))
    for x, v in zip(xs, vals):
        texts.append(ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=INK, rotation=90))
ax.set_xticks(np.arange(len(M.SCENARIOS)))
ax.set_xticklabels([s.replace(", ", ",\n") + f"\n{SC[s]['score_a']:.1%} -> {SC[s]['score_b']:.1%}" for s in M.SCENARIOS],
                   color=INK, fontsize=9)
ax.set_ylim(0, 1.45)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylabel("P(gate fires), exact", color=INK)
ax.set_title("A. Two migrations score 88.3% -> 88.3%", loc="left", color=INK, fontsize=12.5, weight="bold")
ax.legend(loc="upper center", ncol=3, frameon=True, facecolor=BG, edgecolor=GRID, fontsize=8.5)

# ---------------------------------------------------------------- B. the sweep
ax = axes[1]
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
sw = R["sweep"]
xs = [r["churn_cases"] for r in sw]
for g in ("mcnemar", "tost", "churn vs rerun", "per-intent"):
    ax.plot(xs, [r[g] for r in sw], color=GATE_COL[g], lw=2.2, marker="o", ms=3.5,
            label=g + (" (P certified)" if g == "tost" else ""))
ax.axhline(M.ALPHA, color=MUTED, lw=1, ls=":")
ax.set_xlabel("cases that changed behaviour (half broke, half fixed; score unchanged)", color=INK)
ax.set_ylabel("P(gate fires), exact", color=INK)
ax.set_ylim(-0.03, 1.12)
ax.set_title("B. More churn makes McNemar quieter", loc="left", color=INK, fontsize=12.5, weight="bold")
texts.append(ax.text(21, 0.30, f"McNemar: {sw[0]['mcnemar']:.3f} at 0 changed\n-> {sw[-1]['mcnemar']:.4f} at "
                     f"{sw[-1]['churn_cases']} changed", color=GATE_COL["mcnemar"], fontsize=9.5))
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.68), frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9)

# ---------------------------------------------------------------- C. the mechanism, drawn
ax = axes[2]
pa, intent, kind = M.eval_set()
pb, changed = M.scenario("net zero, concentrated", pa, intent, kind)
broke, fixed = changed & (pb < pa), changed & (pb > pa)
row = 0
nc = 20
for j, (name, size) in enumerate(M.INTENTS):
    idx = np.flatnonzero(intent == j)
    for t, i in enumerate(idx):
        r, c = divmod(t, nc)
        col = BROKE if broke[i] else FIXED if fixed[i] else (PASS, FAIL, FLAKY)[kind[i]]
        ax.add_patch(plt.Rectangle((c, -(row + r)), 0.86, 0.86, color=col, lw=0))
    texts.append(ax.text(-0.6, -(row) + 0.43, name, ha="right", va="center", fontsize=8.5,
                         color=BROKE if name == M.CONCENTRATE_IN else INK,
                         weight="bold" if name == M.CONCENTRATE_IN else "normal"))
    row += -(-size // nc) + 0.5
ax.set_xlim(-5.2, nc + 0.3)
ax.set_ylim(-row - 6.2, 1.2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")
for sp in ax.spines.values():
    sp.set_visible(False)
rs = SC["net zero, concentrated"]["intent_scores"][M.CONCENTRATE_IN]
texts.append(ax.text(-5, -row - 0.4, f"red: broke {int(broke.sum())}, all in {M.CONCENTRATE_IN}\n"
                     f"   ({M.CONCENTRATE_IN} {rs[0]:.0%} -> {rs[1]:.0%})\ngreen: fixed {int(fixed.sum())}, spread\n"
                     "pale pass / dark fail / gold flaky", color=INK, fontsize=8.5, va="top"))
ax.set_title("C. The same 88.3%, with 40 cases swapped", loc="left", color=INK, fontsize=12.5, weight="bold")

fig.suptitle("An unchanged score is a net.  Synthetic 400-case eval set, one run per model, exact "
             "probabilities, 5% false-alarm target", x=0.05, ha="left", y=0.965, color=INK, fontsize=14, weight="bold")

fig.canvas.draw()
rend = fig.canvas.get_renderer()
boxes = [t.get_window_extent(rend) for t in texts]
legends = [ax.get_legend().get_window_extent(rend) for ax in axes[:2]]
for i in range(len(boxes)):
    ax_box = texts[i].axes.get_window_extent(rend)
    bb = boxes[i]
    assert ax_box.x0 - 1 <= bb.x0 and bb.x1 <= ax_box.x1 + 1 and ax_box.y0 - 1 <= bb.y0 and bb.y1 <= ax_box.y1 + 1, \
        f"text {texts[i].get_text()!r} leaves its axes"
    for j in range(i + 1, len(boxes)):
        assert not bb.overlaps(boxes[j]), f"{texts[i].get_text()!r} overlaps {texts[j].get_text()!r}"
    for lg in legends:
        assert not bb.overlaps(lg), f"{texts[i].get_text()!r} overlaps a legend"

fig.savefig("migration_audit.png", dpi=300, facecolor=BG)
fig.savefig("migration_audit.svg", facecolor=BG)
print("wrote migration_audit.png / .svg")
