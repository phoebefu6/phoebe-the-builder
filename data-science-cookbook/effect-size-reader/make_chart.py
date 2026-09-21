"""The audit figure. Four panels, every value read from results.json so the chart cannot drift
from evidence.txt or the README.

Encoding discipline: DISTRIBUTION SHAPE is the only thing colour encodes, and the same fill means
the same shape in every panel. Sample size is always position, never colour. The normal
distribution keeps the one neutral ink tone throughout, because in every panel it is the reference
case the others are being read against.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
SHAPE_COLOR = {
    "normal": "#41506b",        # the reference case - neutral ink
    "uniform": "#2f6fdb",
    "t5": "#1a8f7a",
    "contaminated": "#b8860b",
    "lognormal": "#c8562b",
    "exponential": "#9a3f8f",
}

R = json.load(open("results.json"))
S = R["summary"]
SHAPES = R["shapes"]
DS = R["ds"]
NS = R["ns"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
for ax in axes.ravel():
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)

# ------------------------------------------------- 1. same d, six answers
ax = axes[0, 0]
# Grouped dots on a shared baseline rather than bars: the quantity of interest is the SPREAD
# within each d, and bars all starting at 0.5 would encode the baseline as the message.
x = np.arange(len(DS))
for i, shape in enumerate(SHAPES):
    vals = [R["truths"][f"{shape}|{d}"]["ps"] for d in DS]
    off = (i - (len(SHAPES) - 1) / 2) * 0.115
    ax.plot(x + off, vals, "o", color=SHAPE_COLOR[shape], ms=11,
            label=shape, zorder=3, mec=BG, mew=1.2)
for xi, d in enumerate(DS):
    vals = [R["truths"][f"{s}|{d}"]["ps"] for s in SHAPES]
    ax.plot([xi - 0.36, xi + 0.36], [min(vals)] * 2, color=GRID, lw=1)
    ax.plot([xi - 0.36, xi + 0.36], [max(vals)] * 2, color=GRID, lw=1)
    ax.annotate("", xy=(xi + 0.44, min(vals)), xytext=(xi + 0.44, max(vals)),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.text(xi + 0.50, (min(vals) + max(vals)) / 2, f"{max(vals) - min(vals):.3f}\nspread",
            fontsize=9, color=INK, va="center")
ax.set_xticks(x)
ax.set_xticklabels([f"true Cohen's d = {d:g}" for d in DS], fontsize=10)
ax.set_xlim(-0.55, len(DS) - 0.18)
ax.set_ylabel("P(a treated value beats a control value)")
ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower right")
ax.set_title("Cohen's d is IDENTICAL within each column", color=INK, fontsize=12,
             weight="bold", loc="left")

# ------------------------------------------------- 2. the estimators at n = 10
ax = axes[0, 1]
c10 = {c["shape"]: c for c in R["cells"] if c["n"] == 10}
y = np.arange(len(SHAPES))
h = 0.32
for i, shape in enumerate(SHAPES):
    c = c10[shape]
    ax.barh(i + h / 2, c["d_bias"], height=h, color=SHAPE_COLOR[shape], alpha=0.95)
    ax.barh(i - h / 2, c["g_bias"], height=h, color=SHAPE_COLOR[shape], alpha=0.42)
    ax.plot([c["ps_bias"]], [i], "|", color=INK, ms=13, mew=2, zorder=4)
ax.axvline(0, color=INK, lw=1.2)
ax.set_yticks(y)
ax.set_yticklabels(SHAPES, fontsize=9.5)
ax.set_xlabel("bias in the estimate at n = 10 per group (true d = 0.5)")
ax.set_title("Hedges' correction only works where it was derived", color=INK, fontsize=12,
             weight="bold", loc="left")
ax.text(0.985, 0.03,
        "solid = Cohen's d   ·   pale = Hedges' g (the corrected one)\n"
        "tick = bias in P(X>Y), which needs no correction at all",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=MUTED)
ln = c10["lognormal"]
ax.annotate(f"still {ln['g_bias']:+.3f} out\nafter correcting",
            xy=(ln["g_bias"], SHAPES.index("lognormal") - h / 2),
            xytext=(0.40, 0.30), textcoords="axes fraction", fontsize=9.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))

# ------------------------------------------------- 3. robustness
ax = axes[1, 0]
rob = sorted(R["robustness"], key=lambda r: r["pct_change"])
names = [r["metric"].replace("_", " ") for r in rob]
vals = [r["pct_change"] for r in rob]
# One colour for the failure, neutral for everything else: the panel has exactly one finding in
# it and the palette should not imply six.
cols = [SHAPE_COLOR["lognormal"] if abs(v) > 10 else MUTED for v in vals]
ax.barh(np.arange(len(rob)), vals, color=cols, height=0.62)
ax.axvline(0, color=INK, lw=1.2)
for i, v in enumerate(vals):
    ax.text(v + (0.6 if v >= 0 else -0.6), i, f"{v:+.1f}%", va="center",
            ha="left" if v >= 0 else "right", fontsize=9.5,
            weight="bold" if abs(v) > 10 else "normal", color=INK)
ax.set_yticks(np.arange(len(rob)))
ax.set_yticklabels(names, fontsize=9.5)
ax.set_xlim(min(vals) - 6, max(vals) + 7)
ax.set_xlabel("change in the reported effect when 1 treated value in 100 is an outlier")
ax.set_title("The metric recommended for unequal spread is the fragile one", color=INK,
             fontsize=12, weight="bold", loc="left")

# ------------------------------------------------- 4. significance vs magnitude
ax = axes[1, 1]
small = R["smallest_significant"]
ns = [s["n"] for s in small]
ps = [s["ps"] for s in small]
ax.plot(ns, ps, "o-", color=INK, lw=1.8, ms=9, zorder=3)
ax.axhline(0.5, color=INK, ls="--", lw=1.4)
ax.set_xscale("log")
ax.set_ylim(0.495, 0.60)
for s in small:
    ax.annotate(f"d = {s['d']:.3f}", xy=(s["n"], s["ps"]), xytext=(0, 11),
                textcoords="offset points", ha="center", fontsize=9, color=MUTED)
hit = small[3]
ax.annotate(f"at n = {hit['n']:,} per group the smallest effect\nthe study is not guessing about "
            f"is a {hit['ps']:.2%}\nwin rate - a coin flip with a significant lean",
            xy=(hit["n"], hit["ps"]), xytext=(0.30, 0.78), textcoords="axes fraction",
            fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
# The label belongs ON the rule it names, not floating in the corner.
ax.text(0.015, 0.5008, "a coin flip", transform=ax.get_yaxis_transform(), ha="left",
        va="bottom", fontsize=9, color=MUTED)
ax.set_xlabel("n per group (log scale)")
ax.set_ylabel("P(treated beats control) at the smallest detectable effect")
ax.set_title("What 'significant' still means once n is large", color=INK, fontsize=12,
             weight="bold", loc="left")

fig.suptitle("Cohen's d is not the effect - it is one summary of it, and the summaries disagree",
             color=INK, fontsize=16.5, weight="bold", x=0.007, ha="left", y=0.985)
fig.text(0.007, 0.951,
         "Every population is standardised to mean 0 and SD 1 by its ANALYTIC moments, then "
         "shifted by exactly d - so the population Cohen's d is identical across all six shapes.",
         color=MUTED, fontsize=9.5, ha="left")
fig.text(0.007, 0.930,
         f"{R['reps']:,} replicates a cell; {R['reference_draws']:,} draws per population truth; "
         f"the normal closed form reproduced on {S['n_calibrated']} of "
         f"{S['n_calibration_cells']} calibration cells before any finding was read.",
         color=MUTED, fontsize=9.5, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.918))
fig.savefig("effect_size_reader_audit.png", dpi=160, facecolor=BG)
fig.savefig("effect_size_reader_audit.svg", facecolor=BG)
print("wrote effect_size_reader_audit.png and .svg")
