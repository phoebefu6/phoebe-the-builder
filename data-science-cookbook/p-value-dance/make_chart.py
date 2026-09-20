"""The audit figure. Four panels, every value read from results.json so the chart cannot drift
from evidence.txt or the README.

Encoding discipline: one fill per meaning, held across all four panels. The true effect size is
the ONLY thing colour encodes - d=0 grey, d=0.2, d=0.5, d=0.8 on one warm-to-cool ramp. Power is
encoded by position, never by colour, because it is continuous and it is the x-axis of two panels.
The 0.05 line is the same dashed ink rule everywhere it appears.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
# One fill per true effect size, held across every panel.
C = {0.0: "#9aa4b0", 0.2: "#c8562b", 0.5: "#b8860b", 0.8: "#2f6fdb"}

R = json.load(open("results.json"))
CELLS = R["cells"]
ALPHA = R["alpha"]
S = R["summary"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
for ax in axes.ravel():
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)

# ---------------------------------------------- 1. the dance: where 90% of replicates land
ax = axes[0, 0]
# Range bars, not a histogram: the question is not the shape of the p distribution, it is
# whether the central 90% STRADDLES 0.05 - which a range against a reference line answers and a
# histogram does not. n=500 designs are left out; their medians sit near 1e-34 and would flatten
# everything else against the right edge.
rows = [c for c in CELLS if c["n"] <= 100 and (c["d_true"] > 0 or c["n"] == 50)]
rows = sorted(rows, key=lambda c: (c["d_true"], c["n"]))
y = np.arange(len(rows))
for i, c in enumerate(rows):
    lo, hi = c["p_pcts"]["p5"], c["p_pcts"]["p95"]
    col = C[c["d_true"]]
    ax.plot([lo, hi], [i, i], color=col, lw=6, solid_capstyle="round", alpha=0.85)
    ax.plot([c["p_pcts"]["p50"]], [i], "o", color=BG, ms=7, zorder=3)
    ax.plot([c["p_pcts"]["p50"]], [i], "o", color=col, ms=5, zorder=4)
    if c["d_true"] > 0 and c["crosses"]:
        ax.text(9.0, i, "straddles", color=INK, fontsize=8, va="center", ha="center")
ax.axvline(ALPHA, color=INK, ls="--", lw=1.4)
ax.text(ALPHA * 1.5, len(rows) - 0.35, "p = 0.05", color=INK, fontsize=9, va="top")
ax.set_xscale("log")
ax.set_xlim(1e-13, 40)
ax.set_yticks(y)
ax.set_yticklabels([f"d={c['d_true']:g}, n={c['n']}" for c in rows], fontsize=8.5)
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_xlabel("p-value (log scale) - bar spans the 5th to 95th percentile of 40,000 replays")
ax.set_title("Replay the SAME study and the p-value moves over orders of magnitude",
             color=INK, fontsize=12, weight="bold", loc="left")
ax.text(0.0, -0.165, "\"straddles\" = the 0.05 line falls inside that central 90%, so an "
        "identical rerun routinely returns the opposite verdict",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=MUTED)

# ---------------------------------------------- 2. the reversal: spread rises with power
ax = axes[0, 1]
for d in (0.0, 0.2, 0.5, 0.8):
    pts = [(c["power"], c["log10_spread"]) for c in CELLS if c["d_true"] == d]
    pts.sort()
    ax.plot([p for p, _ in pts], [s for _, s in pts], "o-", color=C[d], lw=1.6, ms=7,
            label=f"true d = {d:g}" if d else "null (d = 0)")
ax.axvline(0.95, color=INK, ls=":", lw=1.2)
ax.text(0.945, ax.get_ylim()[1] * 0.98, "power 0.95", rotation=90, ha="right", va="top",
        fontsize=8.5, color=MUTED)
ax.set_xlabel("measured power")
ax.set_ylabel("width of the central 90% of p, in orders of magnitude")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("The dance is WIDEST where it changes nothing", color=INK, fontsize=12,
             weight="bold", loc="left")
ax.text(0.99, 0.02,
        f"widest: {S['widest']['label']} at {S['widest']['orders']:.1f} orders "
        f"(power {S['widest']['power']:.2f})\n"
        f"narrowest: the null, at {S['narrowest']['orders']:.1f} orders",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=MUTED)

# ---------------------------------------------- 3. the winner's curse
ax = axes[1, 0]
alt = sorted([c for c in CELLS if c["d_true"] > 0], key=lambda c: c["power"])
ax.plot([c["power"] for c in alt], [c["type_m"] for c in alt], "-", color=MUTED, lw=1.2, zorder=1)
for c in alt:
    ax.plot([c["power"]], [c["type_m"]], "o", color=C[c["d_true"]], ms=8, zorder=3)
ax.axhline(1.0, color=INK, ls="--", lw=1.2)
ax.text(0.30, 1.04, "no exaggeration", transform=ax.get_yaxis_transform(), ha="left",
        va="bottom", fontsize=8.5, color=MUTED)
wm = S["worst_type_m"]
ws = S["worst_type_s"]
ax.annotate(f"{wm['label']}\npublishes effects {wm['value']:.1f}x the truth\n"
            f"and {ws['value']:.0%} of them point the WRONG WAY",
            xy=(alt[0]["power"], alt[0]["type_m"]), xytext=(0.28, 0.72),
            textcoords="axes fraction", fontsize=9.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
ax2 = ax.twinx()
ax2.bar([c["power"] for c in alt], [c["type_s"] for c in alt], width=0.022, color=INK,
        alpha=0.30, zorder=0)
ax2.set_ylabel("share of significant results with the WRONG SIGN (bars)", color=MUTED, fontsize=9)
ax2.set_ylim(0, 0.30)
ax2.grid(False)
for s in ax2.spines.values():
    s.set_color(GRID)
ax.set_xlabel("measured power")
ax.set_ylabel("exaggeration ratio\n(mean |observed d| when significant) / true d")
ax.set_title("This part is NOT the power calculation restated", color=INK, fontsize=12,
             weight="bold", loc="left")

# ---------------------------------------------- 4. what a significant result is worth
ax = axes[1, 1]
ppv = R["ppv"]
lab = [f"prior {r['prior']:.0%}\nd={r['d_true']:g}, n={r['n']}\npower {r['power']:.2f}"
       for r in ppv]
x = np.arange(len(ppv))
vals = [r["false_share"] for r in ppv]
err = np.array([[r["false_share"] - r["false_share_ci"][0] for r in ppv],
                [r["false_share_ci"][1] - r["false_share"] for r in ppv]])
bars = ax.bar(x, vals, color=[C[r["d_true"]] for r in ppv], width=0.62, alpha=0.9)
ax.errorbar(x, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=4)
ax.plot(x, [r["formula"] for r in ppv], "_", color=BG, ms=26, mew=5, zorder=4)
ax.plot(x, [r["formula"] for r in ppv], "_", color=INK, ms=22, mew=2, zorder=5,
        label="closed form (a check on the simulation)")
ax.axhline(ALPHA, color=INK, ls="--", lw=1.4)
# No room anywhere inside this panel - bars 1, 2 and 5 cross the rule, the legend owns the top
# left and the tallest bar's label owns the top middle. So it becomes a footnote, which is also
# how panel 1 handles its own caption.
ax.text(0.0, -0.22, "dashed rule: alpha = 0.05 - the number people believe this bar is showing",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=MUTED)
for xi, v in zip(x, vals):
    ax.text(xi, v + 0.035, f"{v:.0%}", ha="center", fontsize=10.5, weight="bold", color=INK)
ax.set_xticks(x)
ax.set_xticklabels(lab, fontsize=8.5)
ax.set_ylim(0, 1.0)
ax.set_ylabel("share of SIGNIFICANT findings that were null all along")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("alpha is not the error rate of the claims you publish", color=INK, fontsize=12,
             weight="bold", loc="left")

fig.suptitle("A p-value is a random variable - and its spread is the result",
             color=INK, fontsize=16.5, weight="bold", x=0.007, ha="left", y=0.985)
fig.text(0.007, 0.951,
         "Student's two-sample t on equal-n, equal-variance, normal data - the one configuration "
         "Day 174 verified EXACT, so nothing below is a violated assumption.",
         color=MUTED, fontsize=9.5, ha="left")
fig.text(0.007, 0.930,
         f"{R['reps']:,} replays per design; measured power matched the noncentral-t formula on "
         f"{S['n_calibrated']} of {S['n_cells']} designs before any finding was read.",
         color=MUTED, fontsize=9.5, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.918))
fig.savefig("p_value_dance_audit.png", dpi=160, facecolor=BG)
fig.savefig("p_value_dance_audit.svg", facecolor=BG)
print("wrote p_value_dance_audit.png and .svg")
