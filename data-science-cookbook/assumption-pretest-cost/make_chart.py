"""The audit figure. Four panels, all read from results.json so the chart cannot drift from the
numbers in evidence.txt and the README.

Encoding discipline: one fill per meaning, held across every panel. STUDENT is always the same
orange, WELCH the same blue, the GATED two-stage procedure the same purple, and the pretest a
neutral grey. "Correct" is never a colour - it is distance from the dashed nominal line.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
STUDENT, WELCH, GATED, PRETEST = "#c8562b", "#2f6fdb", "#7a4fa3", "#9aa4b0"

R = json.load(open("results.json"))
CELLS = R["cells"]
LADDER = R["ladder"]
SWEEP = R["alpha_sweep"]
ALPHA = R["meta"]["alpha"]
BAND = R["meta"]["band"]
REPS = R["meta"]["reps"]

fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5), facecolor=BG)
for ax in axes.ravel():
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)

# ------------------------------------------------ 1. the grid, three procedures side by side
ax = axes[0, 0]
# A dot plot, not bars: on a log axis every bar would run off the left edge and the three
# procedures would render as identical full-width slabs. What matters here is each value's
# DISTANCE FROM 0.05, which a dot against a reference line shows and a bar cannot.
labels = [c["design"]["label"] for c in CELLS]
y = np.arange(len(CELLS))
stu = [c["student_error"] for c in CELLS]
wel = [c["welch_error"] for c in CELLS]
gat = [c["gated_error"] for c in CELLS]
for i in y:
    ax.plot([min(stu[i], wel[i], gat[i]), max(stu[i], wel[i], gat[i])], [i, i],
            color=GRID, lw=1.4, zorder=1, solid_capstyle="round")
ax.scatter(stu, y, s=46, color=STUDENT, zorder=3, label="Student always")
ax.scatter(wel, y, s=46, color=WELCH, marker="s", zorder=3, label="Welch always")
ax.scatter(gat, y, s=62, facecolor="none", edgecolor=GATED, lw=2.0, marker="D", zorder=4,
           label="pretest, then pick (the move under audit)")
ax.axvline(ALPHA, color=INK, ls="--", lw=1.8, zorder=2)
ax.set_xscale("log")
ax.set_xlim(4e-5, 2.2)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7.5, color=MUTED)
ax.set_ylim(len(CELLS) - 0.4, -1.4)
ax.set_xlabel("false-positive rate under a TRUE null (log scale)", color=MUTED, fontsize=9)
ax.set_title("1. Three procedures, identical samples", color=INK, fontsize=12, weight="bold", loc="left")
ax.legend(fontsize=8, framealpha=1, edgecolor=GRID, loc="upper left", ncol=1)
ax.text(ALPHA * 1.3, len(CELLS) - 0.5, "nominal 0.05", color=INK, fontsize=8, va="bottom")

# ------------------------------------------------ 2. the danger zone
ax = axes[0, 1]
ratios = [c["design"]["sd_ratio"] for c in LADDER]
ax.plot(ratios, [c["student_error"] for c in LADDER], "-o", color=STUDENT, lw=2.2, ms=5, label="Student always")
ax.plot(ratios, [c["welch_error"] for c in LADDER], "-s", color=WELCH, lw=2.2, ms=5, label="Welch always")
ax.plot(ratios, [c["gated_error"] for c in LADDER], "-D", color=GATED, lw=2.6, ms=6, label="pretest, then pick")
ax.plot(ratios, [c["pretest_reject_rate"] for c in LADDER], ":", color=PRETEST, lw=2.0,
        label="share of samples the pretest flags")
ax.axhline(ALPHA, color=INK, ls="--", lw=1.6)
peak = max(LADDER, key=lambda c: c["gated_error"])
ax.annotate(
    f"worst at sd 1:{peak['design']['sd_ratio']:g}\n{peak['gated_error']:.4f} = "
    f"{peak['gated_error'] / ALPHA:.1f}x nominal",
    xy=(peak["design"]["sd_ratio"], peak["gated_error"]),
    xytext=(2.3, 0.62), fontsize=8.5, color=INK,
    arrowprops=dict(arrowstyle="->", color=INK, lw=1.2),
)
ax.set_yscale("log")
ax.set_ylim(3e-2, 1.5)
ax.set_xlabel("variance gap between the groups (sd ratio), at n=50/10", color=MUTED, fontsize=9)
ax.set_ylabel("rate (log)", color=MUTED, fontsize=9)
ax.set_title("2. The damage peaks in the middle", color=INK, fontsize=12, weight="bold", loc="left")
ax.legend(fontsize=8, framealpha=1, edgecolor=GRID, loc="lower right")

# ------------------------------------------------ 3. the mechanism
ax = axes[1, 0]
mech = [c for c in CELLS if c["share_passed"] >= 0.01 and c["design"]["sd_ratio"] != 1.0]
labels = [c["design"]["label"] for c in mech]
y = np.arange(len(mech))
overall = [c["student_error"] for c in mech]
given = [c["student_error_given_pass"] for c in mech]
ax.barh(y + 0.19, overall, height=0.36, color=PRETEST, label="Student's error on ALL samples")
ax.barh(y - 0.19, given, height=0.36, color=STUDENT, label="...on the samples the pretest CLEARED")
ax.axvline(ALPHA, color=INK, ls="--", lw=1.8)
for i, (o, g) in enumerate(zip(overall, given)):
    ax.text(g * 1.08, i - 0.19, f"{g / o:.2f}x", va="center", fontsize=7.5, color=INK)
ax.set_xscale("log")
ax.set_xlim(5e-4, 1.5)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8, color=MUTED)
ax.invert_yaxis()
ax.set_xlabel("Student's false-positive rate (log scale)", color=MUTED, fontsize=9)
ax.set_title("3. The gate hands Student's its worst cases", color=INK, fontsize=12, weight="bold", loc="left")
ax.legend(fontsize=8, framealpha=1, edgecolor=GRID, loc="lower right")

# ------------------------------------------------ 4. does a liberal alpha help
ax = axes[1, 1]
alphas = [s["pretest_alpha"] for s in SWEEP]
gated = [s["gated_error"] for s in SWEEP]
welch_ref = SWEEP[-1]["welch_error"]
x = np.arange(len(alphas))
colors = [GATED] * (len(alphas) - 1) + [WELCH]
ax.bar(x, gated, width=0.6, color=colors)
ax.axhline(ALPHA, color=INK, ls="--", lw=1.8)
ax.axhline(welch_ref, color=WELCH, ls=":", lw=1.6)
for i, v in enumerate(gated):
    ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([f"{a:g}" for a in alphas], fontsize=9, color=MUTED)
ax.set_ylim(0, max(gated) * 1.22)
ax.set_xlabel(
    f"pretest alpha  (1.0 = always switch, i.e. just use Welch)   n=50/10, sd 1:{peak['design']['sd_ratio']:g}",
    color=MUTED, fontsize=9,
)
ax.set_ylabel("false-positive rate under a TRUE null", color=MUTED, fontsize=9)
ax.set_title("4. A more generous pretest helps, by deleting itself",
             color=INK, fontsize=12, weight="bold", loc="left")

d = R["derived"]
fig.suptitle(
    "Run the variance pretest, then pick a t-test: "
    f"{d['gated_beats_welch_materially_count']} of {d['gate_score']['cells']} designs where that "
    f"beat just using Welch, and {d['gated_broken_count']} where it broke a 5% test",
    color=INK, fontsize=14, weight="bold", x=0.012, ha="left", y=0.985,
)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig("assumption_pretest_cost_audit.png", dpi=170, facecolor=BG)
fig.savefig("assumption_pretest_cost_audit.svg", facecolor=BG)
print("wrote assumption_pretest_cost_audit.png / .svg")
