"""The audit figure: four panels, each one a section of evidence.py recomputed live."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ttests as tt

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
# One fill per meaning, held across all four panels: a test always wears the same colour, and
# "good" is never a colour - it is distance from the dashed nominal line.
STUDENT, WELCH, THIRD, FLAT = "#c8562b", "#2f6fdb", "#7a4fa3", "#9aa4b0"
REPS = 20_000
A = tt.ALPHA

fig, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor=BG)
for ax in axes.ravel():
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)

# ---------------------------------------- 1. Type I error, normal data, sd ratio 3
ax = axes[0, 0]
designs = [(20, 20), (30, 10), (50, 10), (10, 30), (10, 50)]
stu, wel = [], []
for k, (n1, n2) in enumerate(designs):
    r = tt.simulate(tt.Cell(n1=n1, n2=n2, sd2=3.0), reps=REPS, seed=40 + k)
    stu.append(r["student"])
    wel.append(r["welch"])
y = np.arange(len(designs))
ax.barh(y - 0.19, stu, height=0.34, color=STUDENT, label="Student (pooled)")
ax.barh(y + 0.19, wel, height=0.34, color=WELCH, label="Welch")
ax.axvline(A, color=INK, ls="--", lw=1.8)
for i, (s, w) in enumerate(zip(stu, wel)):
    ax.text(max(s, 1e-4) * 1.15, i - 0.19, f"{s:.4f}", va="center", color=INK, fontsize=8)
    ax.text(w * 1.15, i + 0.19, f"{w:.4f}", va="center", color=INK, fontsize=8)
ax.set_xscale("log")
ax.set_xlim(3e-4, 1.6)
ax.set_yticks(y)
ax.set_yticklabels([f"n={a}/{b}" for a, b in designs], color=MUTED, fontsize=9)
ax.set_xlabel("measured Type I error, nominal 0.05 (log scale)", color=MUTED)
ax.set_title("Same data, same null, one variance gap (3x)", color=INK, fontsize=12, loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower right")

# ---------------------------------------- 2. it is imbalance, not variance
ax = axes[0, 1]
ratios = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
for (n1, n2), col, ls in [((20, 20), FLAT, "-"), ((50, 10), STUDENT, "-"), ((10, 50), THIRD, "--")]:
    rates = [tt.simulate(tt.Cell(n1=n1, n2=n2, sd2=r), reps=REPS, seed=80 + j)["student"]
             for j, r in enumerate(ratios)]
    ax.plot(ratios, rates, "o-" if ls == "-" else "o--", color=col, lw=2, ms=5,
            label=f"n={n1}/{n2}")
ax.axhline(A, color=INK, ls="--", lw=1.8)
ax.text(1.05, A * 1.12, "nominal 0.05", color=INK, fontsize=9)
ax.set_yscale("log")
ax.set_xlabel("sd of group 2 / sd of group 1", color=MUTED)
ax.set_ylabel("Student's Type I error", color=MUTED)
ax.set_title("Balanced designs barely move. Unbalanced ones explode.", color=INK, fontsize=12, loc="left")
ax.legend(frameon=False, fontsize=9)

# ---------------------------------------- 3. what Welch costs when Student is valid
ax = axes[1, 0]
labels, lost = [], []
for k, (n1, n2, d) in enumerate([(10, 10, 0.8), (20, 20, 0.8), (50, 50, 0.8),
                                 (30, 10, 0.8), (50, 10, 0.8)]):
    r = tt.simulate(tt.Cell(n1=n1, n2=n2, shift=d), reps=REPS, seed=120 + k)
    labels.append(f"n={n1}/{n2}")
    lost.append(r["student"] - r["welch"])
cols = [FLAT if abs(v) < 0.01 else WELCH for v in lost]
ax.bar(range(len(labels)), lost, color=cols, width=0.55)
for i, v in enumerate(lost):
    ax.text(i, v + 0.0012, f"{v:+.4f}", ha="center", color=INK, fontsize=9)
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, color=MUTED, fontsize=9)
ax.set_ylabel("power given up by using Welch (d=0.8, equal sd)", color=MUTED)
ax.set_ylim(-0.005, max(lost) * 1.35)
ax.set_title("The premium is ~0 when balanced, real when not", color=INK, fontsize=12, loc="left")

# ---------------------------------------- 4. Welch is not a normality fix
ax = axes[1, 1]
pairs = [(50, 10), (150, 30), (500, 100), (2500, 500)]
s_ln = [tt.simulate(tt.Cell(n1=a, n2=b, dist="lognormal"), reps=REPS, seed=160 + k)
        for k, (a, b) in enumerate(pairs)]
xs = np.arange(len(pairs))
ax.plot(xs, [r["student"] for r in s_ln], "o-", color=STUDENT, lw=2, ms=6, label="Student (pooled)")
ax.plot(xs, [r["welch"] for r in s_ln], "o-", color=WELCH, lw=2, ms=6, label="Welch")
ax.axhline(A, color=INK, ls="--", lw=1.8)
for i, r in enumerate(s_ln):
    ax.text(i, r["welch"] + 0.004, f"{r['welch']:.3f}", ha="center", color=INK, fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels([f"{a}/{b}" for a, b in pairs], color=MUTED, fontsize=9)
ax.set_xlabel("group sizes, lognormal data, null exactly true", color=MUTED)
ax.set_ylabel("Type I error", color=MUTED)
ax.set_ylim(0, 0.12)
ax.set_title("Under skew the ranking reverses, and n=3,000 undoes it", color=INK, fontsize=12, loc="left")
ax.legend(frameon=False, fontsize=9)

fig.suptitle("Four procedures are called 'the t-test'. They do not have the same error rate.",
             color=INK, fontsize=15, x=0.012, ha="left", y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("t_test_variants_audit.png", dpi=150, facecolor=BG)
fig.savefig("t_test_variants_audit.svg", facecolor=BG)
print("wrote t_test_variants_audit.png and .svg")
