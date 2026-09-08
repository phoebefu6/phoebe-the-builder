"""Six panels, all of them read from results.json - run evidence.py first."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, BAD, COOL, GOOD, GRID, WARN = "#16222e", "#b3402f", "#2b6ca3", "#1f7a5c", "#dfe5ea", "#c98a1a"
R = json.load(open("results.json"))
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold"})


def clean(ax: plt.Axes) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.4), dpi=150)

# ---------------------------------------------------------------- 1. the closed form
ax = axes[0][0]
rows = R["s2_table"]
labels = [f"K={r['k']}\nate={r['ate']:.2f}" for r in rows]
x = np.arange(len(rows))
ax.bar(x - 0.26, [r["measured"] for r in rows], 0.26, color=INK, label="measured")
ax.bar(x, [r["naive_cf"] for r in rows], 0.26, color=BAD, label="1-(1-0.025)$^K$")
ax.bar(x + 0.26, [r["cf"] for r in rows], 0.26, color=GOOD, label="with the effect in it")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("P(scan calls a segment harmed)")
ax.set_title("The quoted arithmetic is the wrong arithmetic")
ax.legend(fontsize=7.5, frameon=False)
clean(ax)

# ---------------------------------------------------------------- 2. scaling in n
ax = axes[0][1]
rows = R["s2_scaling_in_n"]
ns = [r["n"] for r in rows]
ax.plot(ns, [r["measured"] for r in rows], "o-", color=BAD, label="false 'harmed segment'")
ax.plot(ns, [r["cf"] for r in rows], "--", color=INK, lw=1, label="closed form")
ax.plot(ns, [r["power"] for r in rows], "s-", color=COOL, label="the experiment's own power")
ax.set_xscale("log")
ax.set_xlabel("users in the test")
ax.set_ylabel("probability")
ax.set_title("Not flat in n - the reference is a real effect,\nnot a zero one")
ax.legend(fontsize=7.5, frameon=False)
clean(ax)

# ---------------------------------------------------------------- 3. winner's curse
ax = axes[0][2]
vals = [R["s3_worst_reported"], R["s3_worst_closed_form"], R["s3_worst_replication"], 0.05]
names = ["reported\n(worst of 20)", "predicted by\nE[min $Z_{20}$]", "same segment,\nfresh sample", "the truth"]
cols = [BAD, WARN, COOL, GOOD]
ax.bar(names, vals, 0.6, color=cols)
for i, v in enumerate(vals):
    if v > 0:
        ax.text(i, v + 0.006, f"{v:+.4f}", ha="center", fontsize=8)
    else:
        ax.text(i, v * 0.5, f"{v:+.4f}", ha="center", va="center", fontsize=8, color="white")
ax.axhline(0, color=INK, lw=0.9)
ax.set_ylabel("effect in that segment")
ax.set_title("The minimum operator produced the harm,\nnot the segment")
clean(ax)

# ---------------------------------------------------------------- 4. the guard's power
ax = axes[1][0]
rows = R["s4_power"]
dl = [r["delta"] for r in rows]
ax.plot(dl, [r["prereg"] for r in rows], "o-", color=GOOD, label="one segment, named in advance")
ax.plot(dl, [r["bonf"] for r in rows], "s-", color=BAD, label="scan 20, Bonferroni")
ax.plot(dl, [r["bh"] for r in rows], "^--", color=WARN, lw=1, label="scan 20, BH")
ax.plot(dl, [r["overall"] for r in rows], "d-", color=COOL, label="the overall test (positive)")
ax.set_xlabel("harm in the affected segment")
ax.set_ylabel("P(detect)")
ax.set_title("Naming the segment first is free;\nscanning for it costs 59% of detections")
ax.legend(fontsize=7.5, frameon=False)
clean(ax)

# ---------------------------------------------------------------- 5. fitted subgroup
ax = axes[1][1]
s6 = R["s6_null"]
groups = ["worst-leaf\neffect", "CI covers\nthe truth", "calls it\nharmed"]
ins = [s6["in_sample"], s6["cov_in"], s6["sig_in"]]
hon = [s6["honest"], s6["cov_honest"], s6["sig_honest"]]
x = np.arange(3)
ax.bar(x - 0.2, ins, 0.4, color=BAD, label="scored on the rows that chose the split")
ax.bar(x + 0.2, hon, 0.4, color=GOOD, label="scored on held-out rows")
ax.axhline(0.05, color=INK, lw=0.8, ls=":")
ax.axhline(0.95, color=INK, lw=0.8, ls=":")
ax.text(2.45, 0.96, "0.95", fontsize=7, color=INK, ha="right")
ax.text(2.45, 0.07, "truth 0.05", fontsize=7, color=INK, ha="right")
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_title("A subgroup that does not exist,\nreported as harmed by half the runs")
ax.legend(fontsize=7, frameon=False, loc="upper left")
clean(ax)

# ---------------------------------------------------------------- 6. k effective
ax = axes[1][2]
rows = R["s5_table"]
rh = [r["rho"] for r in rows]
ax.plot(rh, [r["k_eff"] for r in rows], "o-", color=INK, label="tests the family behaved like")
ax.axhline(20, color=BAD, lw=1.2, ls="--", label="the Bonferroni factor people use (20)")
ax.set_ylim(0, 22)
ax.set_xlabel("correlation between the slice definitions")
ax.set_ylabel("k effective")
ax.set_title("20 overlapping cuts were never 20 tests\n- half-sample slices share users")
ax.legend(fontsize=7.5, frameon=False)
clean(ax)

fig.suptitle(
    "It worked overall and hurt one segment - what a segment scan and a CATE tree each report "
    "about a world with no heterogeneity in it",
    fontsize=12,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig("heterogeneous_effects_audit.png", bbox_inches="tight")
fig.savefig("heterogeneous_effects_audit.svg", bbox_inches="tight")
print("wrote heterogeneous_effects_audit.png + .svg")
