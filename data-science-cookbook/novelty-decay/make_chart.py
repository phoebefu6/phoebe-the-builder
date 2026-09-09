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


def clean(ax: plt.Axes, axis: str = "y") -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.4), dpi=150)

# ------------------------------------------------- 1. three numbers off one curve
ax = axes[0][0]
curve = np.array(R["s1"]["curve"])
t = np.arange(len(curve))
truth = R["s1"]["truth"]
m = R["s1"]["means"]
ax.plot(t, curve, "o-", color=INK, ms=3, lw=1.4, zorder=3, label="measured lift by tenure")
ax.axhline(truth, color=GOOD, lw=1.6, ls="--", label=f"long-run truth {truth:.3f}")
for key, col, lab in (("day0", BAD, "first-day"), ("pooled", WARN, "pooled window"),
                      ("lastweek", COOL, "final week")):
    ax.axhline(m[key], color=col, lw=1.1, ls=":")
    xpos, dy = (27.2, -6) if key == "day0" else ((27.2, 4) if key == "pooled" else (13.5, 4))
    ax.annotate(f"{lab} {m[key]:.3f} ({m[key] / truth:.1f}x)", (xpos, m[key]),
                textcoords="offset points", xytext=(0, dy),
                ha="right", va="bottom" if dy > 0 else "top", fontsize=7, color=col)
ax.set_xlabel("user tenure (days)")
ax.set_ylabel("lift")
ax.set_title("Every summary is a lift AT AN AGE")
ax.legend(fontsize=7, frameon=False, loc="center right")
clean(ax)

# ------------------------------------------------- 2. the design constant
ax = axes[0][1]
rows = R["s2"]["rows"]
x = np.array([r["mult"] for r in rows], dtype=float)
ax.plot(x, [r["a_bigbang"] for r in rows], "s-", color=COOL, ms=4, label="big-bang enrollment")
ax.plot(x, [r["a_uniform"] for r in rows], "o-", color=BAD, ms=4, label="continuous enrollment")
for r in rows:
    ax.annotate(f"{r['a_uniform'] / r['a_bigbang']:.2f}x", (r["mult"], r["a_uniform"]),
                textcoords="offset points", xytext=(3, 5), fontsize=7, color=BAD)
ax.set_xlabel("test length / decay constant  (T / $\\lambda$)")
ax.set_ylabel("novelty weight A in the reported lift")
ax.set_title("How users were let in changes the number")
ax.legend(fontsize=7.5, frameon=False)
clean(ax)

# ------------------------------------------------- 3. the asymptote's CI, and r2 backwards
ax = axes[0][2]
s3 = R["s3"]
xs = np.arange(len(s3))
ax.bar(xs, [r["ci_width"] for r in s3], 0.55, color=BAD, zorder=3)
ax.set_yscale("log")
ax.axhline(truth, color=GOOD, lw=1.4, ls="--")
ax.annotate(f"the quantity itself ({truth:.2f})", (-0.35, truth), textcoords="offset points",
            xytext=(0, 5), ha="left", va="bottom", fontsize=7, color=GOOD)
ax.set_xticks(xs)
ax.set_xticklabels([f"{r['T'] / 7:.0f}$\\lambda$" for r in s3])
ax.set_ylabel("95% CI width on the asymptote (log)")
ax.set_xlabel("window length")
ax2 = ax.twinx()
ax2.plot(xs, [r["r2"] for r in s3], "o-", color=INK, ms=4, lw=1.4)
ax2.set_ylabel("$R^2$ of the fit", color=INK)
ax2.set_ylim(0, 0.4)
ax2.spines["top"].set_visible(False)
ax.set_title("The fit's own diagnostic runs backwards")
clean(ax)

# ------------------------------------------------- 4. THE HEADLINE
ax = axes[1][0]
cd = np.array(R["s5"]["curve_decay"])
cm = np.array(R["s5"]["curve_mix"])
t = np.arange(len(cd))
ax.plot(t, cd, "o-", color=INK, ms=3.4, lw=1.5, zorder=3,
        label=f"real decay - long run {R['s5']['truth_decay']:.2f}")
ax.plot(t, cm, "s--", color=BAD, ms=3.4, lw=1.5, zorder=3,
        label=f"cohort mix shift - long run {R['s5']['truth_mix']:.2f}")
ax.set_xlabel("user tenure (days)")
ax.set_ylabel("lift")
ax.set_title(f"Same plot. {R['s5']['truth_mix'] / R['s5']['truth_decay']:.0f}x apart in the decision")
ax.legend(fontsize=7.5, frameon=False)
ax.annotate(f"largest gap anywhere {R['s5']['worst_z']:.2f} $\\sigma$;\nnoise alone gives 2.30",
            (0.53, 0.55), xycoords="axes fraction", fontsize=7.5, color="#5a6b78")
clean(ax)

# ------------------------------------------------- 5. what separates them
ax = axes[1][1]
lab = ["early-vs-late\nguard |z|", "within-cohort\nslope |z|"]
dec = [abs(R["s5"]["guard_z"]["decay"]), abs(R["s5"]["slope_z"]["decay"])]
mix = [abs(R["s5"]["guard_z"]["mix"]), abs(R["s5"]["slope_z"]["mix"])]
x = np.arange(2)
ax.bar(x - 0.19, dec, 0.38, color=INK, label="real decay", zorder=3)
ax.bar(x + 0.19, mix, 0.38, color=BAD, label="cohort mix shift", zorder=3)
for xi, v in zip(x + 0.19, mix):
    ax.annotate(f"{v:.2f}", (xi, v + 0.06), ha="center", fontsize=7.5, color=BAD, zorder=4)
for xi, v in zip(x - 0.19, dec):
    ax.annotate(f"{v:.2f}", (xi, v + 0.06), ha="center", fontsize=7.5, color=INK, zorder=4)
ax.set_ylim(0, 3.9)
ax.axhline(1.96, color=WARN, lw=1.2, ls="--")
ax.annotate("|z| = 1.96", (1.45, 2.1), fontsize=7, color=WARN)
ax.set_xticks(x)
ax.set_xticklabels(lab)
ax.set_ylabel("mean |z|")
ax.set_title("The separator is a COLUMN, not a test")
ax.legend(fontsize=7.5, frameon=False)
clean(ax)

# ------------------------------------------------- 6. policies
ax = axes[1][2]
s8 = R["s8"]
names = [r["policy"] for r in s8]
y = np.arange(len(s8))[::-1]
bias = [abs(r["bias"]) for r in s8]
sd = [r["sd"] for r in s8]
ax.barh(y + 0.0, bias, 0.36, color=BAD, label="|bias|", zorder=3)
ax.barh(y - 0.36, sd, 0.36, color=COOL, label="sd", zorder=3)
for i, r in zip(y, s8):
    ax.annotate(f"RMSE {r['rmse']:.4f}", (max(abs(r["bias"]), r["sd"]) + 0.004, i - 0.18),
                fontsize=7, va="center", color=INK)
ax.set_yticks(y - 0.18)
ax.set_yticklabels(names, fontsize=7.5)
ax.set_xlabel(f"error against the long-run truth ({truth:.2f})")
ax.set_title("Which wrong number to ship on")
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
clean(ax, axis="x")

fig.suptitle(
    "Novelty decay: a measured lift is a lift at an age - and a decaying curve does not tell you the effect decayed",
    fontsize=11.5, fontweight="bold", y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig("novelty_decay_audit.png", bbox_inches="tight")
fig.savefig("novelty_decay_audit.svg", bbox_inches="tight")
print("wrote novelty_decay_audit.png + .svg")
