"""Render binning_audit.png - sentinels, the monotonicity price, IV inflation, and the payoff."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from binning import (
    MISSING,
    SENTINEL_NO_BUREAU,
    SPECIAL,
    audit,
    build_dataset,
    fit,
    noise_screen,
)
from downstream import model_lift, robustness

data = build_dataset()
y = data["y"]
tr, ho = data["train_idx"], data["holdout_idx"]
F = data["features"]

INK = "#1d2433"
GOOD = "#0f766e"
BAD = "#c2410c"
COOL = "#1d4ed8"
MUTED = "#94a3b8"

fig, axes = plt.subplots(1, 4, figsize=(21, 5.3))
fig.suptitle(
    "Binning is easy. Knowing whether the IV you just measured is real is the job.",
    fontsize=14,
    fontweight="bold",
    color=INK,
    y=0.99,
)

# ---- Panel 1: WOE curve with missing / special bins called out --------------------------
ax = axes[0]
scheme = fit(F["n_inquiries"][tr], y[tr], feature="n_inquiries", specials=(SENTINEL_NO_BUREAU,))
labels, woes, colors = [], [], []
for b in scheme.bins:
    labels.append(b.label.replace(" (special)", "\n(no bureau)"))
    woes.append(scheme.woe(b))
    colors.append(BAD if b.kind in (SPECIAL, MISSING) else COOL)

bars = ax.bar(range(len(woes)), woes, color=colors, width=0.68)
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9, fontweight="bold", color=INK)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
ax.axhline(0, color=INK, linewidth=0.9)
ax.set_ylabel("WOE  (higher = riskier)")
ax.set_title(
    "SEPARATE THE SENTINELS\n-999 'no bureau record' is the riskiest group on the book",
    fontsize=11,
    color=BAD,
    fontweight="bold",
    pad=10,
)
ax.set_ylim(min(woes) - 0.28, max(woes) + 0.34)

# ---- Panel 2: monotonicity cost, real signal vs noise ----------------------------------
ax = axes[1]
pairs = []
for name in ("utilization", "income", "age", "months_employed"):
    free = fit(F[name][tr], y[tr], feature=name, monotone=False)
    forced = fit(F[name][tr], y[tr], feature=name, monotone=True)
    pairs.append((name, free.iv, forced.iv))

names = [p[0] for p in pairs]
pos = np.arange(len(names))
ax.barh(pos + 0.19, [p[1] for p in pairs], height=0.36, color=MUTED, label="unconstrained")
ax.barh(pos - 0.19, [p[2] for p in pairs], height=0.36, color=GOOD, label="monotone")
for i, (name, free, forced) in enumerate(pairs):
    pct = (free - forced) / free if free else 0
    ax.text(
        free + 0.006,
        i + 0.19,
        f"-{pct:.0%}",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=BAD if pct > 0.15 else INK,
    )
ax.set_yticks(pos)
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("IV")
ax.set_xlim(0, max(p[1] for p in pairs) * 1.22)
ax.legend(fontsize=8.5, loc="lower right", frameon=False)
ax.invert_yaxis()
ax.set_title(
    "MONOTONICITY HAS A PRICE\nsmall = the wiggle was noise; large = you are deleting signal",
    fontsize=11,
    color=GOOD,
    fontweight="bold",
    pad=10,
)

# ---- Panel 3: IV inflation on pure noise ----------------------------------------------
ax = axes[2]
rows = noise_screen(y[tr][:480], n_columns=12, n_permutations=40)
x = np.arange(len(rows))
ax.plot(x, [r["mean_iv"] for r in rows], "o-", color=BAD, linewidth=2.4, markersize=8,
        label="mean raw IV")
ax.plot(x, [r["mean_excess"] for r in rows], "s-", color=GOOD, linewidth=2.4, markersize=7,
        label="mean IV above permutation null")
ax.axhline(0.10, color=INK, linestyle="--", linewidth=1.2, alpha=0.75)
ax.text(
    -0.06,
    0.113,
    'conventional 0.10 "medium predictor" bar',
    fontsize=8.5,
    color=INK,
    style="italic",
)
for i, r in enumerate(rows):
    ax.annotate(
        f"{r['kept_by_iv']:.0%} kept",
        (i, r["mean_iv"]),
        textcoords="offset points",
        xytext=(0, 11),
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color=BAD,
    )
ax.set_xticks(x)
ax.set_xticklabels(
    [str(r["settings"]).replace(", ", "\n") for r in rows], fontsize=8, rotation=0
)
ax.set_ylabel("IV on a column with NO signal")
ax.set_ylim(-0.02, max(r["mean_iv"] for r in rows) * 1.32)
ax.legend(fontsize=8.5, loc="upper left", frameon=False)
ax.set_title(
    "RAW IV INFLATES WITH BIN COUNT\n12 pure rng.normal() columns, 480 rows each",
    fontsize=11,
    color=BAD,
    fontweight="bold",
    pad=10,
)

# ---- Panel 4: and does any of it produce a better model? -------------------------------
ax = axes[3]
lift = {r["arm"][0]: r for r in model_lift(data)}
rob = robustness()
arms = ["A", "B", "C"]
arm_labels = ["A raw\ncontinuous", "B constrained\nWOE bins", "C loose\nWOE bins"]
arm_colors = [MUTED, GOOD, BAD]
xs = np.arange(3)

ax.bar(
    xs - 0.19,
    [lift[a]["auc_train"] for a in arms],
    0.38,
    color=arm_colors,
    alpha=0.33,
    label="train AUC",
)
ax.bar(xs + 0.19, [lift[a]["auc_holdout"] for a in arms], 0.38, color=arm_colors, label="holdout AUC")
for i, a in enumerate(arms):
    ax.annotate(
        "%.4f" % lift[a]["auc_holdout"],
        (xs[i] + 0.19, lift[a]["auc_holdout"]),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        fontsize=8.8,
        fontweight="bold",
        color=INK,
    )
ax.annotate(
    "most total IV (%.2f),\nbest train AUC,\nworst holdout of the two" % lift["C"]["total_iv_train"],
    xy=(1.81, lift["C"]["auc_train"]),
    xytext=(0.34, 0.783),
    fontsize=8.4,
    color=BAD,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=BAD, lw=1.2),
)
ax.set_xticks(xs)
ax.set_xticklabels(arm_labels, fontsize=8.6)
ax.set_ylim(0.66, 0.80)
ax.set_ylabel("AUC")
ax.legend(fontsize=8.5, loc="lower left", frameon=False)
ax.set_title(
    "IV RANKS THEM BACKWARDS\nconstrained beats loose in %d/%d datasets, raw in %d/%d"
    % (
        rob["constrained_beats_loose"],
        rob["n_seeds"],
        rob["constrained_beats_raw"],
        rob["n_seeds"],
    ),
    fontsize=11,
    color=GOOD,
    fontweight="bold",
    pad=10,
)

for ax in axes:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.22, linestyle=":")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)

noise = audit(F["noise"][tr], y[tr], F["noise"][ho], y[ho], feature="noise")
util = audit(F["utilization"][tr], y[tr], F["utilization"][ho], y[ho], feature="utilization")
fig.text(
    0.5,
    0.015,
    f"On the full 7,200-row sample: utilization excess IV {util.excess_iv:.3f} (p={util.p_value:.3f}, keep) "
    f"vs noise excess IV {noise.excess_iv:+.4f} (p={noise.p_value:.2f}, drop).",
    ha="center",
    fontsize=9.5,
    color=INK,
    style="italic",
)
fig.tight_layout(rect=(0, 0.045, 1, 0.94))
fig.savefig("binning_audit.png", dpi=150, bbox_inches="tight", facecolor="white")
print("wrote binning_audit.png")
