"""The audit figure. Every value read from results.json, so the chart cannot drift from evidence.txt.

Panel 1 is the mechanism rather than a result: every possible table on a 20/20 design drawn as a
lattice, coloured by which tests reject it. The disagreement between the tests is literally the
band of cells between their rejection boundaries, and a table of p-values cannot show that.

The geometry self-check runs BEFORE savefig. Day 180 found the earlier order (save, then assert)
left a finished-looking PNG on disk when the check failed.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
C_Z, C_YATES, C_FISHER, C_BARNARD = "#c8562b", "#b8860b", "#6b7684", "#2f6fdb"
COL = {"z": C_Z, "yates": C_YATES, "fisher": C_FISHER, "barnard": C_BARNARD}
LAB = {"z": "z-test = chi-square", "yates": "chi-square + Yates", "fisher": "Fisher exact",
       "barnard": "Barnard exact"}
TESTS = ("z", "yates", "fisher", "barnard")

R = json.load(open("results.json"))
C = R["config"]
EX = R["exemplar"]
LAT = R["lattice"]

fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=BG)
for ax in axes.ravel():
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)

# ------------------------------------------------------------ 1. the lattice of every table
ax = axes[0, 0]
code = np.array(LAT["code"])
n = LAT["n"]
cmap = ListedColormap(["#f1f3f6", C_Z, C_BARNARD, "#2f7a5a", "#000000"])
ax.imshow(code, origin="lower", cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest",
          extent=(-0.5, n + 0.5, -0.5, n + 0.5), aspect="auto")
ax.set_xlim(-0.5, n + 14.5)   # blank column on the right holds the legend and the callout
ax.plot([-0.5, n + 0.5], [-0.5, n + 0.5], color=MUTED, lw=1.0, ls="--")
ax.plot([EX["x2"]], [EX["x1"]], "o", ms=16, mfc="none", mec=INK, mew=2.4)
ax.annotate("", xy=(EX["x2"] + 0.6, EX["x1"] + 0.3), xytext=(n + 1.3, 3.0),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
ax.text(n + 1.5, 3.0, f"control {EX['x1']}/{n}, variant {EX['x2']}/{n}\n"
                  f"z p = {EX['z']:.3f}   Barnard p = {EX['barnard']:.3f}\n"
                  f"Fisher p = {EX['fisher']:.3f}   Yates p = {EX['yates']:.3f}",
        fontsize=10, color=INK, fontweight="bold", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc=BG, ec=GRID))
ax.text(14.6, 12.2, "no difference\n(diagonal)", fontsize=9, color=MUTED, rotation=45,
        ha="center", va="center")
cts = LAT["counts"]
ax.legend(handles=[Patch(color="#2f7a5a", label=f"all three reject  ({cts['all']})"),
                   Patch(color=C_BARNARD, label=f"z and Barnard, NOT Fisher  ({cts['z_barnard']})"),
                   Patch(color=C_Z, label=f"z only  ({cts['z_only']})"),
                   Patch(color="#f1f3f6", label=f"none  ({cts['none']})")],
          loc="upper right", fontsize=9.5, frameon=True, facecolor=BG, edgecolor=GRID)
ax.set_xlabel(f"variant conversions out of {n}", color=MUTED)
ax.set_ylabel(f"control conversions out of {n}", color=MUTED)
ax.set_xticks(range(0, n + 1, 4))
ax.spines["right"].set_visible(False)
ax.set_yticks(range(0, n + 1, 4))
ax.set_title(f"1.  Every possible {n}/{n} table. The coloured bands are where the tests disagree.",
             loc="left", fontsize=12.5, fontweight="bold", color=INK)

# ------------------------------------------------------ 2. exact size across the true rate
ax = axes[0, 1]
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
row = next(r for r in R["size"] if (r["n1"], r["n2"]) == (20, 80))
ps = C["size_ps"]
ax.axhspan(C["band"][0], C["band"][1], color="#eef1f5", zorder=0)
ax.axhline(C["alpha"], color=INK, lw=1.2)
for t in TESTS:
    ax.plot(ps, row[t + "_curve"], color=COL[t], lw=2.4 if t in ("z", "barnard") else 1.8,
            label=f"{LAB[t]}   worst {row[t]:.4f}")
ax.set_ylim(0, 0.105)
ax.set_xlim(0, 0.51)
ax.set_xlabel("true conversion rate, the same in both arms (so every rejection is a false alarm)",
              color=MUTED)
ax.set_ylabel("EXACT false-alarm rate at alpha = 0.05", color=MUTED)
ax.legend(loc="upper right", fontsize=9.5, frameon=False)
ax.text(0.505, 0.0505 + 0.006, "nominal 5%", ha="right", fontsize=9, color=INK)
ax.set_title("2.  A 20 vs 80 split: z runs hot, Fisher and Yates run cold",
             loc="left", fontsize=12.5, fontweight="bold", color=INK)

# ------------------------------------------------------------------------- 3. exact power
ax = axes[1, 0]
ax.grid(True, axis="y", color=GRID, lw=0.7)
ax.set_axisbelow(True)
pw = R["power"]
size_by = {(r["n1"], r["n2"]): r for r in R["size"]}
xs = np.arange(len(pw))
wbar = 0.2
for k, t in enumerate(TESTS):
    for i, r in enumerate(pw):
        hot = size_by[(r["n1"], r["n2"])][t + "_verdict"] == "INFLATED"
        ax.bar(i + (k - 1.5) * wbar, r[t], width=wbar, color=COL[t],
               alpha=0.35 if hot else 1.0, hatch="///" if hot else None, edgecolor=BG,
               )
ax.set_xticks(xs)
ax.set_xticklabels([f"{r['p1']:.2f} vs {r['p2']:.2f}\n{r['n1']}/{r['n2']}" for r in pw],
                   fontsize=8.5)
ax.set_ylim(0, 1.18)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylabel("EXACT power", color=MUTED)
ax.set_xlabel("true conversion rates and arm sizes", color=MUTED)
# legend swatches built by hand: taking them from the first bar would copy its hatching
ax.legend(handles=[Patch(color=COL[t], label=LAB[t]) for t in TESTS],
          loc="upper left", fontsize=9, frameon=False, ncol=2)
gmax = max(pw, key=lambda r: r["barnard"] - r["fisher"])
ax.text(0.99, 0.975, f"hatched = that test is INFLATED on that design,\nso its lead is not power.\n"
                     f"Barnard over Fisher: up to +{gmax['barnard'] - gmax['fisher']:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=INK)
ax.set_title("3.  Barnard buys power back from Fisher without giving up size",
             loc="left", fontsize=12.5, fontweight="bold", color=INK)

# ---------------------------------------------------------------- 4. the rule of five
ax = axes[1, 1]
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
e_all, s_all = [], []
for r in R["size"]:
    for p, s in zip(ps, r["z_curve"]):
        e_all.append(min(r["n1"], r["n2"]) * min(p, 1 - p))
        s_all.append(s)
e_all, s_all = np.array(e_all), np.array(s_all)
broken = s_all > C["band"][1]
ax.scatter(e_all[~broken], s_all[~broken], s=14, color=C_BARNARD, alpha=0.55,
           label="z-test fine at that cell")
ax.scatter(e_all[broken], s_all[broken], s=26, color=C_Z, label="z-test inflated at that cell")
ax.axvline(5, color=INK, lw=1.6, ls="--")
ax.axhline(C["alpha"], color=INK, lw=1.0)
ax.set_xscale("log")
ax.set_ylim(0, 0.125)
ax.set_xlabel("smallest expected cell count under the null  (log scale)", color=MUTED)
ax.set_ylabel("EXACT z-test false-alarm rate", color=MUTED)
ec = R["expected_count_rule"]
ax.text(0.03, 0.97, f"rule says USE FISHER (<5)\n{ec['unsafe_but_fine']} fine, "
                    f"{ec['unsafe_broken']} inflated",
        transform=ax.transAxes, va="top", fontsize=10, color=INK, fontweight="bold")
ax.text(0.62, 0.97, f"rule says z IS SAFE (>=5)\n{ec['safe_ok']} fine, "
                    f"{ec['safe_but_broken']} inflated",
        transform=ax.transAxes, va="top", fontsize=10, color=INK, fontweight="bold")
ax.legend(loc="lower right", fontsize=9.5, frameon=False)
ax.set_title("4.  'Expected count below 5' misses half the broken cells",
             loc="left", fontsize=12.5, fontweight="bold", color=INK)

fig.suptitle("Did conversion actually move?  Four standard tests, one 2x2 table: two say yes, two say no",
             fontsize=17, fontweight="bold", color=INK, x=0.012, ha="left", y=0.985)
fig.text(0.012, 0.948,
         "Every number EXACT - summed over all possible tables with binomial weights, no simulation. "
         "The z-test and the uncorrected chi-square are the same test.",
         fontsize=10.5, color=MUTED, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))

# --- geometry self-check on the PAINTED figure, BEFORE saving ------------------------------
fig.canvas.draw()
rend = fig.canvas.get_renderer()
for axx in axes.ravel():
    items = [(t.get_text()[:30], t.get_window_extent(renderer=rend))
             for t in axx.texts if t.get_text().strip()]
    leg = axx.get_legend()
    if leg is not None:
        items.append(("<legend>", leg.get_window_extent(renderer=rend)))
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            assert not items[i][1].overlaps(items[j][1]), \
                f"text collision: {items[i][0]!r} vs {items[j][0]!r}"
# the exemplar callout must not cover a coloured (rejecting) lattice cell
p1 = axes[0, 0]
boxes = [t.get_window_extent(renderer=rend) for t in p1.texts if t.get_text().startswith("control")]
boxes.append(p1.get_legend().get_window_extent(renderer=rend))
for box in boxes:
    for i in range(n + 1):
        for j in range(n + 1):
            x, y = p1.transData.transform((j, i))
            assert not (box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1), \
                f"callout or legend covers lattice cell ({i}, {j})"
print("geometry self-check passed: no text collisions; callout and legend clear of the lattice")

fig.savefig("proportion_test_audit.png", dpi=200, facecolor=BG)
fig.savefig("proportion_test_audit.svg", facecolor=BG)
print("wrote proportion_test_audit.png and .svg")
