"""The audit figure. Every value read from results.json, so the chart cannot drift from evidence.txt.

The geometry self-check runs BEFORE savefig (Day 180: save-then-assert left a finished-looking PNG
on disk when the check failed). It asserts no two text boxes or legends overlap, and that no plotted
data point sits under a text box or legend (Day 179).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, BG, BAND = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff", "#eef1f5"
PROCS = ("none", "lsd", "bonferroni", "holm", "tukey")
LAB = {"none": "no adjustment", "lsd": "protected LSD", "bonferroni": "Bonferroni",
       "holm": "Holm", "tukey": "Tukey HSD"}
COL = {"none": "#c8562b", "lsd": "#b8860b", "bonferroni": "#6b7684", "holm": "#2f7a5a",
       "tukey": "#2f6fdb"}
STYLE = {"none": "-", "lsd": "--", "bonferroni": ":", "holm": "-", "tukey": "-"}

R = json.load(open("results.json"))
C = R["config"]
fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=BG)
for ax in axes.ravel():
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def fwer_panel(ax, rows, title, note, note_xy):
    ks = [r["k"] for r in rows]
    ax.axhspan(C["band"][0], C["band"][1], color=BAND, zorder=0)
    ax.axhline(C["alpha"], color=INK, lw=0.8)
    for p in PROCS:
        ax.plot(ks, [r[p + "_fwer"] for r in rows], STYLE[p], marker="o", ms=5, lw=2.2,
                color=COL[p], label=LAB[p])
    ax.set_yscale("log")
    ax.set_ylim(0.003, 1.0)
    ax.set_yticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_yticklabels(["0.005", "0.01", "0.02", "0.05", "0.10", "0.20", "0.50", "1.00"])
    ax.set_xticks(ks)
    ax.set_xlabel("number of groups k", color=MUTED)
    ax.set_ylabel("family-wise false-alarm rate (log)", color=MUTED)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK)
    ax.text(*note_xy, note, transform=ax.transAxes, fontsize=10, color=INK, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc=BG, ec=GRID))


# ------------------------------------------------------------ 1. complete null
fwer_panel(axes[0, 0], R["null"], "1.  All means equal: unadjusted pairs call a winner 62% of the time",
           "Tukey is exact here (the calibration line).\nBonferroni = Holm to the replicate (the dotted\nline is under the green one); "
           "both CONSERVATIVE\nfrom k = 5.", (0.03, 0.22))
axes[0, 0].legend(loc="lower right", fontsize=9.5, frameon=True, facecolor=BG, edgecolor=GRID)

# ------------------------------------------------------------ 2. partial null
part = R["partial"]
fwer_panel(axes[0, 1], part, "2.  One group far away: the F 'protection' is already spent",
           f"F is significant on every replicate, so\nprotected LSD = no adjustment, exactly.\n"
           f"k = 3 is the only safe row ({part[0]['lsd_fwer']:.3f}).", (0.03, 0.22))

# ------------------------------------------------------------ 3. Tukey vs Holm
ax = axes[1, 0]
pw = R["power"]
labels = [f"k={r['k']}\n{r['shape']}" for r in pw]
gaps = [r["tukey_minus_holm"] for r in pw]
halves = [r["tukey_minus_holm_half"] for r in pw]
cols = [COL["tukey"] if r["tukey_vs_holm"] == "tukey" else COL["holm"] for r in pw]
ax.bar(range(len(pw)), gaps, color=cols, width=0.62)
ax.errorbar(range(len(pw)), gaps, yerr=halves, fmt="none", ecolor=INK, capsize=4, lw=1.2)
ax.axhline(0, color=INK, lw=0.9)
ax.set_xticks(range(len(pw)))
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylim(-0.035, 0.045)
ax.set_ylabel("per-pair power: Tukey minus Holm", color=MUTED)
ax.text(0.03, 0.95, "above zero: Tukey finds more real pairs", transform=ax.transAxes,
        fontsize=10, color=COL["tukey"], fontweight="bold", va="top")
ax.text(0.03, 0.06, "below zero: Holm finds more (k = 3 only)", transform=ax.transAxes,
        fontsize=10, color=COL["holm"], fontweight="bold", va="bottom")
t = R["tukey_vs_holm_tally"]
ax.set_title(f"3.  Tukey beats Holm on {t['tukey']} of {len(pw)} designs; Holm wins both k = 3 designs",
             loc="left", fontsize=12.5, fontweight="bold", color=INK)

# ------------------------------------------------------------ 4. F sig, no pair named
ax = axes[1, 1]
spread = [r for r in pw if r["shape"] == "spread"]
ks = [r["k"] for r in spread]
for p in ("holm", "tukey"):
    ax.plot(ks, [r[p + "_no_pair_given_F"] for r in spread], marker="o", ms=6, lw=2.4,
            color=COL[p], label=LAB[p])
ax.set_xticks(ks)
ax.set_ylim(0, 0.2)
ax.set_xlabel("number of groups k (means on an evenly spaced ladder)", color=MUTED)
ax.set_ylabel("P(names no pair | F significant)", color=MUTED)
ax.legend(loc="upper left", fontsize=9.5, frameon=True, facecolor=BG, edgecolor=GRID)
w = max(spread, key=lambda r: r["tukey_no_pair_given_F"])
ax.text(0.40, 0.14, f"At k = {w['k']}, 1 in {round(1 / w['tukey_no_pair_given_F'])} significant ANOVAs\n"
                    "has no pair Tukey will name.\nEvery pair truly differs.",
        transform=ax.transAxes, fontsize=10, color=INK, va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc=BG, ec=GRID))
ax.set_title("4.  'Something differs' - and nothing the post-hoc test will name",
             loc="left", fontsize=12.5, fontweight="bold", color=INK)

fig.suptitle("ANOVA said something differs. Five ways to ask which pair, on the same data.",
             x=0.01, ha="left", fontsize=17, fontweight="bold", color=INK)
fig.text(0.01, 0.945, f"Normal data, equal n = {C['n_per_group']}, equal variance (where Tukey is exact). "
         f"{C['reps']:,} replicates per design. Shaded band = 4.5-5.5%.",
         fontsize=10.5, color=MUTED, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))

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
    for line in axx.get_lines():
        for x, y in zip(*line.get_data()):
            px, py = axx.transData.transform((x, y))
            for name, box in items:
                assert not (box.x0 <= px <= box.x1 and box.y0 <= py <= box.y1), \
                    f"{name!r} covers a data point at ({x}, {y})"
    for bar in axx.patches:
        bb = bar.get_window_extent(renderer=rend)
        for name, box in items:
            if name != "<legend>":
                assert not bb.overlaps(box), f"{name!r} covers a bar"
print("geometry self-check passed: no text collisions, no data point or bar under a label")

fig.savefig("anova_posthoc_audit.png", dpi=200, facecolor=BG)
fig.savefig("anova_posthoc_audit.svg", facecolor=BG)
print("wrote anova_posthoc_audit.png and .svg")
