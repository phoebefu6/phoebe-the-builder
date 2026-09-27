"""The audit figure. Every value read from results.json, so the chart cannot drift from evidence.txt.

Three panels: power of the two analyses of the same data as the pairing tightens, the pairs each
needs for 80% power, and a drawn picture of the mechanism on the exemplar's 12 stores - the spread
BETWEEN stores (what the independent analysis divides by) against the spread of the before/after
differences (what the paired test divides by). The geometry self-check runs BEFORE savefig (Day 180).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID, BG, BAND = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff", "#eef1f5"
PAIR, IND, WARN = "#2f6fdb", "#c8562b", "#b8860b"

R = json.load(open("results.json"))
C = R["config"]
G = R["grid"]
N_SHOW = 20

fig = plt.figure(figsize=(18, 6.6), facecolor=BG)
axes = [fig.add_axes([0.05, 0.14, 0.27, 0.70]), fig.add_axes([0.38, 0.14, 0.27, 0.70]),
        fig.add_axes([0.71, 0.14, 0.27, 0.70])]
for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
texts = []

# ---------------------------------------------------------------- A. power vs rho, one dataset, two analyses
ax = axes[0]
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
rows = sorted((r for r in G if r["n"] == N_SHOW), key=lambda r: r["rho"])
rho = [r["rho"] for r in rows]
ax.plot(rho, [r["power_paired"] for r in rows], color=PAIR, lw=2.4, marker="o", ms=4, label="paired t")
ax.plot(rho, [r["power_independent"] for r in rows], color=IND, lw=2.4, marker="o", ms=4,
        label="same data, two-sample t")
ax.plot(rho, [r["size_independent"] for r in rows], color=IND, lw=1.2, ls="--",
        label="two-sample t, false-positive rate")
ax.axhline(C["alpha"], color=MUTED, lw=0.8, ls=":")
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("correlation between before and after (rho)", color=INK)
ax.set_ylabel(f"probability of rejecting, n = {N_SHOW} pairs", color=INK)
r8 = next(r for r in rows if r["rho"] == 0.8)
ax.set_title(f"A. At rho = 0.8: power {r8['power_paired']:.2f} vs {r8['power_independent']:.2f}",
             loc="left", color=INK, fontsize=12.5, weight="bold")
rn = next(r for r in rows if r["rho"] == -0.5)
texts.append(ax.text(-0.48, 0.16, f"rho < 0: two-sample t\nover-rejects ({rn['size_independent']:.3f})",
                     color=IND, fontsize=9.5, va="bottom"))
ax.legend(loc="upper left", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9)

# ---------------------------------------------------------------- B. pairs needed
ax = axes[1]
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
Nn = R["n_needed"]
xr = [r["rho"] for r in Nn]
ax.fill_between(xr, [r["n_paired"] for r in Nn], [r["n_independent_analysis"] for r in Nn], color=BAND, zorder=0)
ax.plot(xr, [r["n_paired"] for r in Nn], color=PAIR, lw=2.4, marker="o", ms=4, label="paired t")
ax.plot(xr, [r["n_independent_analysis"] for r in Nn], color=IND, lw=2.4, marker="o", ms=4, label="two-sample t")
ax.set_ylim(0, 72)
ax.set_xlabel("correlation between before and after (rho)", color=INK)
ax.set_ylabel(f"pairs for 80% power at a {C['delta_sd']} SD shift", color=INK)
n9 = next(r for r in Nn if r["rho"] == 0.9)
ax.set_title(f"B. At rho = 0.9: {n9['n_paired']} pairs or {n9['n_independent_analysis']}", loc="left",
             color=INK, fontsize=12.5, weight="bold")
texts.append(ax.text(0.72, 34, "pairs the wrong\nanalysis wastes", color=MUTED, fontsize=10, ha="center"))
ax.legend(loc="lower left", frameon=False, fontsize=9.5)

# ---------------------------------------------------------------- C. the mechanism, drawn on the exemplar
ax = axes[2]
ex = R["exemplar"]
b, a = np.array(ex["before"]), np.array(ex["after"])
d = a - b
ax.set_xlim(-0.6, 3.4)
lo_y = min(b.min(), a.min()) - 1
ax.set_ylim(lo_y - 10, max(b.max(), a.max()) + 1.5)
ax.set_xticks([0, 1, 2.6])
ax.set_xticklabels(["before", "after", "after - before"])
ax.set_yticks([])
for yb, ya in zip(b, a):
    ax.plot([0, 1], [yb, ya], color=MUTED, lw=1, alpha=0.8, zorder=1)
ax.scatter(np.zeros_like(b), b, color=INK, s=18, zorder=2)
ax.scatter(np.ones_like(a), a, color=INK, s=18, zorder=2)
# the two denominators (per-observation scale), drawn as bars of their true length, centred
s_ind = float(np.sqrt(np.var(b, ddof=1) + np.var(a, ddof=1)))
s_pair = float(np.std(d, ddof=1))
base = float(np.mean(np.r_[b, a]))
ax.plot([-0.35, -0.35], [base - s_ind, base + s_ind], color=IND, lw=6, solid_capstyle="butt")
yd = base + d - d.mean()
ax.scatter(np.full_like(d, 2.6), yd, color=PAIR, s=18, zorder=2)
ax.plot([3.05, 3.05], [base - s_pair, base + s_pair], color=PAIR, lw=6, solid_capstyle="butt")
texts.append(ax.text(-0.55, lo_y - 2.2, f"two-sample t noise: sqrt(s_b^2 + s_a^2) = {s_ind:.1f}\n"
                     "store levels count as noise", color=IND, fontsize=9.5, va="top"))
texts.append(ax.text(1.45, lo_y - 5.6, f"paired t noise: SD of changes = {s_pair:.1f}\n"
                     "each store is its own control", color=PAIR, fontsize=9.5, va="top"))
ax.set_title(f"C. {ex['n']} stores: the level is shared, the change is not", loc="left", color=INK,
             fontsize=12.5, weight="bold")

fig.suptitle(f"Before-and-after data analysed as two groups throws the pairing away.  Shift = {C['delta_sd']} SD, "
             f"alpha = {C['alpha']}, exact probabilities", x=0.05, ha="left", y=0.965, color=INK,
             fontsize=14, weight="bold")

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

fig.savefig("paired_audit.png", dpi=300, facecolor=BG)
fig.savefig("paired_audit.svg", facecolor=BG)
print("wrote paired_audit.png / .svg")
