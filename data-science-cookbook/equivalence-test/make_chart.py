"""The audit figure. Every value read from results.json, so the chart cannot drift from evidence.txt.

Three panels: how often each report is issued, how much evidence each report carries, and a drawn
schematic of the mechanism - the two intervals against the margin that decide the four outcomes.
The geometry self-check runs BEFORE savefig (Day 180: save-then-assert left a finished-looking PNG
on disk when the check failed): no two text boxes overlap, and every text box sits inside its axes.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, BG, BAND = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff", "#eef1f5"
NHST, TOST, WARN = "#c8562b", "#2f6fdb", "#b8860b"

R = json.load(open("results.json"))
C = R["config"]
G = R["grid"]


def col(f: float, key: str):
    rows = sorted((r for r in G if r["delta_frac"] == f), key=lambda r: r["n"])
    return [r["n"] for r in rows], [r[key] for r in rows]


fig = plt.figure(figsize=(18, 6.6), facecolor=BG)
axes = [fig.add_axes([0.05, 0.14, 0.27, 0.70]), fig.add_axes([0.38, 0.14, 0.27, 0.70]),
        fig.add_axes([0.70, 0.14, 0.28, 0.70])]
for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
texts = []

# ---------------------------------------------------------------- A. how often each report is issued
ax = axes[0]
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
n, ns = col(1.0, "not_significant")
ax.plot(n, ns, color=NHST, lw=2.4, marker="o", ms=4, label="t-test: 'no significant\ndifference', when the\ntrue diff = the margin")
n, eq = col(0.0, "equivalent")
ax.plot(n, eq, color=TOST, lw=2.4, marker="o", ms=4, label="TOST: 'equivalent',\nwhen the true diff = 0")
ax.axhline(0.8, color=MUTED, lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_xticks([10, 20, 50, 100, 200, 500])
ax.set_xticklabels(["10", "20", "50", "100", "200", "500"])
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("n per group", color=INK)
ax.set_ylabel("probability", color=INK)
a20 = next(r for r in G if r["n"] == 20 and r["delta_frac"] == 1.0)["not_significant"]
e20 = next(r for r in G if r["n"] == 20 and r["delta_frac"] == 0.0)["equivalent"]
ax.set_title(f"A. At n = 20 the wrong report is {a20 / e20:.0f}x likelier", loc="left", color=INK, fontsize=12.5, weight="bold")
need = next(r for r in R["n_needed"] if r["delta_frac"] == 0.0)["n_exact"]
ax.axvline(need, color=TOST, lw=0.8, ls=":")
texts.append(ax.text(need * 1.07, 0.72, f"n = {need}: TOST reaches\n80% when truly equal",
                     color=TOST, fontsize=9.5, va="center"))
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9)

# ---------------------------------------------------------------- B. evidence carried by each report
ax = axes[1]
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
L = R["likelihood_ratio"]
ax.plot([r["n"] for r in L], [r["lr_not_significant"] for r in L], color=NHST, lw=2.4, marker="o", ms=4,
        label="'no significant difference'")
ax.plot([r["n"] for r in L], [r["lr_equivalent"] for r in L], color=TOST, lw=2.4, marker="o", ms=4,
        label="'equivalent' (TOST)")
ax.axhline(20, color=MUTED, lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(0.8, 3e3)
ax.set_xticks([10, 20, 50, 100, 200, 500])
ax.set_xticklabels(["10", "20", "50", "100", "200", "500"])
ax.set_xlabel("n per group", color=INK)
ax.set_ylabel("likelihood ratio  P(report | equal) / P(report | diff = margin)", color=INK, fontsize=9.5)
ax.set_title("B. Evidence of absence depends on n", loc="left", color=INK, fontsize=12.5, weight="bold")
texts.append(ax.text(11, 26, "TOST cap = power / alpha = 20", color=MUTED, fontsize=9))
l20 = next(r for r in L if r["n"] == 20)
texts.append(ax.text(22, 1.05, f"n=20: 'not significant' is worth {l20['lr_not_significant']:.2f}x",
                     color=NHST, fontsize=9.5))
ax.legend(loc="upper left", frameon=False, fontsize=9.5)

# ---------------------------------------------------------------- C. the mechanism, drawn
ax = axes[2]
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-0.4, 4.6)
ax.set_yticks([])
ax.set_xticks([-1, 0, 1])
ax.set_xticklabels(["-margin", "0", "+margin"])
ax.axvspan(-1, 1, color=BAND, zorder=0)
ax.axvline(0, color=MUTED, lw=0.8)
for x0 in (-1, 1):
    ax.axvline(x0, color=INK, lw=1.2, ls="--")
cases = [  # (y, centre, half90, half95, label, colour)
    (3.8, 0.10, 0.45, 0.55, "equivalent, not different", TOST),
    (2.8, 0.55, 0.30, 0.36, "equivalent AND different", TOST),
    (1.8, 0.30, 1.05, 1.25, "inconclusive: the n=20 report", WARN),
    (0.8, 1.35, 0.40, 0.48, "different, not equivalent", NHST),
]
for y, c, h90, h95, lab, colour in cases:
    ax.plot([c - h95, c + h95], [y, y], color=colour, lw=1.3, alpha=0.55)
    ax.plot([c - h90, c + h90], [y, y], color=colour, lw=5, solid_capstyle="butt")
    ax.plot([c], [y], "o", color="white", mec=colour, mew=2, ms=7, zorder=3)
    texts.append(ax.text(-2.1, y + 0.30, lab, color=colour, fontsize=10, weight="bold", zorder=5,
                         bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none")))
texts.append(ax.text(-2.1, -0.25, "thick = 90% CI (TOST: inside the band?)   thin = 95% CI (t-test: covers 0?)",
                     color=MUTED, fontsize=8.8, zorder=5,
                     bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none")))
ax.set_title("C. Two intervals, one band, four outcomes", loc="left", color=INK, fontsize=12.5, weight="bold")

fig.suptitle(f"'No significant difference' is not 'no difference'.  Margin = {C['margin_sd']} SD, alpha = "
             f"{C['alpha']}, Student t, exact probabilities", x=0.05, ha="left", y=0.965, color=INK,
             fontsize=14, weight="bold")

# ---------------------------------------------------------------- geometry self-check, before saving
fig.canvas.draw()
rend = fig.canvas.get_renderer()
boxes = [t.get_window_extent(rend) for t in texts]
for i in range(len(boxes)):
    ax_box = texts[i].axes.get_window_extent(rend)
    b = boxes[i]
    assert ax_box.x0 - 1 <= b.x0 and b.x1 <= ax_box.x1 + 1 and ax_box.y0 - 1 <= b.y0 and b.y1 <= ax_box.y1 + 1, \
        f"text {texts[i].get_text()!r} leaves its axes"
    for j in range(i + 1, len(boxes)):
        assert not b.overlaps(boxes[j]), f"{texts[i].get_text()!r} overlaps {texts[j].get_text()!r}"

fig.savefig("equivalence_audit.png", dpi=300, facecolor=BG)
fig.savefig("equivalence_audit.svg", facecolor=BG)
print("wrote equivalence_audit.png / .svg")
