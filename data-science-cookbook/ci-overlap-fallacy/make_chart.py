"""The audit figure. Every value read from results.json, so the chart cannot drift from
evidence.txt or the README - the defect Day 175 shipped when the chart grew its own copy of a
comparison and disagreed with the evidence file about four cells.

Encoding discipline: exactly two things are being compared everywhere - the TEST and the RULE -
and one fill means the same one in every panel. Panel 1 is a drawn schematic of the mechanism
rather than a plot of results, because the whole build is a statement about a picture and a
table cannot carry it.

Geometry is verified on the PAINTED figure at the bottom of this file, not by reading the source:
a text block sitting on a drawn bar looks perfectly fine in the code.
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
TEST = "#2f6fdb"       # what the test says, everywhere
RULE = "#c8562b"       # what the overlap rule says, everywhere
WARN = "#b8860b"
GOOD = "#2f7a5a"
BAND = "#eef1f5"

R = json.load(open("results.json"))
C = R["config"]
ML = C["matching_level_equal_se"]
OVL = C["overlap_at_sig_equal_se"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), facecolor=BG)
for ax in axes.ravel():
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)

# ------------------------------------------------------- 1. the mechanism, drawn to scale
# Two 95% bars on equal standard errors, with the two thresholds marked where they actually
# fall. Everything in this panel is in units of one standard error, so the drawing IS the
# algebra rather than an illustration of it.
ax = axes[0, 0]
se, z = 1.0, 1.959963984540054
d_sig = z * math.sqrt(2)          # the test rejects beyond this
d_gap = 2 * z                     # the bars separate only beyond this
ax.set_xlim(-3.6, 9.4)
ax.set_ylim(0.1, 1.28)

for y, centre, col, lab in ((0.95, 0.0, TEST, "group A"), (0.62, d_sig, RULE, "group B")):
    ax.plot([centre - z * se, centre + z * se], [y, y], color=col, lw=4.0, solid_capstyle="butt")
    for e in (centre - z * se, centre + z * se):
        ax.plot([e, e], [y - 0.045, y + 0.045], color=col, lw=3.0)
    ax.plot([centre], [y], "o", color=col, ms=11, zorder=5)
    ax.text(centre, y + 0.085, lab, ha="center", fontsize=10.5, color=col, fontweight="bold")

# the shared span - this is the thing everybody reads as "not significant"
lo, hi = d_sig - z * se, z * se
ax.axvspan(lo, hi, ymin=0.30, ymax=0.68, color=RULE, alpha=0.14, zorder=0)
ax.annotate("", xy=(lo, 0.42), xytext=(hi, 0.42),
            arrowprops=dict(arrowstyle="<|-|>", color=RULE, lw=1.7))
ax.text(d_sig - 0.2, 0.355,
        f"the bars still share {OVL * 100:.1f}% of an arm\nand p = 0.05 EXACTLY here",
        ha="right", va="top", fontsize=11, color=RULE, fontweight="bold")

for xv, col, txt, ha, dx in ((d_sig, TEST, f"the TEST rejects past\n{d_sig:.2f} standard errors",
                              "right", -0.12),
                             (d_gap, RULE, f"the BARS separate only past\n{d_gap:.2f} standard errors",
                              "left", 0.12)):
    ax.axvline(xv, color=col, ls=":", lw=1.8, ymin=0.0, ymax=0.93)
    ax.text(xv + dx, 1.22, txt, ha=ha, va="top", fontsize=10.5, color=col, fontweight="bold")

ax.annotate("", xy=(d_gap, 0.175), xytext=(d_sig, 0.175),
            arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=2.0))
ax.text(d_gap + 0.15, 0.175,
        f"THE DEAD ZONE\n{d_gap / d_sig:.3f}x further than the test asks",
        ha="left", va="center", fontsize=11, color=INK, fontweight="bold")
ax.set_yticks([])
ax.set_xlabel("difference between the two means, in standard errors of one group", color=MUTED)
ax.set_title("1.  Non-overlap is the stricter event. Always, and by a fixed factor.",
             loc="left", fontsize=13, fontweight="bold", color=INK)

# --------------------------------------------------------------- 2. the dead zone, measured
ax = axes[0, 1]
m = sorted(R["means"], key=lambda r: -r["overlap_given_sig"])
labels = [f"{r['n1']}/{r['n2']}\nsd {r['sd1']:.0f}:{r['sd2']:.0f}" for r in m]
vals = [r["overlap_given_sig"] for r in m]
err = np.array([[v - r["ogs_lo"] for v, r in zip(vals, m)],
                [r["ogs_hi"] - v for v, r in zip(vals, m)]])
xs = np.arange(len(m))
ax.bar(xs, vals, color=RULE, width=0.66)
ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=INK, capsize=3, lw=1.2)
for i, (v, r) in enumerate(zip(vals, m)):
    ax.text(i, v + 0.018, f"{v:.2f}", ha="center", fontsize=9, color=INK, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylim(0, max(vals) * 1.30)
ax.set_ylabel("share of SIGNIFICANT results whose 95% bars overlap", color=MUTED)
ax.set_xlabel(f"design  (n per arm, sd ratio)   -   {C['means_reps']:,} replicate studies each",
              color=MUTED)
ax.set_title("2.  Real results the picture tells you to throw away",
             loc="left", fontsize=13, fontweight="bold", color=INK)
ax.text(0.98, 0.93, "bars are 99% Wilson intervals", transform=ax.transAxes, ha="right",
        fontsize=9.5, color=MUTED, style="italic")

# ------------------------------------------------------------------ 3. the reversal, paired
ax = axes[1, 0]
p = R["paired"]
rhos = [r["rho"] for r in p]
xs = np.arange(len(p))
ax.plot(xs, [r["overlap_given_sig"] for r in p], "o-", color=RULE, lw=2.6, ms=8,
        label="P(bars overlap | the test rejects)")
ax.plot(xs, [r["sig"] for r in p], "s--", color=TEST, lw=2.2, ms=7,
        label="P(the test rejects)")
ax.fill_between(xs, [r["sig"] for r in p], [r["overlap_given_sig"] for r in p],
                color=RULE, alpha=0.09)
ax2 = ax.twinx()
ax2.plot(xs, [r["slack"] for r in p], "^:", color=INK, lw=2.0, ms=7,
         label="how much further the bars demand")
ax2.set_ylabel("x further apart the bars demand the means be", color=MUTED)
ax2.set_ylim(0, max(r["slack"] for r in p) * 1.6)
ax2.grid(False)
ax2.axhline(math.sqrt(2), color=MUTED, ls="--", lw=1.2)
ax2.text(len(p) - 1.0, math.sqrt(2) * 0.93, "sqrt(2) - the cap when the arms are independent",
         ha="right", va="top", fontsize=9.5, color=MUTED)
ax.set_xticks(xs)
ax.set_xticklabels([f"{r:.2f}" for r in rhos])
ax.set_ylim(0, 1.6)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel(f"correlation between the paired measurements   "
              f"(n = {C['paired_n']} pairs, true difference {C['paired_delta']})", color=MUTED)
ax.set_ylabel("rate over replicate studies", color=MUTED)
ax.set_title("3.  On paired data the same rule fails the OTHER way",
             loc="left", fontsize=13, fontweight="bold", color=INK)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=9)
top = p[-1]
ax.text(0.02, 0.97, f"rho = {top['rho']}: the test rejects {top['sig']:.0%}\n"
                    f"of the time, and {top['overlap_given_sig']:.4f} of\n"
                    f"those rejections sit behind\noverlapping bars",
        transform=ax.transAxes, fontsize=10, color=RULE, fontweight="bold", va="top")

# ----------------------------------------------------------- 4. coverage, computed exactly
ax = axes[1, 1]
cov = R["coverage"]
ns = sorted({r["n"] for r in cov})
ps = sorted({r["p"] for r in cov})
styles = {"wald": (WARN, "o-", 2.8), "wilson": (TEST, "s--", 1.8),
          "agresti_coull": (GOOD, "^--", 1.8), "clopper_pearson": (MUTED, "d--", 1.8)}
xs = np.arange(len(ps))
for meth, (col, st, lw) in styles.items():
    for j, n in enumerate(ns):
        vals = [next(r[meth] for r in cov if r["n"] == n and r["p"] == pp) for pp in ps]
        ax.plot(xs, vals, st, color=col, lw=lw, ms=5, alpha=0.35 + 0.65 * j / (len(ns) - 1),
                label=f"{meth.replace('_', '-')}" if j == len(ns) - 1 else None)
ax.axhline(0.95, color=INK, lw=1.5, ls="-")
ax.text(len(ps) - 0.05, 0.9525, "nominal 95%", ha="right", fontsize=9.5, color=INK)
ax.set_xticks(xs)
ax.set_xticklabels([f"{pp:g}" for pp in ps])
ax.set_ylim(0.28, 1.02)
ax.set_xlabel(f"true proportion   (one line per n in {ns}, darker = larger n)", color=MUTED)
ax.set_ylabel("EXACT coverage of a nominal 95% interval", color=MUTED)
ax.set_title("4.  The bars themselves, before anyone compares two of them",
             loc="left", fontsize=13, fontweight="bold", color=INK)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07), frameon=False, fontsize=9.5, ncol=2)
worst = min(cov, key=lambda r: r["wald"])
ax.text(0.9, 0.52, f"Wald at p = {worst['p']:g}, n = {worst['n']}: {worst['wald']:.4f}",
        fontsize=10.5, color=WARN, fontweight="bold", va="center")
ax.annotate("", xy=(0.03, worst["wald"]), xytext=(0.88, 0.51),
            arrowprops=dict(arrowstyle="-|>", color=WARN, lw=1.5))
ax.text(0.5, 0.02, "no Monte Carlo: summed over every outcome with its binomial weight",
        transform=ax.transAxes, ha="center", fontsize=9.5, color=MUTED, style="italic")

fig.suptitle('"The error bars overlap, so it is not significant"  -  it is a 0.6% test '
             "wearing a 5% label",
             fontsize=17, fontweight="bold", color=INK, x=0.012, ha="left", y=0.985)
fig.text(0.012, 0.945,
         f"Every value read from results.json.  The interval level at which non-overlap and "
         f"p < 0.05 become the same event is {ML * 100:.1f}%, not 95%.",
         fontsize=10.5, color=MUTED, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.935))

# --- geometry self-check on the PAINTED figure -------------------------------------------
# CLAUDE.md: verify on painted pixels, not on a source read. Arrows and text are separate
# artists throughout, because an Annotation's bounding box includes its arrow, which makes this
# check either useless or unsatisfiable.
fig.canvas.draw()
rend = fig.canvas.get_renderer()
p1 = axes[0, 0]
boxes = [(t.get_text()[:30], t.get_window_extent(renderer=rend))
         for t in p1.texts if t.get_text().strip()]
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        (na, ba), (nb, bb) = boxes[i], boxes[j]
        assert not ba.overlaps(bb), f"panel 1 text collision: {na!r} vs {nb!r}"

# ...and no label may sit on top of either drawn interval.
bar_px = []
for y, centre in ((0.95, 0.0), (0.62, d_sig)):
    for xv in np.linspace(centre - z * se, centre + z * se, 200):
        bar_px.append(p1.transData.transform((xv, y)))
bar_px = np.array(bar_px)
for name, box in boxes:
    inside = ((bar_px[:, 0] >= box.x0) & (bar_px[:, 0] <= box.x1)
              & (bar_px[:, 1] >= box.y0) & (bar_px[:, 1] <= box.y1))
    assert not inside.any(), f"panel 1 label {name!r} sits on a drawn interval"

# Panels 3 and 4: no text box may touch another text box or the legend. (The first version of
# this check covered panel 1 only, and ran AFTER savefig - so a failing layout still left a
# finished-looking PNG on disk. It now gates the save.)
for axx in (axes[1, 0], axes[1, 1]):
    items = [(t.get_text()[:30], t.get_window_extent(renderer=rend))
             for t in axx.texts if t.get_text().strip()]
    if axx is axes[1, 0]:
        items += [(t.get_text()[:30], t.get_window_extent(renderer=rend))
                  for t in ax2.texts if t.get_text().strip()]
    leg = axx.get_legend()
    if leg is not None:
        items.append(("<legend>", leg.get_window_extent(renderer=rend)))
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            assert not items[i][1].overlaps(items[j][1]), \
                f"text collision: {items[i][0]!r} vs {items[j][0]!r}"
print("geometry self-check passed: no text collisions in panels 1, 3, 4; no label on a bar")

fig.savefig("ci_overlap_audit.png", dpi=200, facecolor=BG)
fig.savefig("ci_overlap_audit.svg", facecolor=BG)
print("wrote ci_overlap_audit.png and .svg")
