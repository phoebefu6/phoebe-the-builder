"""The audit figure. Four panels, every value read from results.json, so the chart cannot drift
from evidence.txt or the README - the defect Day 175 shipped when the chart grew its own copy of
the comparison and disagreed with the evidence file about four cells.

Encoding discipline: the two TESTS are the only thing colour encodes, and the same fill means the
same test in every panel. Sample size is always position, never colour. Panel 1 is a drawn
schematic of the mechanism rather than a plot of results, because the whole build rests on a
picture of two distributions that a table cannot carry.
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
T_COLOR = "#2f6fdb"        # the t-test, everywhere
MW_COLOR = "#c8562b"       # Mann-Whitney, everywhere
BAND = "#eef1f5"
WARN = "#b8860b"

R = json.load(open("results.json"))
C = R["config"]
ALPHA = C["alpha"]
LO, HI = C["band"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), facecolor=BG)
for ax in axes.ravel():
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)

# ---------------------------------------------------------------- 1. the mechanism, drawn
ax = axes[0, 0]
sd, w, mus = 0.5, (0.7, 0.3), (-1.0, 5.0)
x = np.linspace(-4.6, 7.2, 1500)
control = stats.norm.pdf(x, 0.0, sd)
treated = sum(wi * stats.norm.pdf(x, mi, sd) for wi, mi in zip(w, mus))

ax.fill_between(x, control, color=T_COLOR, alpha=0.16)
ax.plot(x, control, color=T_COLOR, lw=2.2, label="control  N(0, 0.5$^2$)")
ax.fill_between(x, treated, color=MW_COLOR, alpha=0.16)
ax.plot(x, treated, color=MW_COLOR, lw=2.2,
        label="treated  0.7$\\cdot$N($-$1, 0.5$^2$) + 0.3$\\cdot$N(5, 0.5$^2$)")

ax.axvline(0.0, color=T_COLOR, ls=":", lw=1.6)
ax.axvline(C["mix_mean"], color=MW_COLOR, ls=":", lw=1.6)
top = float(max(control.max(), treated.max()))
ax.annotate("", xy=(C["mix_mean"], top * 1.06), xytext=(0.0, top * 1.06),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
ax.text(C["mix_mean"] / 2, top * 1.10,
        f"mean difference  {C['mix_mean']:+.2f}\nthe t-test's target: treated is HIGHER",
        ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")

# The bulk of the treated mass sits BELOW the control centre - that is the whole disagreement.
# Arrows and text are separate artists on purpose: an Annotation's bounding box includes its
# arrow, which would make the collision check below either useless or unsatisfiable.
lobe_lo = 0.7 * stats.norm.pdf(0, 0, sd)
lobe_hi = 0.3 * stats.norm.pdf(0, 0, sd)
ax.text(-4.45, top * 1.60, "70% of treated cases land here,\nbelow the control centre",
        fontsize=10.5, color=MW_COLOR, fontweight="bold", ha="left", va="bottom")
ax.annotate("", xy=(-1.05, lobe_hi * 1.45), xytext=(-2.55, top * 1.56),
            arrowprops=dict(arrowstyle="-|>", color=MW_COLOR, lw=1.5,
                            connectionstyle="arc3,rad=-0.18"))
ax.text(-4.45, top * 1.02,
        f"P(treated > control) = {C['mix_prob_superiority']:.3f}\n"
        f"Mann-Whitney's target:\ntreated is LOWER",
        fontsize=10.5, color=MW_COLOR, fontweight="bold", ha="left", va="top")
ax.text(2.35, top * 0.80, "30% land far above,\nand they carry the mean",
        fontsize=10.5, color=INK, ha="left", va="center")
ax.annotate("", xy=(4.85, lobe_hi * 1.25), xytext=(4.15, top * 0.72),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4,
                            connectionstyle="arc3,rad=0.22"))
assert lobe_lo > lobe_hi

ax.set_ylim(0, top * 1.95)
ax.set_xlim(-4.6, 7.2)
ax.set_xlabel("outcome", color=MUTED)
ax.set_ylabel("density", color=MUTED)
ax.set_title("1.  Both tests are right. They are answering different questions.",
             loc="left", fontsize=13, fontweight="bold", color=INK)
ax.legend(loc="upper right", frameon=False, fontsize=9.5)
ax.set_yticks([])

# ---------------------------------------------------------------- 2. the headline
ax = axes[0, 1]
d = R["disagreement"]
ns = [int(r["n"]) for r in d]
xs = np.arange(len(ns))
ax.plot(xs, [r["t_up"] for r in d], "o-", color=T_COLOR, lw=2.2, ms=7,
        label="t-test significant, says UP")
ax.plot(xs, [r["mw_down"] for r in d], "s-", color=MW_COLOR, lw=2.2, ms=7,
        label="Mann-Whitney significant, says DOWN")
ax.plot(xs, [r["opposite"] for r in d], "D-", color=INK, lw=3.0, ms=8,
        label="BOTH significant, OPPOSITE directions")
ax.fill_between(xs, 0, [r["opposite"] for r in d], color=INK, alpha=0.08)

peak = max(d, key=lambda r: r["opposite"])
ax.annotate(f"{peak['opposite']:.0%} of studies\nat n = {int(peak['n'])}",
            xy=(len(ns) - 1, peak["opposite"]), xytext=(len(ns) - 2.45, 0.72),
            fontsize=11, color=INK, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
ax.text(0.04, 0.92, "more data makes it MORE likely, not less",
        transform=ax.transAxes, fontsize=10.5, color=MUTED, style="italic")
ax.set_xticks(xs)
ax.set_xticklabels(ns)
ax.set_ylim(0, 1.06)
ax.set_xlabel("n per group", color=MUTED)
ax.set_ylabel(f"rate over {int(C['disagree_reps']):,} replicate studies", color=MUTED)
ax.set_title("2.  The contradiction rate rises to certainty",
             loc="left", fontsize=13, fontweight="bold", color=INK)
ax.legend(loc="center right", frameon=False, fontsize=9.5)

# ---------------------------------------------------------------- 3. efficiency
ax = axes[1, 0]
eff = [e for e in R["efficiency"] if e["measured"] is not None]
eff = sorted(eff, key=lambda e: e["are"])
xs = np.arange(len(eff))
ax.bar(xs - 0.19, [e["are"] for e in eff], width=0.36, color=MUTED, alpha=0.55,
       label="predicted ARE (asymptotic)")
ax.bar(xs + 0.19, [e["measured"] for e in eff], width=0.36, color=MW_COLOR,
       label=f"measured n$_t$/n$_{{MW}}$ at n = {C['location_ref_n']}, d = {C['location_d']}")
ax.axhline(1.0, color=INK, lw=1.4, ls="--", label="equal efficiency")
for i, e in enumerate(eff):
    ax.text(i + 0.19, e["measured"] + 0.09, f"{e['measured']:.2f}", ha="center",
            fontsize=9, color=INK, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels([e["shape"] for e in eff], fontsize=10)
ax.set_ylabel("observations the t-test needs, per Mann-Whitney observation", color=MUTED)
ax.set_ylim(0, 3.4)
ax.set_title("3.  As a POWER decision the folk advice is right",
             loc="left", fontsize=13, fontweight="bold", color=INK)
ax.legend(loc="upper left", frameon=False, fontsize=9.5)
ax.text(0.52, 0.93, "above 1 = Mann-Whitney wins", transform=ax.transAxes,
        fontsize=10, color=MUTED, style="italic")

# ---------------------------------------------------------------- 4. unequal spread
ax = axes[1, 1]
u = R["unequal"]
labels, mw_vals, w_vals = [], [], []
for ratio in sorted({r["sd_ratio"] for r in u}):
    for n1, n2 in ((30, 30), (50, 10), (10, 50)):
        rows = [r for r in u if r["sd_ratio"] == ratio and r["n1"] == n1 and r["n2"] == n2]
        labels.append(f"{ratio:.0f}x\n{n1}/{n2}")
        mw_vals.append(float(np.mean([r["mw"] for r in rows])))
        w_vals.append(float(np.mean([r["welch"] for r in rows])))
xs = np.arange(len(labels))
ax.axhspan(LO, HI, color=BAND, zorder=0)
ax.axhline(ALPHA, color=INK, lw=1.3, ls="--")
ax.bar(xs - 0.19, w_vals, width=0.36, color=T_COLOR, label="Welch t-test")
ax.bar(xs + 0.19, mw_vals, width=0.36, color=MW_COLOR, label="Mann-Whitney")
for i, v in enumerate(mw_vals):
    ax.text(i + 0.19, v * 1.10, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
ax.set_yscale("log")
ax.set_ylim(0.002, 0.9)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9)
ax.set_xlabel("SD ratio  /  n$_1$ per n$_2$   (mean over the four symmetric populations)",
              color=MUTED)
ax.set_ylabel("false-positive rate, log scale", color=MUTED)
ax.set_title("4.  Same centre, different spread - and the MW null is EXACTLY true",
             loc="left", fontsize=13, fontweight="bold", color=INK)
ax.legend(loc="upper left", frameon=False, fontsize=9.5)
ax.text(len(labels) - 0.45, ALPHA * 1.09, "nominal 5%", ha="right", va="bottom",
        fontsize=9.5, color=INK)
ax.annotate("wide group in the SMALL arm:\nfires 3x too often",
            xy=(4.19, mw_vals[4] * 1.15), xytext=(3.4, 0.40), fontsize=9.5, color=WARN,
            fontweight="bold", ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-|>", color=WARN, lw=1.4))
ax.annotate("wide group in the LARGE arm:\nall but stops firing",
            xy=(8.19, mw_vals[8] * 1.25), xytext=(7.75, 0.40), fontsize=9.5, color=WARN,
            fontweight="bold", ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-|>", color=WARN, lw=1.4))

fig.suptitle('"Just use Mann-Whitney"  -  it is not a robust t-test, '
             "it is a test of a different hypothesis",
             fontsize=17, fontweight="bold", color=INK, x=0.012, ha="left", y=0.985)
fig.text(0.012, 0.945,
         f"Every value read from results.json.  Verdicts are 99% Wilson intervals; a rate is "
         f"only called broken when the whole interval clears [{LO}, {HI}].",
         fontsize=10.5, color=MUTED, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.935))
fig.savefig("nonparametric_swap_audit.png", dpi=200, facecolor=BG)
fig.savefig("nonparametric_swap_audit.svg", facecolor=BG)
print("wrote nonparametric_swap_audit.png and .svg")
assert math.isfinite(C["mix_prob_superiority"])


# --- geometry self-check on the PAINTED figure -------------------------------------------
# CLAUDE.md: verify on painted pixels, not on a source read. A text block that overlaps a curve
# looks perfectly fine in the code. This measures the rendered bounding boxes and fails the
# build rather than shipping a chart nobody re-opened.
fig.canvas.draw()
rend = fig.canvas.get_renderer()
p1 = axes[0, 0]
boxes = [(t.get_text()[:28], t.get_window_extent(renderer=rend))
         for t in p1.texts if t.get_text().strip()]
leg = p1.get_legend()
if leg is not None:
    boxes.append(("<legend>", leg.get_window_extent(renderer=rend)))
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        (na, ba), (nb, bb) = boxes[i], boxes[j]
        assert not ba.overlaps(bb), f"panel 1 text collision: {na!r} vs {nb!r}"

# ... and no annotation may sit on top of either density curve.
curve_px = np.array([p1.transData.transform((xi, yi))
                     for xi, yi in list(zip(x, control)) + list(zip(x, treated))])
for name, box in boxes:
    if name == "<legend>":
        continue
    inside = ((curve_px[:, 0] >= box.x0) & (curve_px[:, 0] <= box.x1)
              & (curve_px[:, 1] >= box.y0) & (curve_px[:, 1] <= box.y1))
    assert not inside.any(), f"panel 1 annotation {name!r} sits on a density curve"
print("geometry self-check passed: no text collisions in panel 1")
