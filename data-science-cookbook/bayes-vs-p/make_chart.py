"""The audit figure. Every value read from results.json, so the chart cannot drift from evidence.txt.

Three panels: Lindley's curve (the Bayes factor of a p = 0.05 result as n grows), the verdict
probabilities for a real but tiny effect, and a drawn picture of the mechanism on the exemplar - the
likelihood is a narrow spike, the alternative spreads its bet over a wide prior, and the point null
collects more of the likelihood than the average of the alternative does. The geometry self-check
runs BEFORE savefig (Day 180).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

INK, MUTED, GRID, BG, BAND = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff", "#eef1f5"
FREQ, BAYES, WARN = "#2f6fdb", "#c8562b", "#b8860b"

R = json.load(open("results.json"))
C = R["config"]

fig = plt.figure(figsize=(18, 6.6), facecolor=BG)
axes = [fig.add_axes([0.05, 0.14, 0.27, 0.70]), fig.add_axes([0.38, 0.14, 0.27, 0.70]),
        fig.add_axes([0.71, 0.14, 0.27, 0.70])]
for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
texts = []

# ---------------------------------------------------------------- A. Lindley's curve
ax = axes[0]
ax.grid(True, color=GRID, lw=0.7, which="major")
ax.set_axisbelow(True)
L = R["lindley"]
ns = [r["n"] for r in L]
ax.axhspan(1, 1000, color=BAND, zorder=0)
ax.plot(ns, [r["bf01_jzs"] for r in L], color=BAYES, lw=2.4, marker="o", ms=4, label="JZS, r = 0.707 (software default)")
ax.plot(ns, [r["bf01_tau1"] for r in L], color=BAYES, lw=1.3, ls="--", label="normal prior, tau = 1")
ax.plot(ns, [r["bf01_tau01"] for r in L], color=WARN, lw=1.6, marker="s", ms=3.5, label="normal prior, tau = 0.1")
ax.axhline(1, color=INK, lw=0.9)
for k in (3, 10):
    ax.axhline(k, color=MUTED, lw=0.7, ls=":")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(0.3, 300)
ax.set_xlabel("sample size n (p held at exactly 0.05)", color=INK)
ax.set_ylabel("Bayes factor BF01  (above 1 = evidence FOR no effect)", color=INK)
big = L[-1]
ax.set_title(f"A. p = 0.05 at n = 1M: BF01 = {big['bf01_jzs']:.0f} for the null", loc="left", color=INK,
             fontsize=12.5, weight="bold")
texts.append(ax.text(6e4, 1.25, "evidence for H0", color=MUTED, fontsize=9.5))
texts.append(ax.text(6e4, 0.72, "evidence for H1", color=MUTED, fontsize=9.5, va="top"))
ax.legend(loc="upper left", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9)

# ---------------------------------------------------------------- B. a real but tiny effect
ax = axes[1]
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
T = R["tiny_effect"]
tn = [r["n"] for r in T]
ax.plot(tn, [r["p_significant"] for r in T], color=FREQ, lw=2.4, marker="o", ms=4, label="p < 0.05")
ax.plot(tn, [r["bf10_over_3"] for r in T], color=BAYES, lw=2.4, marker="o", ms=4, label="BF10 > 3 (evidence for effect)")
ax.plot(tn, [r["bf01_over_3"] for r in T], color=BAYES, lw=1.4, ls="--", marker="o", ms=3,
        label="BF01 > 3 (evidence of NO effect)")
ax.set_xscale("log")
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("sample size n", color=INK)
ax.set_ylabel("probability of the verdict, true effect = 0.02 SD", color=INK)
v20 = next(r for r in T if r["n"] == 20000)
ax.set_title(f"B. n = 20k: {v20['p_significant']:.0%} 'significant', {v20['bf01_over_3']:.0%} 'no effect'",
             loc="left", color=INK, fontsize=12.5, weight="bold")
ax.axvline(20000, color=MUTED, lw=0.8, ls=":")
texts.append(ax.text(18000, 0.9, "same effect,\nsame n", color=MUTED, fontsize=9.5, ha="right"))
ax.legend(loc="lower right", bbox_to_anchor=(1, 0.08), frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9)

# ---------------------------------------------------------------- C. the mechanism, drawn on the exemplar
ax = axes[2]
ex = R["exemplar"]
se = 1 / np.sqrt(ex["n"])
dh = ex["d"]
x = np.linspace(-0.06, 0.08, 800)
lik = np.exp(-0.5 * ((x - dh) / se) ** 2)  # peak scaled to 1
ax.fill_between(x, lik, color=FREQ, alpha=0.18, lw=0)
ax.plot(x, lik, color=FREQ, lw=2.2)
l0 = float(np.exp(-0.5 * (dh / se) ** 2))
ax.plot([0, 0], [0, l0], color=INK, lw=3.5, solid_capstyle="butt")
ax.scatter([0], [l0], color=INK, s=30, zorder=3)
# the alternative's average height = likelihood averaged over the N(0, 1) prior
avg = float(stats.norm.pdf(dh, 0, np.sqrt(1 + se ** 2)) * se * np.sqrt(2 * np.pi))
ax.axhline(avg, color=BAYES, lw=2.2)
share = float(stats.norm.cdf(dh + 2 * se) - stats.norm.cdf(dh - 2 * se))
ax.set_xlim(-0.06, 0.08)
ax.set_ylim(-0.04, 1.12)
ax.set_yticks([])
ax.set_xlabel("true effect delta (SD units)", color=INK)
texts.append(ax.text(0.002, l0 + 0.03, f"H0: likelihood at 0 = {l0:.3f}", color=INK, fontsize=9.5))
texts.append(ax.text(-0.058, avg + 0.03, f"H1 average = {avg:.4f}", color=BAYES, fontsize=9.5))
texts.append(ax.text(0.035, 0.80, f"likelihood of delta\n(p = {ex['p']:.3f}, n = {ex['n']:,})", color=FREQ, fontsize=9.5))
texts.append(ax.text(-0.058, 0.52, f"H1 spreads its prior over\n+/- 2 SD: {share:.1%} of it\nlands where the data are",
                     color=BAYES, fontsize=9.5))
ax.set_title(f"C. BF01 = {l0:.3f} / {avg:.4f} = {l0 / avg:.1f}", loc="left", color=INK, fontsize=12.5, weight="bold")

fig.suptitle("Same data, two verdicts: a p = 0.05 result is evidence FOR the null once n is large.  "
             "Exact probabilities, alpha = 0.05", x=0.05, ha="left", y=0.965, color=INK, fontsize=14, weight="bold")

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

fig.savefig("bayes_audit.png", dpi=300, facecolor=BG)
fig.savefig("bayes_audit.svg", facecolor=BG)
print(f"wrote bayes_audit.png / .svg   (panel C: {l0:.4f} / {avg:.5f} = {l0 / avg:.2f}, exemplar tau=1 BF01 "
      f"{ex['bf01_normal_tau1']:.2f})")
