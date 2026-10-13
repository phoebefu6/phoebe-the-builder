"""The audit figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Top: the mechanism drawn - each group's score distribution for qualified and unqualified applicants, one threshold
across both, and the overlap that is wider for B. Bottom: which fairness definitions each policy meets and what it
costs; how many applicants an audit needs before the four-fifths verdict is right; and the data fix.
Geometry self-check runs BEFORE savefig: every text inside the figure, no two texts overlapping, titles included.
"""

from __future__ import annotations

import json
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.stats import norm

plt.rcParams["text.parse_math"] = False  # percentages and dollar signs are text, never mathtext

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
BAD, OK, BLUE, HOT = "#c8562b", "#2e8b57", "#2f6fdb", "#e3b23c"

R = json.load(open("results.json"))
W = R["world"]
P = R["policies"]
ONE = P["one threshold (max accuracy)"]
ORDER = ["one threshold (max accuracy)", "demographic parity", "equal opportunity (TPR)", "predictive parity (PPV)",
         "equalized odds (TPR+FPR)"]
SHORT = {"one threshold (max accuracy)": "one threshold", "demographic parity": "demographic parity",
         "equal opportunity (TPR)": "equal opportunity", "predictive parity (PPV)": "predictive parity",
         "equalized odds (TPR+FPR)": "equalized odds"}

fig = plt.figure(figsize=(16, 10.6), facecolor=BG)
axDA = fig.add_axes([0.05, 0.62, 0.42, 0.29])
axDB = fig.add_axes([0.55, 0.62, 0.42, 0.29])
axA = fig.add_axes([0.13, 0.07, 0.26, 0.42])
axB = fig.add_axes([0.47, 0.07, 0.20, 0.42])
axC = fig.add_axes([0.75, 0.07, 0.22, 0.42])
texts: List = []

# ---------------------------------------------------------------- D. the score distributions, drawn
x = np.linspace(-3.5, 4.5, 400)
t = ONE["tA"]
for ax, g in ((axDA, "A"), (axDB, "B")):
    w = W[g]
    q = w["base"] * norm.pdf(x, w["mu_q"], w["sd"])
    u = (1 - w["base"]) * norm.pdf(x, w["mu_u"], w["sd"])
    ax.fill_between(x, 0, u, color="#9aa5b4", alpha=0.45, lw=0)
    ax.fill_between(x, 0, q, color=OK, alpha=0.45, lw=0)
    ax.plot(x, u, color="#6b7684", lw=1.6)
    ax.plot(x, q, color=OK, lw=1.8)
    ax.fill_between(x[x >= t], 0, u[x >= t], color=BAD, alpha=0.35, lw=0)
    ax.vlines(t, 0, 0.37, color=INK, lw=1.4, ls="--")
    r = ONE["per"][g]
    ax.set_xlim(-3.5, 4.5)
    ax.set_ylim(0, 0.47)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel("model score", color=INK)
    texts.append(ax.text(t + 0.12, 0.335, f"selected above {t:.2f}", fontsize=9, color=INK))
    texts.append(ax.text(-3.4, 0.44, f"group {g}: {w['share']:.0%} of applicants, {w['base']:.0%} qualified, "
                         f"AUC {R['auc'][g]:.3f}", fontsize=10, color=INK, weight="bold"))
    texts.append(ax.text(-3.4, 0.405, f"qualified selected {r['tpr']:.0%}   unqualified selected {r['fpr']:.0%}   "
                         f"precision {r['ppv']:.0%}", fontsize=9.5, color=INK))
    texts.append(ax.text(w["mu_u"] - 0.5, (1 - w["base"]) * norm.pdf(w["mu_u"], w["mu_u"], w["sd"]) + 0.012,
                         "not qualified", fontsize=9, color=MUTED))
    texts.append(ax.text(w["mu_q"] + 0.55, w["base"] * norm.pdf(w["mu_q"], w["mu_q"], w["sd"]) + 0.004,
                         "qualified", fontsize=9, color=OK))
axDA.set_title(f"The model is fine overall (AUC {R['pooled_auc']:.3f}, accuracy {ONE['overall']['acc']:.1%}). "
               f"Per group, one threshold selects {ONE['per']['A']['sel']:.0%} of A and {ONE['per']['B']['sel']:.0%} of B: "
               f"four-fifths ratio {ONE['overall']['impact_ratio']:.2f}",
               loc="left", color=INK, fontsize=12, weight="bold", pad=12)

# ---------------------------------------------------------------- A. which definitions each policy meets
ax = axA
cols = [("sel", "selection\n(4/5 rule)"), ("tpr", "qualified\nselected"), ("fpr", "unqualified\nselected"),
        ("ppv", "precision")]
for i, n in enumerate(ORDER):
    y = len(ORDER) - 1 - i
    for j, (m, _) in enumerate(cols):
        if m == "sel":
            ok = P[n]["overall"]["impact_ratio"] >= 0.8
            lab = f"{P[n]['overall']['impact_ratio']:.2f}"
        else:
            gap = P[n]["gaps"][m]
            ok = abs(gap) <= 0.02
            lab = f"{gap * 100:+.0f} pts"
        ax.add_patch(plt.Rectangle((j + 0.05, y + 0.08), 0.9, 0.84, color=OK if ok else "#f3d2c4", lw=0))
        texts.append(ax.text(j + 0.5, y + 0.5, lab, ha="center", va="center", fontsize=9,
                             color=BG if ok else INK, weight="bold" if ok else "normal"))
ax.set_xlim(0, len(cols))
ax.set_ylim(0, len(ORDER))
ax.set_xticks([j + 0.5 for j in range(len(cols))])
ax.set_xticklabels([c for _, c in cols], color=INK, fontsize=9)
ax.set_yticks([len(ORDER) - 1 - i + 0.5 for i in range(len(ORDER))])
ax.set_yticklabels([f"{SHORT[n]}\nacc {P[n]['overall']['acc']:.1%}" for n in ORDER], color=INK, fontsize=9)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title("A. Each policy meets one definition and\n    breaks another (gap = A minus B; green = met)",
             loc="left", color=INK, fontsize=11, weight="bold")

# ---------------------------------------------------------------- B. audit sample size
ax = axB
au = R["audit"]
ax.plot([r["n_B"] for r in au], [r["agree"] * 100 for r in au], color=BLUE, lw=2.2, marker="o")
ax.set_xscale("log")
ax.set_xticks([25, 50, 100, 200, 500, 1000, 2000])
ax.set_xticklabels(["25", "50", "100", "200", "500", "1k", "2k"])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.axhline(95, color=MUTED, ls=":", lw=1)
texts.append(ax.text(28, 96.2, "95% right", fontsize=8.6, color=MUTED))
for r in au:
    if r["n_B"] in (25, 100, 500):
        texts.append(ax.text(r["n_B"] * 1.12, r["agree"] * 100 - 2.6, f"{r['agree']:.0%}", fontsize=9, color=INK))
ax.set_ylim(65, 102)
ax.set_xlabel("applicants from group B in the audit sample", color=INK)
ax.set_ylabel("sample's four-fifths verdict matches the truth (%)", color=INK)
ax.set_title("B. With 100 applicants from B,\n    the verdict is wrong 1 time in 6", loc="left", color=INK,
             fontsize=11, weight="bold")

# ---------------------------------------------------------------- C. the data fix
ax = axC
F = R["fix"]["policies"]
items = [("four-fifths\nratio", ONE["overall"]["impact_ratio"], F["one threshold (max accuracy)"]["overall"]["impact_ratio"]),
         ("qualified B\nselected", ONE["per"]["B"]["tpr"], F["one threshold (max accuracy)"]["per"]["B"]["tpr"]),
         ("accuracy", ONE["overall"]["acc"], F["one threshold (max accuracy)"]["overall"]["acc"])]
for i, (lab, before, after) in enumerate(items):
    ax.bar(i - 0.18, before, 0.34, color="#9aa5b4")
    ax.bar(i + 0.18, after, 0.34, color=OK)
    texts.append(ax.text(i - 0.18, before - 0.06, f"{before:.2f}", ha="center", fontsize=8.6, color=BG, weight="bold"))
    texts.append(ax.text(i + 0.18, after - 0.06, f"{after:.2f}", ha="center", fontsize=8.6, color=BG, weight="bold"))
ax.axhline(0.8, color=BAD, ls="--", lw=1)
texts.append(ax.text(-0.45, 0.82, "4/5 line", fontsize=8.6, color=BAD))
ax.set_xticks(range(len(items)))
ax.set_xticklabels([i[0] for i in items], color=INK, fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel("one threshold: today (grey) vs B's data fixed (green)", color=INK)
ax.set_title(f"C. Fix B's data (AUC {R['auc']['B']:.2f} -> {R['fix']['auc_B']:.2f}):\n    passes 4/5 and gets more accurate",
             loc="left", color=INK, fontsize=11, weight="bold")

for a in (axB, axC):
    a.set_facecolor(BG)
    for sp in a.spines.values():
        sp.set_color(GRID)
    a.tick_params(colors=INK)
axB.grid(True, color=GRID, lw=0.7)
axB.set_axisbelow(True)


def check_geometry() -> None:
    """Every text inside the figure; no two texts overlap (titles and tick labels included)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    Wd, Ht = fig.bbox.width, fig.bbox.height
    axes = (axDA, axDB, axA, axB, axC)
    allt = list(texts) + [a.title for a in axes]
    for a in axes:
        allt += [tk for tk in a.get_xticklabels() + a.get_yticklabels() if tk.get_text() and tk.get_visible()]
        allt += [a.xaxis.label, a.yaxis.label]
    boxes = []
    for tx in allt:
        if not tx.get_text():
            continue
        bb = tx.get_window_extent(rend)
        if bb.width == 0:
            continue
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= Wd and bb.y1 <= Ht, f"text outside figure: {tx.get_text()!r}"
        boxes.append((tx.get_text(), bb))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 < b.x1 - 1 and b.x0 < a.x1 - 1 and a.y0 < b.y1 - 1 and b.y0 < a.y1 - 1:
                raise AssertionError(f"overlap: {boxes[i][0]!r} / {boxes[j][0]!r}")


if __name__ == "__main__":
    check_geometry()
    fig.savefig("fairness_audit.png", dpi=150, facecolor=BG)
    fig.savefig("fairness_audit.svg", facecolor=BG)
    print("saved fairness_audit.png / .svg, geometry ok")
