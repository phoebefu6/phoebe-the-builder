"""The exploration-budget figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Top: the mechanism drawn - one evaluation period's applicants as a strip, split by who would lend to them.
Production's approvals carry labels already; the challenger-only band is all targeted explore ever lends into;
the band nobody wants is what uniform explore pays for, and where the fairness audit's missing good payers live.
Bottom left: net cost vs the chance of ranking the two challengers right. Bottom right: the fairness bias that
targeted spend cannot buy down. Geometry self-check runs BEFORE savefig.
"""

from __future__ import annotations

import json
from typing import List

import matplotlib

matplotlib.use("Agg")
import explore as E
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

plt.rcParams["text.parse_math"] = False

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
BAD, OK, BLUE, HOT = "#c8562b", "#2e8b57", "#2f6fdb", "#e3b23c"

R = json.load(open("results.json"))["main"]

fig = plt.figure(figsize=(16, 10.5), facecolor=BG)
axM = fig.add_axes([0.05, 0.60, 0.90, 0.30])
axP = fig.add_axes([0.07, 0.08, 0.48, 0.40])
axF = fig.add_axes([0.66, 0.08, 0.31, 0.40])
texts: List = []

# ---------------------------------------------------------------- M. the mechanism: who would lend to whom
ax = axM
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(-2, 10)
fig.text(0.05, 0.945, "Who gets a label, and who pays for it", fontsize=14, color=INK, weight="bold")
prod = E.APPROVE * 100
pool_share = E.APPROVE * (1 - R["agree"]["mean"]) * 100 * 1.25  # challenger-only band, drawn to rough scale
bands = [
    (0, prod, OK, "production approved", "labeled for free\n(repay rates known)"),
    (
        prod,
        prod + pool_share,
        HOT,
        "a challenger wants,\nproduction refused",
        f"targeted explore lends here\nrepays {R['challenger_only_repay']['mean']:.0%}",
    ),
    (
        prod + pool_share,
        100,
        "#e9edf2",
        "nobody wants",
        f"uniform explore also lends here\nrepays far less; all refused rows {R['rejected_repay']['mean']:.0%}",
    ),
]
for x0, x1, col, head, sub in bands:
    ax.add_patch(
        FancyBboxPatch(
            (x0 + 0.3, 4.2),
            x1 - x0 - 0.6,
            3.0,
            boxstyle="round,pad=0,rounding_size=0.6",
            fc=col,
            ec=INK,
            lw=1,
        )
    )
    texts.append(
        ax.text((x0 + x1) / 2, 5.7, head, ha="center", va="center", fontsize=10.5, color=INK, weight="bold")
    )
    texts.append(ax.text((x0 + x1) / 2, 2.6, sub, ha="center", va="center", fontsize=9.5, color=INK))
# sampled ticks: what each design actually lends to
rng = np.random.default_rng(3)
for xv in rng.uniform(0.8, 99.2, 34):
    ax.plot([xv, xv], [7.4, 8.1], color=BLUE, lw=1.4)
for xv in rng.uniform(prod + 0.8, prod + pool_share - 0.8, 12):
    ax.plot([xv, xv], [3.6, 4.1], color=BAD, lw=1.4)
texts.append(
    ax.text(
        0.5,
        8.9,
        "uniform: a random slice of EVERY applicant is approved (blue ticks)",
        fontsize=9.5,
        color=BLUE,
    )
)
texts.append(
    ax.text(
        prod + 0.5,
        0.35,
        "targeted: only challenger-only rows are sampled (orange ticks), weighted 1/q",
        fontsize=9.5,
        color=BAD,
    )
)
texts.append(
    ax.text(
        99.5,
        -1.3,
        f"{R['unreachable_good_b']['mean']:.0%} of group B's good payers sit in the grey band: only uniform can label them",
        fontsize=9.5,
        color=MUTED,
        ha="right",
    )
)

# ---------------------------------------------------------------- P. cost vs power
ax = axP
for design, grid, col, lab in (
    ("uniform", E.UNIFORM, BLUE, "uniform explore"),
    ("targeted", E.TARGETED, BAD, "targeted explore"),
):
    c = [R[f"{design}_{s:g}"]["net_cost"]["mean"] / 1000 for s in grid]
    p = [R[f"{design}_{s:g}"]["p_right"]["mean"] * 100 for s in grid]
    lo = [R[f"{design}_{s:g}"]["p_right"]["lo"] * 100 for s in grid]
    ax.plot(c, p, "o-", color=col, lw=2, label=lab)
    ax.fill_between(c, lo, p, color=col, alpha=0.12, lw=0)
ax.axhline(95, color=MUTED, ls=":", lw=1)
ax.set_xscale("log")
ax.set_xlim(8, 800)
ax.set_ylim(74, 103)
ax.set_xticks([10, 25, 50, 100, 250, 500])
ax.set_xticklabels(["$10k", "$25k", "$50k", "$100k", "$250k", "$500k"])
ax.set_xlabel(f"net cost of the extra loans, per {E.PER:,} applicants (log scale)")
ax.set_ylabel("chance the slice ranks the better model first (%)")
ax.set_title("Same verdict, a quarter of the bill", loc="left", fontsize=12, color=INK, weight="bold")
u, t = E.cheapest(R, "uniform"), E.cheapest(R, "targeted")
d5 = R["uniform_0.05"]["net_cost"]["mean"] / 1000
texts.append(
    ax.annotate(
        f"targeted 2%: ${t['net_cost'] / 1000:.0f}k",
        xy=(t["net_cost"] / 1000, 99.5),
        xytext=(12, 100.5),
        fontsize=9.5,
        color=BAD,
        va="center",
        arrowprops=dict(arrowstyle="-|>", color=BAD, lw=1),
    )
)
texts.append(
    ax.annotate(
        f"uniform 2%: ${u['net_cost'] / 1000:.0f}k",
        xy=(u["net_cost"] / 1000, 99.5),
        xytext=(110, 81),
        fontsize=9.5,
        color=BLUE,
        va="center",
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1),
    )
)
texts.append(
    ax.annotate(
        f"Day 200's 5%: ${d5:.0f}k",
        xy=(d5, 100),
        xytext=(240, 91),
        fontsize=9.5,
        color=MUTED,
        va="center",
        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1),
    )
)
texts.append(ax.text(750, 95.6, "95% bar", fontsize=9, color=MUTED, ha="right"))
texts.append(ax.text(60, 76, "shaded: down to the worst of 3 seeds", fontsize=9, color=MUTED))
leg = ax.legend(frameon=False, loc="lower left", fontsize=9.5, bbox_to_anchor=(0.0, 0.08))
ax.spines[["top", "right"]].set_visible(False)

# ---------------------------------------------------------------- F. fairness bias that money cannot buy
ax = axF
pts = [("uniform", 0.05, "uniform 5%"), ("targeted", 0.05, "targeted 5%"), ("targeted", 0.5, "targeted 50%")]
vals = [R[f"{d}_{s:g}"]["tpr_b_bias"]["mean"] * 100 for d, s, _ in pts]
cols = [BLUE, BAD, BAD]
xs = np.arange(len(pts))
ax.bar(xs, vals, 0.55, color=cols)
for x, v, (d, s, _) in zip(xs, vals, pts):
    texts.append(ax.text(x, v + 0.4, f"{v:+.1f}", ha="center", fontsize=10, color=INK))
    texts.append(
        ax.text(
            x,
            -1.6,
            f"cost ${R[f'{d}_{s:g}']['net_cost']['mean'] / 1000:.0f}k",
            ha="center",
            fontsize=9,
            color=MUTED,
        )
    )
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(list(xs))
ax.set_xticklabels([n for _, _, n in pts], fontsize=9.5)
ax.set_ylim(-2.5, 13)
ax.set_ylabel("bias in group B's TPR estimate (pts)")
ax.set_title("Targeted cannot audit fairness", loc="left", fontsize=12, color=INK, weight="bold")
texts.append(ax.text(2.35, 11.6, "10x the spend,\nsame bias", fontsize=9, color=BAD, ha="right", va="center"))
ax.spines[["top", "right"]].set_visible(False)


def check_geometry() -> None:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    Wd, Ht = fig.bbox.width, fig.bbox.height
    allt = [tx for tx in texts if tx.get_text()] + [axP.title, axF.title] + list(leg.get_texts())
    for a in (axP, axF):
        allt += [tk for tk in a.get_xticklabels() + a.get_yticklabels() if tk.get_text() and tk.get_visible()]
        allt += [a.xaxis.label, a.yaxis.label]
    boxes = []
    for tx in allt:
        # Annotation.get_window_extent includes its arrow; measure the label text alone
        bb = matplotlib.text.Text.get_window_extent(tx, rend)
        if bb.width == 0:
            continue
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= Wd and bb.y1 <= Ht, (
            f"text outside figure: {tx.get_text()!r}"
        )
        boxes.append((tx.get_text(), bb))
    lb = leg.get_window_extent(rend)
    for tx in texts:
        if tx.axes is axP and tx.get_text():
            bb = tx.get_window_extent(rend)
            assert not (bb.x0 < lb.x1 and lb.x0 < bb.x1 and bb.y0 < lb.y1 and lb.y0 < bb.y1), (
                f"legend covers {tx.get_text()!r}"
            )
    # band text must stay inside its own band (a text-vs-text check cannot see a patch edge)
    for (x0, x1, *_), i in zip(bands, range(3)):
        for tx in texts[2 * i : 2 * i + 2]:
            bb = tx.get_window_extent(rend).transformed(axM.transData.inverted())
            assert bb.x0 >= x0 and bb.x1 <= x1, f"band text crosses its band: {tx.get_text()!r}"
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            # 3px gap required: boxes that merely touch read as one run-on label (caught 'weighted 1/q15%' on the PNG)
            if a.x0 < b.x1 + 3 and b.x0 < a.x1 + 3 and a.y0 < b.y1 + 3 and b.y0 < a.y1 + 3:
                raise AssertionError(f"overlap: {boxes[i][0]!r} / {boxes[j][0]!r}")


check_geometry()
fig.savefig("audit.png", dpi=150, facecolor=BG)
fig.savefig("audit.svg", facecolor=BG)
print("wrote audit.png / audit.svg")
