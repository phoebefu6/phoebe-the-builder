"""The audit figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Three panels: the 'always cite' instruction change as each checker reports it against the truth; the lexical
threshold sweep (fails an honest paraphrase vs passes a polarity flip - no tau has both low); and a drawn
picture of the mechanism - five claims about one passage, the passage each one cites, and which checkers
pass it. The geometry self-check runs BEFORE savefig (Day 180), tick labels included (Day 189).
"""

from __future__ import annotations

import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import citation as T
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
TRUTH, BAD, OK = "#1f2733", "#c8562b", "#2e8b57"
COL = {"has_citation": "#8a94a3", "context_lexical": "#8a6d3b", "cited_lexical": "#e3b23c",
       "cited_lexical+numbers": "#2f6fdb", "cited_lexical+numbers+negation": "#7a4fb5"}
SHORT = {"has_citation": "has citation", "context_lexical": "context lexical", "cited_lexical": "cited lexical",
         "cited_lexical+numbers": "cited + numbers", "cited_lexical+numbers+negation": "cited + num + negation"}
TINY = {"has_citation": "cite", "context_lexical": "ctx", "cited_lexical": "lex", "cited_lexical+numbers": "+num",
        "cited_lexical+numbers+negation": "+neg"}

R = json.load(open("results.json"))
V1, V2 = R["models"]["v1 cites when sure"], R["models"]["v2 always cite"]

fig = plt.figure(figsize=(18, 7.4), facecolor=BG)
axes = [fig.add_axes([0.135, 0.14, 0.185, 0.66]), fig.add_axes([0.385, 0.14, 0.205, 0.66]),
        fig.add_axes([0.62, 0.05, 0.37, 0.75])]
for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
texts = []

# ---------------------------------------------------------------- A. the instruction change, as each checker sees it
ax = axes[0]
ax.grid(True, color=GRID, lw=0.7, axis="x")
ax.set_axisbelow(True)
names = ["truly supported"] + list(T.CHECKERS)
vals = [R["deltas"][k] * 100 for k in names]
ys = list(range(len(names)))[::-1]
ax.barh(ys, vals, 0.62, color=[TRUTH] + [COL[k] for k in T.CHECKERS])
for y, v in zip(ys, vals):
    texts.append(ax.text(v + (0.8 if v >= 0 else -0.8), y, f"{v:+.1f}", ha="left" if v >= 0 else "right",
                         va="center", fontsize=9.5, color=INK))
ax.set_yticks(ys)
ax.set_yticklabels(["truly supported"] + [SHORT[k] for k in T.CHECKERS], color=INK, fontsize=10)
ax.set_xlim(-14, 30)
ax.set_xlabel("change in reported pass rate, v1 -> v2 (points)", color=INK)
ax.set_title("A. 'Always cite' cost 8 points of support", loc="left", color=INK, fontsize=12.5, weight="bold")
texts.append(ax.text(-13, -0.95, f"citations {V1['p_cited']:.0%} -> {V2['p_cited']:.0%} of claims", fontsize=9,
                     color=MUTED))
ax.set_ylim(-1.4, len(names) - 0.5)

# ---------------------------------------------------------------- B. the threshold sweep
ax = axes[1]
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
taus = [r["tau"] for r in R["sweep"]]
ax.plot(taus, [r["passes_polarity_flip"] for r in R["sweep"]], color=BAD, lw=2.2, drawstyle="steps-post")
ax.plot(taus, [r["fails_supported_paraphrase"] for r in R["sweep"]], color=OK, lw=2.2, drawstyle="steps-post")
ax.axvline(T.TAU, color=MUTED, lw=1, ls=":")
ax.set_xlim(0.28, 1.02)
ax.set_ylim(-0.04, 1.08)
ax.set_xlabel("lexical threshold tau (coverage of claim words in the cited passage)", color=INK, fontsize=9.5)
ax.set_ylabel("rate", color=INK)
ax.set_title("B. No tau splits flip from paraphrase", loc="left", color=INK, fontsize=12.5, weight="bold")
texts.append(ax.text(0.31, 0.72, "passes a polarity flip\n('does not include' for 'includes')", fontsize=9, color=BAD))
texts.append(ax.text(0.56, 0.24, "fails a SUPPORTED paraphrase\n('comes with' for 'includes')", fontsize=9, color=OK))
texts.append(ax.text(T.TAU + 0.01, 0.52, f"tau = {T.TAU}\nused here", fontsize=8.5, color=MUTED))
o = R["order"]
texts.append(ax.text(0.60, 0.03, f"flip scores above the\nparaphrase in {o['flip_above']:.0%} of pairs", fontsize=9,
                     color=MUTED))

# ---------------------------------------------------------------- C. the mechanism, drawn
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 7.2)
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
SRC = 5  # [6] Pro plan offers refunds within 30 days for first-time purchases.
examples = [("supported, paraphrased", T.build(SRC, "none", "source", True)),
            ("true, cites the neighbour", T.build(SRC, "none", "neighbour", False)),
            ("polarity flipped", T.build(SRC, "polarity", "source", False)),
            ("scope widened", T.build(SRC, "scope", "source", False)),
            ("other plan's fact", T.build(SRC, "entity", "source", False))]
py = {5: 5.9, 6: 3.55, 7: 1.2}
for j, y in py.items():
    ax.add_patch(FancyBboxPatch((6.0, y - 0.42), 3.9, 0.84, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#eef1f5", ec=MUTED, lw=0.8))
    texts.append(ax.text(6.12, y + 0.02, f"[{j + 1}] " + "\n".join(textwrap.wrap(T.PASSAGES[j], 34)), fontsize=8.0,
                         color=INK, va="center"))
for i, (label, r) in enumerate(examples):
    y = 6.55 - i * 1.42
    bad = not r["supported"]
    ax.add_patch(FancyBboxPatch((0.05, y - 1.0), 5.0, 1.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=BG, ec=BAD if bad else OK, lw=1.5))
    texts.append(ax.text(0.17, y + 0.1, label, fontsize=9.3, color=BAD if bad else OK, weight="bold", va="center"))
    texts.append(ax.text(0.17, y - 0.38, "\n".join(textwrap.wrap(r["claim"], 56)), fontsize=8.1, color=INK,
                         va="center"))
    passed = [TINY[k] for k in T.CHECKERS if r["checks"][k]]
    texts.append(ax.text(0.17, y - 0.82, f"coverage {r['cov']:.2f}   passes: " + (", ".join(passed) if passed else "none"),
                         fontsize=8.2, color=MUTED, va="center"))
    ty = py[r["cited"]] + {0: 0.3, 2: 0.1, 3: -0.1, 4: -0.3}.get(i, 0.0)
    ax.add_patch(FancyArrowPatch((5.07, y - 0.35), (5.98, ty), arrowstyle="-|>", mutation_scale=9, color=MUTED,
                                 lw=0.9))
ax.set_title("C. Five claims, the passage each cites, and who passes it", loc="left", color=INK, fontsize=12.5,
             weight="bold")
texts.append(ax.text(0.05, -0.38, "cite = has citation, ctx = context lexical, lex = cited lexical, +num, +neg = "
                                 "with number / negation checks", fontsize=8.6, color=MUTED))

fig.suptitle("The citation does not say that.  Synthetic cited answers, every claim the generator can emit "
             "enumerated, exact rates", x=0.02, ha="left", y=0.965, color=INK, fontsize=14, weight="bold")

fig.canvas.draw()
rend = fig.canvas.get_renderer()
boxes = [t.get_window_extent(rend) for t in texts]
for i in range(len(boxes)):
    ax_box = texts[i].axes.get_window_extent(rend)
    bb = boxes[i]
    assert ax_box.x0 - 1 <= bb.x0 and bb.x1 <= ax_box.x1 + 1 and ax_box.y0 - 1 <= bb.y0 and bb.y1 <= ax_box.y1 + 1, \
        f"text {texts[i].get_text()!r} leaves its axes"
    for j in range(i + 1, len(boxes)):
        assert not bb.overlaps(boxes[j]), f"{texts[i].get_text()!r} overlaps {texts[j].get_text()!r}"

fig_box = fig.bbox
for ax in axes[:2]:
    for lab in ax.get_yticklabels() + ax.get_xticklabels():
        if lab.get_text():
            bb = lab.get_window_extent(rend)
            assert bb.x0 >= fig_box.x0 and bb.x1 <= fig_box.x1, f"tick label {lab.get_text()!r} leaves the figure"

fig.savefig("citation_audit.png", dpi=300, facecolor=BG)
fig.savefig("citation_audit.svg", facecolor=BG)
print("wrote citation_audit.png / .svg")
