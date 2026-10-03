"""The audit figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Three panels: the v1 -> v2 prompt change as each scorer reports it against the truth; each scorer's error
profile (passes a broken path vs fails a valid one); and a drawn picture of the mechanism - six real
trajectories as tool-call chains, with the scorers that pass each one. The geometry self-check runs BEFORE
savefig (Day 180).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trajectory as T
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
TRUTH, BAD, OK = "#1f2733", "#c8562b", "#2e8b57"
COL = {"outcome": "#2f6fdb", "strict": "#8a94a3", "unordered": "#8a6d3b", "superset": "#e3b23c",
       "outcome+superset": "#7a4fb5"}
TOOL_COL = {"search_kb": "#eef1f5", "lookup_order": "#dfe7f1", "check_policy": "#dfe7f1", "issue_refund": "#f6e3c3"}
SHORT = {"search_kb": "kb", "lookup_order": "lookup", "check_policy": "policy", "issue_refund": "refund"}

R = json.load(open("results.json"))
A = R["agents"]
V1, V2 = A["v1 careful"], A["v2 fewer calls"]

fig = plt.figure(figsize=(18, 7.2), facecolor=BG)
axes = [fig.add_axes([0.09, 0.14, 0.22, 0.66]), fig.add_axes([0.36, 0.14, 0.23, 0.66]),
        fig.add_axes([0.63, 0.06, 0.36, 0.76])]
for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
texts = []

# ---------------------------------------------------------------- A. the prompt change, as each scorer sees it
ax = axes[0]
ax.grid(True, color=GRID, lw=0.7, axis="x")
ax.set_axisbelow(True)
names = ["truly valid"] + list(T.SCORERS)
vals = [R["deltas"][k] * 100 for k in names]
ys = list(range(len(names)))[::-1]
ax.barh(ys, vals, 0.62, color=[TRUTH] + [COL[k] for k in T.SCORERS])
for y, v in zip(ys, vals):
    texts.append(ax.text(v - 0.8, y, f"{v:+.1f}", ha="right", va="center", fontsize=9.5, color=INK))
ax.set_yticks(ys)
ax.set_yticklabels(names, color=INK, fontsize=10)
ax.set_xlim(-42, 2)
ax.set_xlabel("change in reported pass rate, v1 -> v2 (points)", color=INK)
ax.set_title("A. A prompt change cost 31 points", loc="left", color=INK, fontsize=12.5, weight="bold")
texts.append(ax.text(-41, -0.95, f"tool calls {V1['calls']:.2f} -> {V2['calls']:.2f} per run", fontsize=9,
                     color=MUTED))
ax.set_ylim(-1.4, len(names) - 0.5)

# ---------------------------------------------------------------- B. error profiles
ax = axes[1]
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
offs = {"outcome": (0.025, 0.03), "strict": (0.02, 0.03), "unordered": (0.025, 0.0), "superset": (0.025, 0.0),
        "outcome+superset": (0.025, -0.045)}
for k in T.SCORERS:
    s1, s2 = V1["scorers"][k], V2["scorers"][k]
    ax.annotate("", xy=(s2["false_fail"], s2["false_pass"]), xytext=(s1["false_fail"], s1["false_pass"]),
                arrowprops=dict(arrowstyle="->", color=COL[k], lw=1.3))
    ax.plot(s1["false_fail"], s1["false_pass"], "o", color=COL[k], ms=9)
    dx, dy = offs[k]
    texts.append(ax.text(s1["false_fail"] + dx, s1["false_pass"] + dy, k, fontsize=9.5, color=COL[k],
                         weight="bold", ha="right" if dx < 0 else "left"))
ax.set_xlim(-0.05, 0.72)
ax.set_ylim(-0.05, 0.72)
ax.set_xlabel("fails a valid run  (P fail | valid)", color=INK)
ax.set_ylabel("passes a broken run  (P pass | broken)", color=INK)
ax.set_title("B. No name matcher gets both low", loc="left", color=INK, fontsize=12.5, weight="bold")
texts.append(ax.text(0.30, 0.62, "dot = v1, arrow head = v2", fontsize=9, color=MUTED))
texts.append(ax.text(0.03, -0.03, "the spec (invariants) sits here", fontsize=9, color=OK))

# ---------------------------------------------------------------- C. the mechanism, drawn
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(-0.4, 7.0)
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
E, OID, W = T.ORDER, T.ORDER, T.WRONG
paths = [
    ("reference", [("lookup_order", OID), ("check_policy", None), ("issue_refund", OID)], True),
    ("valid, other order", [("check_policy", None), ("lookup_order", OID), ("issue_refund", OID)], True),
    ("valid, extra search", [("search_kb", None), ("lookup_order", OID), ("check_policy", None), ("issue_refund", OID)],
     True),
    ("refund before reading", [("issue_refund", W), ("lookup_order", OID), ("check_policy", None)], True),
    ("refunded twice", [("lookup_order", OID), ("check_policy", None), ("issue_refund", OID), ("issue_refund", OID)], True),
    ("refund to wrong order", [("lookup_order", OID), ("check_policy", None), ("issue_refund", W)], True),
]
for i, (label, steps, correct) in enumerate(paths):
    y = 6.2 - i * 1.12
    v = T.violations(steps, True)
    passed = [k for k, ok in T.score(steps, True, correct).items() if ok]
    texts.append(ax.text(0.05, y + 0.36, label, fontsize=9.5, color=BAD if v else OK, weight="bold", va="center"))
    x = 0.05
    for j, (tool, arg) in enumerate(steps):
        w = 1.05
        bad_step = tool == "issue_refund" and (arg == W or j == 0 or steps[:j].count(("issue_refund", arg)))
        ax.add_patch(FancyBboxPatch((x, y - 0.24), w, 0.42, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=TOOL_COL[tool], ec=BAD if bad_step else MUTED, lw=1.6 if bad_step else 0.8))
        texts.append(ax.text(x + w / 2, y - 0.03, SHORT[tool] + (f" {arg}" if tool == "issue_refund" else ""),
                             ha="center", va="center", fontsize=8.2, color=INK))
        if j:
            ax.add_patch(FancyArrowPatch((x - 0.17, y - 0.03), (x - 0.02, y - 0.03), arrowstyle="-|>",
                                         mutation_scale=8, color=MUTED, lw=0.9))
        x += w + 0.2
    verdict = ("passed by all five" if len(passed) == len(T.SCORERS) else
               "passed by: " + ", ".join(passed) if passed else "passed by none")
    texts.append(ax.text(3.0, y + 0.36, verdict, fontsize=8.6, va="center", color=MUTED))
ax.set_title("C. Six paths with the right answer, and who passes them", loc="left", color=INK, fontsize=12.5,
             weight="bold")
texts.append(ax.text(0.05, -0.25, f"task: eligible refund on order {E}; red border = the broken step", fontsize=9,
                     color=MUTED))

fig.suptitle("The answer was right and the path was not.  Synthetic refund agent, every trajectory enumerated, "
             "exact probabilities", x=0.02, ha="left", y=0.965, color=INK, fontsize=14, weight="bold")

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

fig.savefig("trajectory_audit.png", dpi=300, facecolor=BG)
fig.savefig("trajectory_audit.svg", facecolor=BG)
print("wrote trajectory_audit.png / .svg")
