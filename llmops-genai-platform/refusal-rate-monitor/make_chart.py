"""The audit figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Three panels: the three versions on the true rate and on what the keyword dashboard reads (the upgrade's sign
flips); the 26-week monitor, where the aggregate alarm fires on a mix change and celebrates a drop on the
upgrade while the refusal audit does the opposite; and a drawn picture of the mechanism - traffic, the two
errors, the phrasing, the detector, the dashboard. Geometry self-check runs BEFORE savefig (Day 180), tick
labels (Day 189), all three title artists (Day 191), and plotted lines sampled against text boxes (Day 192).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
BAD, OK, DASH, HOT, BLUE, PURPLE = "#c8562b", "#2e8b57", "#8a94a3", "#e3b23c", "#2f6fdb", "#7a4fb5"

R = json.load(open("results.json"))
V = R["versions"]
KW = "keyword list (written on v1)"

fig = plt.figure(figsize=(18, 7.4), facecolor=BG)
axes = [fig.add_axes([0.125, 0.14, 0.2, 0.66]), fig.add_axes([0.395, 0.14, 0.215, 0.66]),
        fig.add_axes([0.645, 0.05, 0.345, 0.75])]
for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
texts = []

# ---------------------------------------------------------------- A. true vs measured, three versions
ax = axes[0]
ax.grid(True, color=GRID, lw=0.7, axis="x")
ax.set_axisbelow(True)
names = list(V)
ys = [2.0, 1.0, 0.0]
cols = {names[0]: DASH, names[1]: BLUE, names[2]: PURPLE}
for y, n in zip(ys, names):
    e = V[n]
    ax.barh(y + 0.19, e["refusal_rate"] * 100, 0.34, color=cols[n])
    ax.barh(y - 0.19, e["measured"][KW] * 100, 0.34, color=cols[n], alpha=0.35, hatch="//", edgecolor=cols[n])
    texts.append(ax.text(e["refusal_rate"] * 100 + 0.25, y + 0.19, f"true {e['refusal_rate']:.1%}", va="center", fontsize=9.2,
                         color=INK, weight="bold"))
    texts.append(ax.text(e["measured"][KW] * 100 + 0.25, y - 0.19, f"dashboard reads {e['measured'][KW]:.1%}  (recall {e['recall'][KW]:.2f})",
                         va="center", fontsize=8.6, color=MUTED))
ax.set_yticks(ys)
ax.set_yticklabels([f"{n}\nover {V[n]['over_refusal']:.1%} / under {V[n]['under_refusal']:.0%}" for n in names], color=INK, fontsize=9)
ax.set_xlim(0, 17.5)
ax.set_ylim(-0.75, 2.85)
ax.set_xlabel("refusal rate (%)", color=INK)
ax.set_title("A. Upgrade: true +39%, dashboard -63%", loc="left", color=INK, fontsize=11, weight="bold")
texts.append(ax.text(0.3, 2.62, "solid: true rate   hatched: keyword detector's reading", fontsize=8.4, color=MUTED))
d = R["deltas"]
texts.append(ax.text(0.3, -0.6, f"upgrade: true {d['upgrade: true refusal rate'] * 100:+.1f} pts, measured {d['upgrade: measured (keyword)'] * 100:+.1f} pts",
                     fontsize=9, color=BAD))

# ---------------------------------------------------------------- B. the monitor
ax = axes[1]
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
mon = R["monitor"]
wk = [r["week"] for r in mon]
cw = R["config"]["campaign_weeks"]
ax.axvspan(min(cw) - 0.5, max(cw) + 0.5, color=HOT, alpha=0.18, lw=0)
ax.axvspan(R["config"]["upgrade_week"] - 0.5, max(wk) + 0.5, color=BLUE, alpha=0.10, lw=0)
ax.plot(wk, [r["true_rate"] * 100 for r in mon], color=INK, lw=2.2, label="true refusal rate")
ax.plot(wk, [r["measured"] * 100 for r in mon], color=DASH, lw=2.2, ls="--", label="keyword dashboard")
ax.plot(wk, [r["over_refusal"] * 100 for r in mon], color=BAD, lw=2.0, label="over-refusal")
ax.set_xlim(0.5, 26.5)
ax.set_ylim(0, 15.5)
ax.set_xlabel("week", color=INK)
ax.set_ylabel("share of responses (%)", color=INK, fontsize=9.5)
ax.set_title("B. Alarms on the mix, cheers the upgrade", loc="left", color=INK, fontsize=11, weight="bold")
texts.append(ax.text(1.0, 14.6, "campaign weeks 9-11 (model unchanged):", fontsize=8.4, color="#8a6d3b"))
texts.append(ax.text(1.0, 13.7, f"aggregate alarm fires P={R['monitor_summary']['campaign_alarm_high']:.2f}, "
                               f"audit P={R['monitor_summary']['campaign_audit_alarm']:.2f}", fontsize=8.4, color="#8a6d3b"))
texts.append(ax.text(12.3, 11.3, "upgrade at week 19: true rises,", fontsize=8.4, color=BLUE))
texts.append(ax.text(12.3, 10.4, f"chart flags a DROP P={R['monitor_summary']['upgrade_flag_low']:.2f},", fontsize=8.4, color=BLUE))
texts.append(ax.text(12.3, 9.5, f"refusal audit alarms P={R['monitor_summary']['upgrade_audit_alarm']:.2f}", fontsize=8.4, color=BLUE))
texts.append(ax.text(19.6, 7.9, "true", fontsize=8.6, color=INK, weight="bold"))
texts.append(ax.text(19.6, 1.2, "dashboard", fontsize=8.6, color=MUTED, weight="bold"))
texts.append(ax.text(19.6, 5.4, "over-refusal", fontsize=8.6, color=BAD, weight="bold"))
texts.append(ax.text(1.0, 7.4, "true", fontsize=8.6, color=INK))
texts.append(ax.text(1.0, 4.7, "dashboard", fontsize=8.6, color=MUTED))
texts.append(ax.text(1.0, 0.9, "over-refusal", fontsize=8.6, color=BAD))

# ---------------------------------------------------------------- C. the mechanism, drawn
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 7.2)
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
v1, v2 = V["v1 baseline"], V["v2 upgraded model"]
cls = R["config"]["classes"]


def box(x, y, w, h, fc, ec, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=lw))


def arrow(a, b, col, ls="-", lw=1.3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=col, lw=lw, ls=ls))


# traffic
box(0.2, 4.6, 2.6, 2.1, "#eef1f5", MUTED)
texts.append(ax.text(0.4, 6.4, "TRAFFIC", fontsize=9.5, color=MUTED, weight="bold"))
for i, c in enumerate(cls):
    col = BAD if not c["legit"] else INK
    texts.append(ax.text(0.4, 5.85 - i * 0.42, f"{c['share']:.0%} {c['name']}", fontsize=8.6, color=col))
# model
box(3.5, 4.6, 2.9, 2.1, BG, INK, 1.1)
texts.append(ax.text(3.7, 6.4, "THE MODEL refuses...", fontsize=9.5, color=INK, weight="bold"))
texts.append(ax.text(3.7, 5.85, f"{v1['per_class']['benign']:.1%} of benign", fontsize=8.6, color=INK))
texts.append(ax.text(3.7, 5.43, f"{v1['per_class']['sensitive']:.0%} of sensitive", fontsize=8.6, color=INK))
texts.append(ax.text(3.7, 5.01, f"{v1['per_class']['harmful']:.0%} of harmful", fontsize=8.6, color=INK))
arrow((2.85, 5.65), (3.45, 5.65), INK)
# the two errors
box(6.9, 5.75, 3.0, 0.95, "#fbeee9", BAD, 1.2)
texts.append(ax.text(7.05, 6.38, "OVER-REFUSAL", fontsize=8.8, color=BAD, weight="bold"))
texts.append(ax.text(7.05, 5.98, f"P(refuse | legit) = {v1['over_refusal']:.1%}", fontsize=8.4, color=INK))
box(6.9, 4.6, 3.0, 0.95, "#fbeee9", BAD, 1.2)
texts.append(ax.text(7.05, 5.23, "UNDER-REFUSAL", fontsize=8.8, color=BAD, weight="bold"))
texts.append(ax.text(7.05, 4.83, f"P(comply | harmful) = {v1['under_refusal']:.0%}", fontsize=8.4, color=INK))
arrow((6.45, 6.0), (6.85, 6.2), BAD)
arrow((6.45, 5.3), (6.85, 5.1), BAD)
# phrasing - full width
box(0.2, 2.8, 9.7, 1.5, BG, HOT, 1.2)
texts.append(ax.text(0.4, 4.02, "...IN WORDS THAT CHANGE WITH THE VERSION", fontsize=9.2, color="#8a6d3b", weight="bold"))
ph1 = R["config"]["versions"]["v1 baseline"]["phrasing"]
ph2 = R["config"]["versions"]["v2 upgraded model"]["phrasing"]
texts.append(ax.text(0.4, 3.64, "v1:  " + "   ".join(f"'{p}' {q:.0%}" for p, q in ph1.items() if q >= 0.1), fontsize=8.4, color=INK))
texts.append(ax.text(0.4, 3.3, "v2:  " + "   ".join(f"'{p}' {q:.0%}" for p, q in ph2.items() if q >= 0.1), fontsize=8.4, color=INK))
texts.append(ax.text(0.4, 2.94, "detector matches 'I cannot' and 'I can't'  ->  recall "
                               f"{v1['recall'][KW]:.2f} on v1, {v2['recall'][KW]:.2f} on v2", fontsize=8.6, color=BAD, weight="bold"))
arrow((4.95, 4.55), (4.95, 4.35), INK)
# dashboard - full width
box(0.2, 1.7, 9.7, 0.9, "#eef1f5", MUTED)
texts.append(ax.text(0.4, 2.3, "THE DASHBOARD", fontsize=9.5, color=MUTED, weight="bold"))
texts.append(ax.text(2.9, 2.3, f"v1 reads {v1['measured'][KW]:.1%}  (true {v1['refusal_rate']:.1%})", fontsize=8.8, color=INK))
texts.append(ax.text(6.3, 2.3, f"v2 reads {v2['measured'][KW]:.1%}  (true {v2['refusal_rate']:.1%})", fontsize=8.8, color=INK))
texts.append(ax.text(0.4, 1.88, '"upgrade cut refusals by two thirds" - phrasing changed, over-refusal doubled',
                     fontsize=8.6, color=BAD, weight="bold"))
arrow((4.95, 2.75), (4.95, 2.65), INK)
# bottom lines
tw = R["twin"]
camp = R["campaign"]
texts.append(ax.text(0.2, 1.3, f"mix: harmful {R['config']['classes'][2]['share']:.0%} -> {camp['mix']['harmful']:.0%} moves the rate "
                               f"{v1['refusal_rate']:.1%} -> {camp['refusal_rate']:.1%} with both errors unchanged", fontsize=8.8, color=INK))
texts.append(ax.text(0.2, 0.92, f"flat rate at 2x over-refusal forces under-refusal {v1['under_refusal']:.0%} -> {tw['under_refusal']:.0%}: "
                               "a flat aggregate is never neutral", fontsize=8.8, color=INK))
aud = R["audit"]
texts.append(ax.text(0.2, 0.54, f"{aud['refusals'][1]['n']} labels on detected refusals: power {aud['refusals'][1]['power']:.2f};  "
                               f"the same {aud['traffic'][1]['n']} on traffic: {aud['traffic'][1]['power']:.2f}", fontsize=8.8, color=INK))
texts.append(ax.text(0.2, 0.15, "two errors, a mix, a detector: one number; three move without the model",
                     fontsize=8.8, color=BAD, weight="bold"))
ax.set_title("C. What the refusal rate is made of", loc="left", color=INK, fontsize=11, weight="bold")

fig.suptitle("It started refusing valid requests.  A refusal rate is two errors, a traffic mix and a detector, "
             "and the dashboard sees one number", x=0.02, ha="left", y=0.965, color=INK, fontsize=12.5, weight="bold")

fig.canvas.draw()
rend = fig.canvas.get_renderer()
titles = [t for ax in axes for t in (ax.title, ax._left_title, ax._right_title) if t.get_text()]
assert len(titles) == len(axes), "a panel title was not found by the geometry check"
boxes = [t.get_window_extent(rend) for t in texts]
tboxes = [t.get_window_extent(rend) for t in titles]
for i in range(len(tboxes)):
    for j in range(i + 1, len(tboxes)):
        assert not tboxes[i].overlaps(tboxes[j]), f"title {titles[i].get_text()!r} overlaps {titles[j].get_text()!r}"
    for j, bb in enumerate(boxes):
        assert not tboxes[i].overlaps(bb), f"title {titles[i].get_text()!r} overlaps {texts[j].get_text()!r}"
problems = []
for i in range(len(boxes)):
    ax_box = texts[i].axes.get_window_extent(rend)
    bb = boxes[i]
    if not (ax_box.x0 - 1 <= bb.x0 and bb.x1 <= ax_box.x1 + 1 and ax_box.y0 - 1 <= bb.y0 and bb.y1 <= ax_box.y1 + 1):
        inv = texts[i].axes.transData.inverted()
        (xa, ya), (xb, yb) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        problems.append(f"text {texts[i].get_text()!r} leaves its axes: data x {xa:.2f}..{xb:.2f} y {ya:.2f}..{yb:.2f}")
    for j in range(i + 1, len(boxes)):
        if bb.overlaps(boxes[j]):
            problems.append(f"{texts[i].get_text()!r} overlaps {texts[j].get_text()!r}")
assert not problems, "\n".join(problems)
fig_box = fig.bbox
for ax in axes[:2]:
    for lab in ax.get_yticklabels() + ax.get_xticklabels():
        if lab.get_text():
            bb = lab.get_window_extent(rend)
            assert bb.x0 >= fig_box.x0 and bb.x1 <= fig_box.x1, f"tick label {lab.get_text()!r} leaves the figure"
            assert bb.y0 >= fig_box.y0 and bb.y1 <= fig_box.y1, f"tick label {lab.get_text()!r} leaves the figure"
sup = fig._suptitle.get_window_extent(rend)
assert sup.x1 <= fig_box.x1, "suptitle leaves the figure"
for bb in tboxes:
    assert not sup.overlaps(bb), "suptitle overlaps a panel title"
for ax in axes[:2]:
    for line in ax.get_lines():
        xy = ax.transData.transform(np.column_stack(line.get_data()))
        for t, bb in zip(texts, boxes):
            if t.axes is ax and any(bb.contains(x, y) for x, y in xy):
                raise AssertionError(f"text {t.get_text()!r} sits on a plotted line")

fig.savefig("refusal_audit.png", dpi=300, facecolor=BG)
fig.savefig("refusal_audit.svg", facecolor=BG)
print("wrote refusal_audit.png / .svg")
