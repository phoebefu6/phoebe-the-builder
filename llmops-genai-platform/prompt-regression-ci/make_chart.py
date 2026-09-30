"""The audit figure. Every value read from results.json, so the chart cannot drift from evidence.txt.

Three panels: how many phantom "broken" cases a no-op PR produces under three rules (exact pmf) with
each rule's calibrated threshold; P(red) of each rule under each kind of real break; and a drawn picture
of the mechanism - the held-out set as 300 cells, with one simulated no-op diff ringed. The geometry
self-check runs BEFORE savefig (Day 180).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import gate as G
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
NAIVE, FISH, QUAR, PASS, FAIL, FLAKY, RING = "#c8562b", "#2f6fdb", "#2e8b57", "#dfe7f1", "#56657a", "#e3b23c", "#c8562b"

R = json.load(open("results.json"))
C = R["config"]
EV = {r["rule"]: r for r in R["rules"]}
pi_a = G.held_out_set()

fig = plt.figure(figsize=(18, 6.8), facecolor=BG)
axes = [fig.add_axes([0.05, 0.14, 0.27, 0.69]), fig.add_axes([0.38, 0.14, 0.29, 0.69]),
        fig.add_axes([0.72, 0.14, 0.26, 0.69])]
for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
texts = []

# ---------------------------------------------------------------- A. phantom breaks on a no-op
ax = axes[0]
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
show = (("naive k=1", NAIVE), ("quarantine + majority k=3", QUAR), ("fisher k=5", FISH))
for name, col in show:
    label, kind, k, quar = next(x for x in G.RULES if x[0] == name)
    pmf = G.count_pmf(G.flag_prob(pi_a, pi_a, kind, k, quar))[:21]
    ax.plot(np.arange(len(pmf)), pmf, color=col, lw=2.2, marker="o", ms=3.5, label=f"{name}  (T = {EV[name]['T']})")
    ax.axvline(EV[name]["T"] - 0.5, color=col, lw=1, ls=":")
ax.set_xlim(-0.5, 20.5)
ax.set_xlabel("cases listed as 'broken' on a PR that changed nothing", color=INK)
ax.set_ylabel("probability (exact)", color=INK)
nv = EV["naive k=1"]
ax.set_title(f"A. No-op PR: naive diff lists {nv['noop_expected_flags']:.1f} breaks on average", loc="left",
             color=INK, fontsize=12.5, weight="bold")
texts.append(ax.text(10.2, 0.30, f"zero tolerance (T = 1):\nnaive red {nv['noop_red_at_T1']:.0%} of the time\n"
                     f"fisher k=5 red {EV['fisher k=5']['noop_red_at_T1']:.0%}", color=MUTED, fontsize=9.5))
ax.legend(loc="upper right", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9)

# ---------------------------------------------------------------- B. power by scenario
ax = axes[1]
ax.grid(True, color=GRID, lw=0.7, axis="y")
ax.set_axisbelow(True)
rules = ["naive k=1", "majority k=5", "fisher k=5", "quarantine + majority k=3"]
cols = [NAIVE, "#8a6d3b", FISH, QUAR]
scs = G.SCENARIOS[1:]
w = 0.2
for j, (rname, col) in enumerate(zip(rules, cols)):
    vals = [EV[rname][sc]["red_at_T"] for sc in scs]
    xs = np.arange(len(scs)) + (j - 1.5) * w
    ax.bar(xs, vals, w * 0.92, color=col, label=f"{rname}  ({EV[rname]['calls_per_pr']:,} calls)")
    for x, v in zip(xs, vals):
        texts.append(ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=8, color=INK))
ax.set_xticks(np.arange(len(scs)))
ax.set_xticklabels([f"{s}\n({C['n_break']} cases)" for s in scs], color=INK)
ax.set_ylim(0, 1.25)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylabel("P(gate red) at the calibrated threshold", color=INK)
ax.set_title("B. No rule catches all three kinds of break", loc="left", color=INK, fontsize=12.5, weight="bold")
ax.legend(loc="upper right", ncol=2, frameon=True, facecolor=BG, edgecolor=GRID, fontsize=8.5)

# ---------------------------------------------------------------- C. the mechanism, drawn
ax = axes[2]
target = int(round(EV["naive k=1"]["noop_expected_flags"]))
for seed in range(1000):  # the first draw whose phantom count is the TYPICAL one, not a flattering one
    rng = np.random.default_rng(seed)
    a = rng.random(G.N) < pi_a
    b = rng.random(G.N) < pi_a
    flag = a & ~b
    if flag.sum() == target:
        break
cols_ = np.where(np.arange(G.N) < C["stable_pass"], PASS, np.where(np.arange(G.N) < C["stable_pass"] + C["stable_fail"], FAIL, FLAKY))
nc = 20
for i in range(G.N):
    r, c = divmod(i, nc)
    ax.add_patch(plt.Rectangle((c, -r), 0.86, 0.86, color=cols_[i], lw=0))
    if flag[i]:
        ax.add_patch(plt.Rectangle((c - 0.08, -r - 0.08), 1.02, 1.02, fill=False, ec=RING, lw=2))
ax.set_xlim(-0.5, nc + 0.3)
ax.set_ylim(-G.N // nc - 3.2, 1.4)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")
texts.append(ax.text(0, -G.N // nc - 0.6, f"pale: stable pass ({C['stable_pass']})   dark: stable fail ({C['stable_fail']})\n"
                     f"gold: flaky ({C['flaky']})   ringed: 'broken' in this no-op diff ({int(flag.sum())})",
                     color=INK, fontsize=9, va="top"))
ax.set_title("C. The same prompt, run twice, diffed", loc="left", color=INK, fontsize=12.5, weight="bold")

fig.suptitle("A prompt CI gate is a test with a false-alarm rate.  Synthetic 300-case held-out set, exact "
             "Poisson-binomial probabilities, false-red target 5%", x=0.05, ha="left", y=0.965, color=INK,
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

fig.savefig("gate_audit.png", dpi=300, facecolor=BG)
fig.savefig("gate_audit.svg", facecolor=BG)
print(f"wrote gate_audit.png / .svg  (panel C draw: {int(flag.sum())} phantom breaks)")
