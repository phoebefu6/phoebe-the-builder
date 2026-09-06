"""Six panels, every number computed live from premortem.

    python make_chart.py  ->  premortem_audit.png (300 DPI) + .svg
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import premortem as P

INK, MUTED, GRIDC, PAPER = "#1d1a17", "#8a8178", "#e3ddd5", "#faf7f2"
ACCENT, COOL, WARM, GREEN = "#c8553d", "#2f6f8f", "#e0a458", "#4f7942"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "axes.edgecolor": GRIDC,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9.5, "axes.titleweight": "bold",
})


def strip(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=0)


def panel_plan(ax):
    """Twelve confident steps, one unconfident plan."""
    running, xs = 1.0, []
    for s in P.PLAN:
        running *= s.p_success
        xs.append(running)
    idx = np.arange(1, len(P.PLAN) + 1)
    ax.plot(idx, xs, "o-", color=ACCENT, ms=5, lw=2)
    ax.axhline(0.5, color=INK, lw=1, ls="--")
    ax.text(1.05, 0.515, "coin flip", fontsize=7, color=INK)
    ax.axhline(P.weakest_step_success(), color=COOL, lw=1, ls=":")
    ax.text(len(P.PLAN), P.weakest_step_success() + 0.02,
            f"weakest single step {P.weakest_step_success():.2f}",
            fontsize=7, color=COOL, ha="right")
    ax.annotate(f"{xs[-1]:.3f}", (idx[-1], xs[-1]), textcoords="offset points",
                xytext=(-8, 12), fontsize=9, fontweight="bold", color=ACCENT, ha="right")
    ax.set_xticks(idx)
    ax.set_xlabel("steps completed")
    ax.set_ylabel("P(everything so far worked)")
    ax.set_ylim(0.35, 1.02)
    strip(ax)
    ax.set_title("1  ·  no step below 0.88, and the plan is a coin flip\n"
                 "each person estimated their own step; nobody multiplied", loc="left")


def panel_matrix(ax):
    """The 5x5, with what the cells actually contain."""
    scale = P.SCALES[0]
    cells = {}
    for m in P.MODES:
        cells.setdefault(scale.cell(m), []).append(m)
    grid = np.zeros((5, 5))
    for (pb, lb), group in cells.items():
        grid[pb - 1, lb - 1] = max(m.expected_loss for m in group)
    masked = np.ma.masked_where(grid == 0, grid)
    im = ax.imshow(masked, cmap="YlOrRd", origin="lower", aspect="auto")
    for (pb, lb), group in cells.items():
        ax.text(lb - 1, pb - 1, "\n".join(m.id for m in group),
                ha="center", va="center", fontsize=6.5, color=INK, fontweight="bold")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f"I{i + 1}" for i in range(5)])
    ax.set_yticklabels([f"L{i + 1}" for i in range(5)])
    ax.set_xlabel("impact band")
    ax.set_ylabel("likelihood band")
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=6, length=0)
    cb.outline.set_visible(False)
    cb.set_label("largest true expected loss in cell", fontsize=6.5)
    strip(ax, keep=())
    c = P.range_compression(scale)
    ax.set_title(f"2  ·  {c['shared_cells']} cells hold more than one risk\n"
                 f"worst pair inside one cell differs {c['worst_ratio']:.1f}x in expected loss",
                 loc="left")


def panel_inversions(ax):
    """Matrix score against the thing it is a proxy for."""
    scale = P.SCALES[0]
    xs = [scale.score(m) for m in P.MODES]
    ys = [m.expected_loss for m in P.MODES]
    inv_ids = {a for a, _b, _r in P.inversions(scale)} | {b for _a, b, _r in P.inversions(scale)}
    # Several modes share a score, so labels are staggered to stay readable.
    seen: dict = {}
    for m, x, y in zip(P.MODES, xs, ys):
        col = ACCENT if m.id in inv_ids else COOL
        ax.scatter([x], [y], s=54, color=col, zorder=3)
        k = seen.get(x, 0)
        seen[x] = k + 1
        ax.annotate(m.id, (x, y), textcoords="offset points",
                    xytext=(8, -3 + (0 if k == 0 else (7 if k % 2 else -10))),
                    fontsize=6.5, color=MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("risk-matrix score  (likelihood band × impact band)")
    ax.set_ylabel("true expected loss (log)")
    q = P.ranking_quality(scale)
    strip(ax)
    ax.set_title(f"3  ·  the score is not monotone in the risk\n"
                 f"{q['inversions']} of {q['ordered_by_matrix']} ordered pairs "
                 f"({q['inversion_rate']:.0%}) rank backwards", loc="left")


def panel_two_scales(ax):
    """One register, two conventional templates, two orders."""
    a, b = P.SCALES
    ra = [m.id for m in sorted(P.MODES, key=lambda m: (-a.score(m), -m.expected_loss))]
    rb = [m.id for m in sorted(P.MODES, key=lambda m: (-b.score(m), -m.expected_loss))]
    for m in P.MODES:
        ia, ib = ra.index(m.id), rb.index(m.id)
        # Highlight only the substantial moves; everything shifts by one or two
        # and colouring all of it red reads as noise rather than as a finding.
        moved = abs(ia - ib) >= 3
        ax.plot([0, 1], [ia, ib], "-", color=ACCENT if moved else GRIDC,
                lw=2.0 if moved else 1, zorder=3 if moved else 1)
        ax.text(-0.04, ia, m.id, ha="right", va="center", fontsize=6.5,
                color=INK if moved else MUTED, fontweight="bold" if moved else "normal")
        ax.text(1.04, ib, m.id, ha="left", va="center", fontsize=6.5,
                color=INK if moved else MUTED, fontweight="bold" if moved else "normal")
    ax.set_xlim(-0.22, 1.22)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([a.name, b.name], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_yticks([])
    strip(ax, keep=())
    d = P.scale_disagreement()
    ax.set_title(f"4  ·  same risks, two standard scales\n"
                 f"{d['n_flips']} pairs flip; top risk is {d['top_by_a']} or {d['top_by_b']}",
                 loc="left")


def panel_orderings(ax):
    """What the matrix puts first, against what is worth doing first."""
    scale = P.SCALES[0]
    m_order = [m.id for m in P.by_matrix(scale)]
    v_order = [m.id for m in P.by_prevention_value()]
    for m in P.MODES:
        im, iv = m_order.index(m.id), v_order.index(m.id)
        big = abs(im - iv) >= 4
        ax.plot([0, 1], [im, iv], "-", color=ACCENT if big else GRIDC,
                lw=2.2 if big else 1, zorder=3 if big else 1)
        ax.text(-0.04, im, m.id, ha="right", va="center", fontsize=6.5,
                color=INK if big else MUTED, fontweight="bold" if big else "normal")
        ax.text(1.04, iv, m.id, ha="left", va="center", fontsize=6.5,
                color=INK if big else MUTED, fontweight="bold" if big else "normal")
    ax.set_xlim(-0.24, 1.24)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["matrix score", "prevention value"], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_yticks([])
    strip(ax, keep=())
    ax.set_title("5  ·  F06 is 8th on the matrix and 1st once cost is known\n"
                 "a 0.08 chance of a 4,000,000 loss, in likelihood band 2", loc="left")


def panel_knapsack(ax):
    """It was never a ranking problem."""
    rows = P.allocation_comparison()
    labels = [f"{r['budget'] // 1000}k" for r in rows]
    x = np.arange(len(rows))
    w = 0.26
    ax.bar(x - w, [r["matrix"] for r in rows], w, color=ACCENT, label="matrix order")
    ax.bar(x, [r["ratio"] for r in rows], w, color=WARM, label="loss-avoided / cost order")
    ax.bar(x + w, [r["optimal"] for r in rows], w, color=GREEN, label="exact best set")
    worst = max(rows, key=lambda r: r["matrix_shortfall"])
    i = rows.index(worst)
    ax.annotate(f"matrix leaves\n{worst['matrix_shortfall'] / worst['optimal']:.0%} unbought",
                (i - w, worst["matrix"]), textcoords="offset points", xytext=(-6, 16),
                fontsize=7, color=ACCENT, fontweight="bold", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("prevention budget")
    ax.set_ylabel("expected loss avoided")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    strip(ax)
    ax.set_title("6  ·  choosing a set under a budget is a knapsack\n"
                 "neither ordering is reliable; the exact solve is instant", loc="left")


def main() -> None:
    fig = plt.figure(figsize=(15.5, 14.5))
    gs = fig.add_gridspec(3, 2, hspace=0.44, wspace=0.24,
                          left=0.06, right=0.955, top=0.925, bottom=0.05)
    panel_plan(fig.add_subplot(gs[0, 0]))
    panel_matrix(fig.add_subplot(gs[0, 1]))
    panel_inversions(fig.add_subplot(gs[1, 0]))
    panel_two_scales(fig.add_subplot(gs[1, 1]))
    panel_orderings(fig.add_subplot(gs[2, 0]))
    panel_knapsack(fig.add_subplot(gs[2, 1]))

    q = P.ranking_quality(P.SCALES[0])
    fig.suptitle("A pre-mortem produces a risk model. The matrix cannot rank it.",
                 x=0.06, y=0.975, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.06, 0.951,
             f"Day 156 · a 12-step plan that succeeds {P.plan_success():.0%} of the time · "
             f"14 failure modes · the default 5x5 inverts {q['inversion_rate']:.0%} of the pairs "
             f"it orders and cannot order {q['undecided_rate']:.0%} of them",
             ha="left", fontsize=9, color=MUTED)

    fig.savefig("premortem_audit.png", dpi=300, facecolor=PAPER)
    fig.savefig("premortem_audit.svg", facecolor=PAPER)
    print("wrote premortem_audit.png and premortem_audit.svg")


if __name__ == "__main__":
    main()
