"""The six-panel audit figure. Run: python3 make_chart.py -> sparkline_audit.png

Each panel is one of the six experiments, drawn at a size where the failure is visible - which
is itself the point: none of these are visible at 80x20.
"""

from __future__ import annotations

import contextlib
import io

import matplotlib

matplotlib.use("Agg")
import evidence
import matplotlib.pyplot as plt
import numpy as np
from sparkline import (
    Geometry,
    Series,
    banking_deg,
    build_path,
    resolve_domain,
)

INK = "#1f2933"
BLUE = "#1f4e79"
RED = "#c0392b"
GREY = "#9aa5b1"
AMBER = "#b7791f"
GREEN = "#2f855a"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": "#cbd2d9",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "figure.dpi": 300,
    }
)


def mini(ax, values, color=BLUE, lw=1.4, label=None):
    """Draw a series in an axes stripped down to sparkline conditions."""
    xs = np.arange(len(values))
    ax.plot(xs, values, color=color, lw=lw, solid_capstyle="round")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if label:
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7, color=GREY)


fig = plt.figure(figsize=(13.5, 8.4))
gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.30, top=0.855, bottom=0.06, left=0.07, right=0.97)

fig.suptitle(
    "Six ways a sparkline lies, and what it costs - none of them visible at 80x20 px",
    fontsize=12,
    fontweight="bold",
    y=0.975,
)

# --------------------------------------------------------------------------------------
# Panel 1: per-row autoscale collapses level
# --------------------------------------------------------------------------------------
shape = np.array([0, 3, 2, 5, 4, 7, 6, 8]) / 8.0
levels = [1.0, 512.0, 2.0 ** 21]
sub = gs[0, 0].subgridspec(3, 2, hspace=0.75, wspace=0.15)
for r, lv in enumerate(levels):
    vals = lv * (1.0 + 0.5 * shape)
    ax = fig.add_subplot(sub[r, 0])
    mini(ax, (vals - vals.min()) / (vals.max() - vals.min()), label="%.3g" % lv)
    if r == 0:
        ax.set_title("per_row (autoscale)", fontsize=7.5, color=RED, pad=3)
    ax = fig.add_subplot(sub[r, 1])
    hi = levels[-1] * 1.5
    mini(ax, vals / hi, color=GREEN)
    ax.set_ylim(-0.03, 1.03)
    if r == 0:
        ax.set_title("shared domain", fontsize=7.5, color=GREEN, pad=3)
ax_lab = fig.add_subplot(gs[0, 0], frameon=False)
ax_lab.set_xticks([])
ax_lab.set_yticks([])
ax_lab.set_title(
    "1. Three rows, one shape, 2,000,000x apart in level\n"
    "left: identical paths, byte for byte.  right: honest, and nearly unreadable",
    pad=26,
)

# --------------------------------------------------------------------------------------
# Panel 2: aspect ratio
# --------------------------------------------------------------------------------------
rng = np.random.default_rng(3)
n = 24
vals = 100 + 0.9 * np.arange(n) + rng.normal(0, 2.2, n)
s = Series("x", list(map(float, vals)))
d = resolve_domain(s, "per_row")
sub = gs[0, 1].subgridspec(4, 1, hspace=0.9)
for r, w in enumerate((15.0, 30.0, 120.0, 480.0)):
    p = build_path(s, d, Geometry(width=w, height=20.0))
    deg = banking_deg(p.points)
    ax = fig.add_subplot(sub[r, 0])
    xs = [pt[0] for pt in p.points]
    ys = [-pt[1] for pt in p.points]
    ax.plot(xs, ys, color=BLUE, lw=1.3, solid_capstyle="round")
    ax.set_xlim(0, 480)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(
        0, 2.5, "%.0f px wide - median slope %.0f°" % (w, deg),
        va="bottom", ha="left", fontsize=6.8,
        color=RED if (deg > 45 or deg < 8) else GREY,
    )
    ax.set_ylim(-22, 12)
ax_lab = fig.add_subplot(gs[0, 1], frameon=False)
ax_lab.set_xticks([])
ax_lab.set_yticks([])
ax_lab.set_title(
    "2. One series, four column widths, drawn to scale\n"
    "'volatile' and 'flat' are the same 24 numbers",
    pad=26,
)

# --------------------------------------------------------------------------------------
# Panel 3: the bridged gap
# --------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 2])
truth = np.array([120.0, 124, 130, 240, 260, 205, 150, 128, 126, 124])
xs = np.arange(len(truth))
mask = np.array([True, True, True, False, False, False, True, True, True, True])
ax.plot(xs, truth, color=GREY, lw=1.2, ls=":", label="truth (not collected)")
ax.plot(xs[:3], truth[:3], color=BLUE, lw=2.0, label="observed")
ax.plot(xs[6:], truth[6:], color=BLUE, lw=2.0)
ax.plot([xs[2], xs[6]], [truth[2], truth[6]], color=RED, lw=2.0, label="bridged (invented)")
ax.axvspan(2.05, 5.95, color="#fdecea", zorder=0)
ax.scatter(xs[mask], truth[mask], s=12, color=BLUE, zorder=5)
ax.set_title(
    "3. Bridging a gap: 44% of the ink is invented\n"
    "and misses the truth by 81% of the cell height",
    pad=8,
)
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
ax.set_xticks([])
ax.set_ylabel("p95 ms", fontsize=7)
ax.spines[["top", "right"]].set_visible(False)

# --------------------------------------------------------------------------------------
# Panel 4: trend estimator accuracy
# --------------------------------------------------------------------------------------
with contextlib.redirect_stdout(io.StringIO()):
    trend = evidence.experiment_endpoint_vs_robust(trials=4000, n=24, seed=7)
ax = fig.add_subplot(gs[1, 0])
models = ["trend+iid", "random_walk", "trend+outlier"]
xpos = np.arange(len(models))
ax.bar(xpos - 0.19, [100 * trend[m]["endpoint"] for m in models], 0.36,
       color=RED, label="sign(last - first)")
ax.bar(xpos + 0.19, [100 * trend[m]["robust"] for m in models], 0.36,
       color=GREEN, label="Theil-Sen")
ax.axhline(50, color=GREY, lw=0.8, ls="--")
ax.text(2.42, 52, "coin flip", fontsize=6.5, color=GREY, ha="right")
for i, m in enumerate(models):
    for off, key in ((-0.19, "endpoint"), (0.19, "robust")):
        v = 100 * trend[m][key]
        ax.text(i + off, v + 2, "%.0f" % v, ha="center", fontsize=6.5, color=INK)
ax.set_xticks(xpos)
ax.set_xticklabels(models, fontsize=7)
ax.set_ylim(0, 112)
ax.set_ylabel("% of 4000 trials with the\ntrue direction recovered", fontsize=7)
ax.set_title(
    "4. Which trend reading is right depends on the\nnoise model - and the picture never shows it",
    pad=8,
)
ax.legend(frameon=False, fontsize=6.5, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)

# --------------------------------------------------------------------------------------
# Panel 5: time vs index x-positions
# --------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
times = np.array([0.0, 1, 2, 3, 9, 10])
nps = np.array([31.0, 33, 34, 36, 41, 42])
ax.plot(times, nps, color=GREEN, lw=1.8, marker="o", ms=3.5, label="x from `times` (months)")
ax.plot(np.linspace(0, 10, len(nps)), nps, color=RED, lw=1.8, marker="o", ms=3.5,
        ls="--", label="x from index (equal steps)")
ax.axvspan(3.05, 8.95, color="#f7f9fb", zorder=0)
ax.text(6.0, 31.6, "6 months\nnot reported", fontsize=6.5, color=GREY, ha="center")
ax.set_title(
    "5. Irregular reporting plotted at equal steps\n"
    "index x overstates the monthly trend by 120%",
    pad=8,
)
ax.set_xlabel("month", fontsize=7)
ax.set_ylabel("NPS", fontsize=7)
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

# --------------------------------------------------------------------------------------
# Panel 6: stroke scaling under a responsive viewBox
# --------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 2])
widths = np.array([40.0, 80.0, 160.0, 320.0])
eff = 1.25 * widths / 80.0
cats = np.arange(len(widths))
ax.plot(cats, eff, color=RED, marker="o", ms=4, lw=1.6, label="stroke as rendered")
ax.axhline(1.25, color=GREEN, lw=1.6, label='vector-effect="non-scaling-stroke"')
for c, e in zip(cats, eff):
    ax.annotate("%.2f px" % e, (c, e), textcoords="offset points", xytext=(5, -2),
                fontsize=6.8, color=INK)
ax.set_xlim(-0.25, len(widths) - 0.5)
ax.set_xticks(cats)
ax.set_xticklabels(["%.0f" % w for w in widths])
ax.set_xlabel("rendered column width (px), one 80x20 viewBox", fontsize=7)
ax.set_ylabel("effective stroke (px)", fontsize=7)
ax.set_title(
    "6. The same 1.25 px line renders 0.62 to 5.00 px\npurely from the column it lands in",
    pad=8,
)
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

# --------------------------------------------------------------------------------------
# Bottom strip: the same six sample rows, wrong way and right way, at real size
# --------------------------------------------------------------------------------------
from sparkline import sample_table

rows = sample_table()
sub = gs[2, :].subgridspec(1, 2, wspace=0.16)
for col, (mode, title, color) in enumerate(
    (
        ("per_row", "per_row autoscale - what a one-liner emits", RED),
        ("indexed", "indexed to each row's first value - comparable", GREEN),
    )
):
    inner = sub[0, col].subgridspec(len(rows), 1, hspace=0.55)

    def rebased(srs):
        """Present values, divided by the row's first present value in indexed mode."""
        pv = [v for v in srs.values if v is not None]
        if mode != "indexed" or not pv or not pv[0]:
            return pv
        return [v / pv[0] for v in pv]

    # indexed is a SHARED domain over rebased values; per_row is resolved per row below.
    shared = [v for srs in rows for v in rebased(srs)] if mode == "indexed" else None

    for r, srs in enumerate(rows):
        ax = fig.add_subplot(inner[r, 0])
        vals = rebased(srs)
        idx = [i for i, v in enumerate(srs.values) if v is not None]
        lo, hi = (min(shared), max(shared)) if shared else (min(vals), max(vals))
        norm = [(v - lo) / (hi - lo) if hi > lo else 0.5 for v in vals]
        # break the line at gaps, exactly as build_path does
        runs, cur = [], []
        prev = None
        for i, v in zip(idx, norm):
            if prev is not None and i != prev + 1:
                runs.append(cur)
                cur = []
            cur.append((i, v))
            prev = i
        runs.append(cur)
        for run in runs:
            if len(run) > 1:
                ax.plot([p[0] for p in run], [p[1] for p in run], color=color, lw=1.3,
                        solid_capstyle="round")
            elif run:
                ax.plot(run[0][0], run[0][1], marker="o", ms=2.2, color=color)
        ax.set_xlim(-0.3, len(srs.values) - 0.7)
        ax.set_ylim(-0.12, 1.12)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_ylabel(srs.label, rotation=0, ha="right", va="center", fontsize=6.5, color=GREY)
        if r == 0:
            ax.set_title(title, fontsize=8, color=color, pad=4)
        # The right-hand panel prints what the picture cannot carry. That is the thesis: a
        # sparkline is only honest next to the number it is a sparkline OF.
        if mode == "indexed":
            pv = [v for v in srs.values if v is not None]
            ax.text(
                len(srs.values) - 0.4, 0.5,
                "  level %-8.4g  %+.1f%%" % (pv[0], 100 * (pv[-1] / pv[0] - 1)),
                va="center", ha="left", fontsize=6.3, color=INK,
                family="monospace",
            )

fig.text(
    0.06, 0.012,
    "Bottom: enterprise_mrr and self_serve_mrr differ by 1000x in level and are the SAME picture "
    "on both sides - correctly on the right, since their percent change is identical. The "
    "difference is that the right-hand rows carry the level and the change beside them. No "
    "scale mode makes a 60x18 px path self-sufficient; the printed number is part of the chart, "
    "not a caption on it.",
    fontsize=7, color=GREY,
)

fig.savefig("sparkline_audit.png", dpi=300, bbox_inches="tight", facecolor="white")
print("wrote sparkline_audit.png")
