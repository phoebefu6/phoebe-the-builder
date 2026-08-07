"""Generate demo.ipynb for sparkline-gen. Run once, then nbconvert --execute.

The notebook is self-contained: the setup cell writes `sparkline.py` and `evidence.py` to disk
from embedded source, so Colab and Binder get the same modules the repo has without a clone
step, and there is no second copy of the logic to drift out of sync.
"""

from __future__ import annotations

import json
import pathlib

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def md(src: str) -> None:
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src: str) -> None:
    nb["cells"].append(
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}
    )


BASE = "analytics-engineering-bi/sparkline-gen"

SPARKLINE = pathlib.Path("sparkline.py").read_text()
EVIDENCE = pathlib.Path("evidence.py").read_text()
for src in (SPARKLINE, EVIDENCE):
    assert "'''" not in src, "embedded source must not contain triple single-quotes"


# --------------------------------------------------------------------------------------

md(f"""# Sparkline Generator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/{BASE}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath={BASE}/demo.ipynb)

> A sparkline is a chart. Every failure mode of a chart applies to it - but at 80x20 pixels,
> inside a table cell, nobody audits it. **The default one-liner autoscales each row to its own
> min and max, which makes a row that grew 1% and a row that grew 9,900% into the same picture.**

**What this covers**

1. What a sparkline is, and the four decisions hiding inside it
2. **The main event**: per-row autoscale emits byte-identical paths for data 2,000,000x apart
3. The cell's aspect ratio, not the data, sets the perceived trend
4. A gap drawn through is invented data at full stroke weight
5. Endpoints vs a robust trend - and why the right answer depends on a noise model
6. Irregular time plotted at equal steps is a different series
7. SVG mechanics at 20 pixels tall: clipping, stroke scaling, byte budget, escaping
8. The verdict the tool is entitled to print
9. Try your own table

*Fully offline - all data is generated, standard library plus numpy and matplotlib.*""")

# --------------------------------------------------------------------------------------

md("""## 0. Setup

`sparkline.py` has no dependencies beyond the standard library. numpy and matplotlib are used
only for the experiments and the charts.""")

code(
    "import pathlib\n\n"
    "SPARKLINE_SRC = r'''" + SPARKLINE + "'''\n\n"
    "EVIDENCE_SRC = r'''" + EVIDENCE + "'''\n\n"
    'pathlib.Path("sparkline.py").write_text(SPARKLINE_SRC)\n'
    'pathlib.Path("evidence.py").write_text(EVIDENCE_SRC)\n\n'
    "import evidence\n"
    "from sparkline import *  # noqa: F403\n"
    "from IPython.display import HTML, display\n\n"
    'print("modules written and imported")'
)

# --------------------------------------------------------------------------------------

md("""## 1. What a sparkline is, and the four decisions inside it

Tufte's definition: a small, high-resolution graphic embedded in a context of words, numbers
and images - "data-intense, design-simple, word-sized". The appeal is that it costs no page
space and the reader spends no time on it.

That last part is the problem. Nobody spends time on it, so nobody notices that producing one
required four decisions, every one of which can silently invert the message:

| decision | the lazy default | what it costs |
|---|---|---|
| **scale** | each row to its own min/max | rows stop being comparable |
| **geometry** | whatever the table cell is | perceived steepness is set by CSS |
| **gaps** | drop the NaNs and plot the rest | the gap is drawn as a trend |
| **time axis** | one point per index | irregular reporting is redrawn as regular |

Here is the module's whole surface. Note that the domain is a **return value with a `mode` on
it**, not an implicit side effect - you cannot draw without saying what you scaled against.""")

code(
    'row = Series("weekly_signups", [402.0, 404.0, 406.0, 408.0, 410.0, 412.0, 414.0, 416.0])\n'
    'geom = Geometry(width=80, height=20, stroke=1.25)\n\n'
    'domain = resolve_domain(row, "per_row")\n'
    "print(domain)\n"
    'print("pad = %.3f px (half the stroke, so the line is not clipped at the edges)" % geom.pad)\n\n'
    "svg = sparkline_svg(row, domain, geom)\n"
    "print(svg)\n"
    "display(HTML(svg))"
)

# --------------------------------------------------------------------------------------

md("""## 2. The main event: per-row autoscale discards level

Two rows of a real SaaS table. `enterprise_mrr` runs around $402k; `self_serve_mrr` runs around
$402. Same shape, 1000x apart in level.

Under per-row autoscale, they do not merely *look* similar. The emitted path strings are equal,
character for character. There is no rendering subtlety to recover, no anti-aliasing artefact to
squint at - the information is gone before the SVG is written.""")

code(
    'enterprise = Series("enterprise_mrr", [402.0, 404, 406, 408, 410, 412, 414, 416])\n'
    'self_serve = Series("self_serve_mrr", [0.402, 0.404, 0.406, 0.408, 0.410, 0.412, 0.414, 0.416])\n\n'
    'a = build_path(enterprise, resolve_domain(enterprise, "per_row"), geom)\n'
    'b = build_path(self_serve, resolve_domain(self_serve, "per_row"), geom)\n\n'
    'print("enterprise:", a.d)\n'
    'print("self-serve :", b.d)\n'
    'print()\n'
    'print("identical?", a.d == b.d)'
)

md("""Scaled up, the collapse is exact across eight rows spanning a factor of two million - and a
row with a genuinely *different shape* still separates, which is what proves the collapse is
about level specifically rather than a broken renderer.""")

code("scale = evidence.experiment_per_row_scale()")

md("""Read the `comparable` column, then the extent table under it. There is no free lunch here,
only a choice about which question you want the picture to answer:

- **`per_row`** answers *"what shape did this row make?"* and nothing else. Legible, incomparable.
- **`shared`** answers *"how do these rows compare in level?"* honestly, and at a two-million-fold
  spread it renders the small rows as sub-pixel flat lines. Comparable, illegible.
- **`indexed`** answers *"how much did each row change, proportionally?"* Every row gets full
  vertical extent and the comparison still means something. Usually the right default for a table
  of heterogeneous metrics.

The tool's position is that the *choice* must be printed, because the reader cannot see it.""")

code(
    "rows = sample_table()\n"
    'for m in ("per_row", "shared", "indexed"):\n'
    '    print("=" * 78)\n'
    "    print(audit_table(rows, mode=m).text())\n"
    "    display(HTML(render_table(rows, mode=m)))"
)

# --------------------------------------------------------------------------------------

md("""## 3. Geometry: the column width sets the trend

Cleveland's finding on slope judgement: accuracy peaks when the average absolute slope of the
line is near 45 degrees. Steeper and shallower both degrade it, and shallow degrades it into
"flat".

A table cell picks that angle from its own CSS. So the same 24 numbers read as a crisis in a
narrow column and as stability in a wide one - and a responsive table changes the story when the
browser window is resized.""")

code("aspect = evidence.experiment_aspect_ratio()")

md("""`banked_width()` inverts the relationship and tells you what cell the data actually wants.
The solve is closed-form: `tan` is monotone on [0, 90), so `median(tan theta) == tan(median
theta)` exactly, and horizontal spacing is proportional to the drawable width while the vertical
excursions are fixed by the height and the domain.""")

code(
    "rows = sample_table()\n"
    'print("%-18s %14s %16s" % ("row", "at 80x20", "banked to 45 deg"))\n'
    'print("-" * 52)\n'
    "for s in rows:\n"
    "    if len(s.present) < 2:\n"
    "        continue\n"
    '    d = resolve_domain(s, "per_row")\n'
    "    p = build_path(s, d, geom)\n"
    '    print("%-18s %13.0f° %13.0f px" % (s.label, banking_deg(p.points), banked_width(s, d, geom)))'
)

# --------------------------------------------------------------------------------------

md("""## 4. Gaps: drawing through a hole invents data

An API had an outage window where metrics were not collected. The values existed; they were not
recorded. The one-liner drops the NaNs and plots what is left, which joins the two sides with a
straight line at full stroke weight - visually indistinguishable from measurement, and always
monotone, so a gap over a spike renders as a clean trend.""")

code("gaps = evidence.experiment_gaps()")

md("""Side by side, at a size where you can see it. The middle row is what the data supports; the
bottom row is what the shortcut draws.""")

code(
    """truth = [120.0, 124, 130, 240, 260, 205, 150, 128, 126, 124]
observed = list(truth)
for i in (3, 4, 5):
    observed[i] = None

g = Geometry(width=200, height=44, stroke=1.6)
d = resolve_domain(Series("t", truth), "per_row")

cases = [
    ("truth (all 10 points)", Series("t", truth), False),
    ("observed, gap broken", Series("o", observed), False),
    ("observed, gap bridged", Series("o", observed), True),
]

CELL = "<tr><td style='padding:6px 14px;font:13px system-ui'>{}</td><td>{}</td></tr>"
body = "".join(
    CELL.format(label, sparkline_svg(srs, d, g, bridge_gaps=bridge))
    for label, srs, bridge in cases
)
display(HTML("<table style='border-collapse:collapse'>" + body + "</table>"))"""
)

# --------------------------------------------------------------------------------------

md("""## 5. Endpoints vs a robust trend

A sparkline's shape invites one summary above all others: is the right end higher than the left
end? That is `sign(last - first)` - two observations out of n, and the noisiest reading available.

The honest answer to "so use a robust trend instead" is: *it depends on the noise model, and the
sparkline shows you neither*. Under a trend plus independent noise, throwing away n-2 points is
expensive. Under a random walk, last-minus-first is the efficient statistic and a robust slope is
worse. Under one contaminated final reading, the endpoint answer is not noisy - it is inverted.""")

code("trend = evidence.experiment_endpoint_vs_robust(trials=4000, n=24, seed=7)")

md("""Concretely, on the `active_seats` row from the sample table: seats fell every week for a
quarter and then had one big final week. The endpoints report growth.""")

code(
    'seats = Series("active_seats", [980.0, 940, 900, 860, 820, 790, 760, 1010])\n'
    'arrow = {1: "up", -1: "down", 0: "flat"}\n'
    'print("endpoints say         :", arrow[endpoint_direction(seats)])\n'
    'print("Theil-Sen trend says  :", arrow[trend_direction(seats)])\n'
    'print("Theil-Sen slope       : %.1f seats per week" % theil_sen_slope(seats))\n'
    'print("first 7 weeks         : %.0f -> %.0f (%.0f%%)"\n'
    "      % (seats.values[0], seats.values[6], 100 * (seats.values[6] / seats.values[0] - 1)))\n"
    'display(HTML(sparkline_svg(seats, resolve_domain(seats, "per_row"), Geometry(width=160, height=36, stroke=1.6))))'
)

# --------------------------------------------------------------------------------------

md("""## 6. Time: equal steps for unequal intervals

A metric reported monthly, then not reported for six months, then reported again. Plotting one
point per index puts the six-month jump at the same horizontal width as a one-month step, so the
steepest-looking segment in the chart is the one covering the longest period - exactly backwards.""")

code("timing = evidence.experiment_time_axis()")

code(
    """nps = Series("nps", [31.0, 33, 34, 36, 41, 42], times=[0.0, 1, 2, 3, 9, 10])
g = Geometry(width=200, height=40, stroke=1.6)
d = resolve_domain(nps, "per_row")

body = "".join(
    CELL.format(label, sparkline_svg(nps, d, g, use_times=use))
    for label, use in (("x from <code>times</code>", True), ("x from index", False))
)
display(HTML("<table style='border-collapse:collapse'>" + body + "</table>"))

print("Theil-Sen per month, x from times: %.3f" % theil_sen_slope(nps, use_times=True))
print("Theil-Sen per month, x from index: %.3f" % theil_sen_slope(nps, use_times=False))"""
)

# --------------------------------------------------------------------------------------

md("""## 7. SVG mechanics at 20 pixels tall

Four things that are invisible at this size and wrong anyway:

- **Half-stroke clipping.** A value at the top of the domain sits at `y = 0`, so half its stroke
  renders outside the viewBox. `pad = stroke/2` is not decoration.
- **Stroke scaling.** A responsive `viewBox` scales `stroke-width` with the cell, so the same
  1.25px line is 0.62px in a narrow column and 5px in a wide one - the same data reading bolder
  in one place than another. `vector-effect="non-scaling-stroke"` pins it.
- **Coordinate precision** is a much smaller lever than the wrapper. Most of the payload is
  `xmlns`, `viewBox`, `title` and `aria-label`.
- **Escaping.** Row labels come from data. Data is hostile.""")

code("mech = evidence.experiment_svg_mechanics(rows=500)")

# --------------------------------------------------------------------------------------

md("""## 8. The picture, at a size where the failures are visible

Six panels, one per experiment. Every one of these is drawn from the same code path the 80x20
version uses - the only thing that changed is that you can see it.""")

code(
    "%matplotlib inline\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "\n"
    'INK, BLUE, RED, GREY, GREEN = "#1f2933", "#1f4e79", "#c0392b", "#9aa5b1", "#2f855a"\n'
    'plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",\n'
    '                     "figure.dpi": 110, "axes.edgecolor": "#cbd2d9"})\n'
    "\n"
    "fig, axes = plt.subplots(2, 3, figsize=(15, 7.2))\n"
    "fig.suptitle(\n"
    '    "Six ways a sparkline lies - none of them visible at 80x20 px",\n'
    '    fontsize=13, fontweight="bold",\n'
    ")\n"
    "\n"
    "# (1) per-row autoscale collapses level\n"
    "ax = axes[0][0]\n"
    "shape = np.array([0, 3, 2, 5, 4, 7, 6, 8]) / 8.0\n"
    "for k, lv in enumerate((1.0, 512.0, 2.0 ** 21)):\n"
    "    v = lv * (1.0 + 0.5 * shape)\n"
    "    ax.plot(np.arange(8), (v - v.min()) / (v.max() - v.min()) - k * 1.25, color=BLUE, lw=1.6)\n"
    '    ax.text(-0.5, 0.5 - k * 1.25, "%.3g" % lv, ha="right", va="center", fontsize=7, color=GREY)\n'
    "ax.set_xlim(-2.6, 7.3)\n"
    "ax.set_xticks([])\n"
    "ax.set_yticks([])\n"
    'ax.set_title("1. Three rows, one shape, 2,000,000x apart\\nin level - byte-identical paths")\n'
    'for sp in ax.spines.values():\n'
    "    sp.set_visible(False)\n"
    "\n"
    "# (2) aspect ratio\n"
    "ax = axes[0][1]\n"
    "rng = np.random.default_rng(3)\n"
    "vals = 100 + 0.9 * np.arange(24) + rng.normal(0, 2.2, 24)\n"
    's = Series("x", list(map(float, vals)))\n'
    'dd = resolve_domain(s, "per_row")\n'
    "for k, w in enumerate((15.0, 30.0, 120.0, 480.0)):\n"
    "    p = build_path(s, dd, Geometry(width=w, height=20.0))\n"
    "    ax.plot([q[0] for q in p.points], [-q[1] - k * 30 for q in p.points], color=BLUE, lw=1.3)\n"
    '    ax.text(0, 6 - k * 30, "%.0f px - median slope %.0f°" % (w, banking_deg(p.points)),\n'
    '            fontsize=6.8, color=RED if banking_deg(p.points) > 45 or banking_deg(p.points) < 8 else GREY)\n'
    "ax.set_xlim(0, 500)\n"
    'ax.set_aspect("equal")\n'
    "ax.set_xticks([])\n"
    "ax.set_yticks([])\n"
    'ax.set_title("2. One series, four column widths, to scale\\n\'volatile\' and \'flat\' are the same numbers")\n'
    "for sp in ax.spines.values():\n"
    "    sp.set_visible(False)\n"
    "\n"
    "# (3) the bridged gap\n"
    "ax = axes[0][2]\n"
    "truth = np.array([120.0, 124, 130, 240, 260, 205, 150, 128, 126, 124])\n"
    "xs = np.arange(10)\n"
    'ax.plot(xs, truth, color=GREY, lw=1.2, ls=":", label="truth (not collected)")\n'
    'ax.plot(xs[:3], truth[:3], color=BLUE, lw=2.2, label="observed")\n'
    "ax.plot(xs[6:], truth[6:], color=BLUE, lw=2.2)\n"
    'ax.plot([2, 6], [truth[2], truth[6]], color=RED, lw=2.2, label="bridged (invented)")\n'
    'ax.axvspan(2.05, 5.95, color="#fdecea", zorder=0)\n'
    'ax.set_title("3. Bridging a gap: 44%% of the ink is invented,\\nmissing the truth by 81%% of the cell height")\n'
    'ax.legend(frameon=False, fontsize=6.8, loc="upper right")\n'
    "ax.set_xticks([])\n"
    'ax.set_ylabel("p95 ms", fontsize=7)\n'
    'ax.spines[["top", "right"]].set_visible(False)\n'
    "\n"
    "# (4) trend estimator accuracy\n"
    "ax = axes[1][0]\n"
    'models = ["trend+iid", "random_walk", "trend+outlier"]\n'
    "xp = np.arange(3)\n"
    'ax.bar(xp - 0.19, [100 * trend[m]["endpoint"] for m in models], 0.36, color=RED, label="sign(last - first)")\n'
    'ax.bar(xp + 0.19, [100 * trend[m]["robust"] for m in models], 0.36, color=GREEN, label="Theil-Sen")\n'
    'ax.axhline(50, color=GREY, lw=0.8, ls="--")\n'
    "for i, m in enumerate(models):\n"
    '    for off, key in ((-0.19, "endpoint"), (0.19, "robust")):\n'
    '        ax.text(i + off, 100 * trend[m][key] + 2, "%.0f" % (100 * trend[m][key]),\n'
    '                ha="center", fontsize=6.8)\n'
    "ax.set_xticks(xp)\n"
    "ax.set_xticklabels(models, fontsize=7)\n"
    "ax.set_ylim(0, 112)\n"
    'ax.set_ylabel("% of trials recovering\\nthe true direction", fontsize=7)\n'
    'ax.set_title("4. Which trend reading is right depends on the\\nnoise model - the picture never shows it")\n'
    'ax.legend(frameon=False, fontsize=6.8, loc="lower left")\n'
    'ax.spines[["top", "right"]].set_visible(False)\n'
    "\n"
    "# (5) time vs index\n"
    "ax = axes[1][1]\n"
    "tt = np.array([0.0, 1, 2, 3, 9, 10])\n"
    "nn = np.array([31.0, 33, 34, 36, 41, 42])\n"
    'ax.plot(tt, nn, color=GREEN, lw=1.8, marker="o", ms=3.5, label="x from `times` (months)")\n'
    'ax.plot(np.linspace(0, 10, 6), nn, color=RED, lw=1.8, marker="o", ms=3.5, ls="--",\n'
    '        label="x from index (equal steps)")\n'
    'ax.axvspan(3.05, 8.95, color="#f7f9fb", zorder=0)\n'
    'ax.text(6.0, 31.6, "6 months\\nnot reported", fontsize=6.8, color=GREY, ha="center")\n'
    'ax.set_title("5. Irregular reporting at equal steps\\noverstates the monthly trend by 120%")\n'
    'ax.set_xlabel("month", fontsize=7)\n'
    'ax.set_ylabel("NPS", fontsize=7)\n'
    'ax.legend(frameon=False, fontsize=6.8, loc="upper left")\n'
    'ax.spines[["top", "right"]].set_visible(False)\n'
    "\n"
    "# (6) stroke scaling\n"
    "ax = axes[1][2]\n"
    "widths = np.array([40.0, 80.0, 160.0, 320.0])\n"
    "eff = 1.25 * widths / 80.0\n"
    "cats = np.arange(4)\n"
    'ax.plot(cats, eff, color=RED, marker="o", ms=4, lw=1.6, label="stroke as rendered")\n'
    'ax.axhline(1.25, color=GREEN, lw=1.6, label=\'vector-effect="non-scaling-stroke"\')\n'
    "for c, e in zip(cats, eff):\n"
    '    ax.annotate("%.2f px" % e, (c, e), textcoords="offset points", xytext=(5, -2), fontsize=6.8)\n'
    "ax.set_xticks(cats)\n"
    'ax.set_xticklabels(["%.0f" % w for w in widths])\n'
    "ax.set_xlim(-0.25, 3.55)\n"
    'ax.set_xlabel("rendered column width (px), one 80x20 viewBox", fontsize=7)\n'
    'ax.set_ylabel("effective stroke (px)", fontsize=7)\n'
    'ax.set_title("6. The same 1.25 px line renders 0.62 to 5.00 px\\npurely from the column it lands in")\n'
    'ax.legend(frameon=False, fontsize=6.8, loc="upper left")\n'
    'ax.spines[["top", "right"]].set_visible(False)\n'
    "\n"
    "plt.tight_layout(rect=(0, 0, 1, 0.95))\n"
    'plt.savefig("sparkline_audit_nb.png", dpi=150, bbox_inches="tight", facecolor="white")\n'
    "plt.show()"
)

# --------------------------------------------------------------------------------------

md("""## 9. The verdict

`audit_table()` runs all four checks before anything is rendered and returns what the reader is
entitled to conclude. The sample table is deliberately nasty - every check fires on something.""")

code(
    "rows = sample_table()\n"
    'for m in ("per_row", "indexed"):\n'
    '    print("=" * 78)\n'
    "    print(audit_table(rows, mode=m, bridge_gaps=False).text())\n"
    "    print()\n"
    'print("=" * 78)\n'
    'print("and the same table with the shortcut turned on:")\n'
    'print(audit_table(rows, mode="per_row", bridge_gaps=True).text())'
)

# --------------------------------------------------------------------------------------

md("""## 10. Try your own table

Uncomment, point it at a wide CSV (first column the label, the rest the periods), and read the
verdict before the picture.""")

code(
    "# import pandas as pd\n"
    "#\n"
    '# df = pd.read_csv("your_metrics.csv")          # label column first, then one column per period\n'
    "# label_col, value_cols = df.columns[0], df.columns[1:]\n"
    "# rows = [\n"
    "#     Series(\n"
    "#         str(r[label_col]),\n"
    "#         [None if pd.isna(r[c]) else float(r[c]) for c in value_cols],\n"
    "#     )\n"
    "#     for _, r in df.iterrows()\n"
    "# ]\n"
    "#\n"
    "# # 1. read this BEFORE looking at any picture\n"
    '# print(audit_table(rows, mode="indexed").text())\n'
    "#\n"
    "# # 2. then render. indexed is usually right for a table of heterogeneous metrics;\n"
    "# #    shared when the rows are the same metric across segments.\n"
    '# display(HTML(render_table(rows, mode="indexed", geom=Geometry(width=90, height=22))))\n'
    "#\n"
    "# # 3. and check what cell width the data actually wants\n"
    "# for s in rows[:10]:\n"
    "#     if len(s.present) > 1:\n"
    '#         d = resolve_domain(s, "per_row")\n'
    '#         print("%-24s wants %.0f px at 22 px tall"\n'
    '#               % (s.label, banked_width(s, d, Geometry(width=90, height=22))))\n'
    'print("ready - uncomment the block above and point it at your own CSV")'
)

# --------------------------------------------------------------------------------------

md(f"""---

## What to take away

1. **Per-row autoscale does not compress level, it deletes it.** Eight rows spanning 2,000,000x
   emitted one path, byte for byte. If a table's sparklines are meant to be compared, the domain
   has to be shared or indexed - and the mode has to be printed, because the reader cannot see it.
2. **A sparkline's steepness is a CSS property.** The same 24 numbers ran from 59 degrees to 3
   degrees of median rendered slope across plausible column widths. Bank the cell to the data
   (`banked_width`) or accept that the layout is writing the conclusion.
3. **Never draw through a gap.** On the bundled outage, 44% of the drawn ink was invented and it
   missed the true path by 81% of the cell height, turning a spike-and-recovery into a clean rise.
4. **The trend reading a sparkline invites uses two points out of n**, and whether that is
   defensible depends on a noise model the picture cannot show you. Print a slope; do not imply one.
5. **Plot time on the time axis.** Equal index spacing made a six-month interval the steepest
   segment in the chart and overstated the monthly trend by 120%.
6. **At 20 pixels tall, `pad = stroke/2` and `non-scaling-stroke` are correctness, not polish** -
   and the byte budget lives in the wrapper, not the coordinates.

The honest summary: a sparkline is only ever half a statement. The other half is the number
printed beside it and the domain printed under it. That is why `render_table` emits all three and
`audit_table` runs first.

## Run the code

```bash
git clone https://github.com/phoebefu6/phoebe-the-builder
cd phoebe-the-builder/{BASE}
pip install -r requirements.txt

python3 test_sparkline.py    # 80 tests over the core
python3 test_evidence.py     # 29 tests over the experiments above
python3 evidence.py          # every table in this notebook
python3 make_chart.py        # the six-panel audit figure
streamlit run app.py         # the interactive version
```

Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder) - Day 137,
Analytics Engineering & BI.""")


pathlib.Path("demo.ipynb").write_text(json.dumps(nb, indent=1))
print(f"wrote demo.ipynb with {len(nb['cells'])} cells")
