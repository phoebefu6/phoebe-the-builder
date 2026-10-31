"""The k-anonymity-check figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Top: the mechanism drawn - one export row walked down the generalisation ladder, with the rows an insider (who
knows the person is a patient) and a stranger (who does not) can re-identify at each step, and where k = 5 is
reached. Bottom left: unique-in-the-export vs unique-in-the-town as the export grows, at a coarse key. Bottom
right: a linkage attack with a complete register and with one missing 20% of the town.
Geometry self-check runs BEFORE savefig: every text inside the canvas, no two labels overlapping, no legend on data.
"""

from __future__ import annotations

import json
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["text.parse_math"] = False
plt.rcParams["font.family"] = ["DejaVu Sans"]

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
INSIDER, STRANGER, OK, BAD, NEUTRAL = "#2f6fdb", "#c8562b", "#2e8b57", "#c8562b", "#b8c0cb"

R = json.load(open("results.json"))["main"]
N = R["export_n"]
BY = {tuple(p["levels"]): p for p in R["ladder"]}
STEPS = [
    (("6 digits", "full date", "kept"), "as exported", "070701 / 1993-08-04 / M"),
    (("6 digits", "year", "kept"), "year of birth", "070701 / 1993 / M"),
    (("6 digits", "decade", "kept"), "decade", "070701 / 1990s / M"),
    (("sector (2 digits)", "year", "kept"), "sector + year", "07 / 1993 / M"),
    (("sector (2 digits)", "decade", "dropped"), "sector + decade,\nsex dropped", "07 / 1990s / *"),
    (("sector (2 digits)", "dropped", "kept"), "sector + sex", "07 / * / M"),
]

fig = plt.figure(figsize=(16, 11.5), facecolor=BG)
axM = fig.add_axes([0.03, 0.53, 0.94, 0.34])
axP = fig.add_axes([0.08, 0.09, 0.40, 0.37])
axS = fig.add_axes([0.58, 0.17, 0.39, 0.29])
labels: List = []

labels.append(
    fig.text(
        0.03,
        0.945,
        "The anonymised export was re-identified: how few columns it takes, counted on one town",
        fontsize=15,
        color=INK,
        weight="bold",
    )
)
full = BY[("6 digits", "full date", "kept")]
labels.append(
    fig.text(
        0.03,
        0.925,
        f"A fictional town of {R['pop_n']:,}, a {N:,}-row clinic export with names removed. Postcode + date of birth + sex "
        f"leave {full['unique_pop']:,} of {N:,} rows unique in the whole town;\nonly {len(R['k_anonymous'])} of "
        f"{len(R['ladder'])} generalisations reach k = 5 without suppressing rows, and the export's own k is not the town's.",
        fontsize=11,
        color=MUTED,
        va="top",
        linespacing=1.4,
    )
)

# ---------------------------------------------------------------- M. one row down the ladder
ax = axM
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 12)
labels.append(ax.text(0.5, 11.3, "the first export row (a fictional resident), six ways to write it", fontsize=10.5, color=INK, weight="bold"))
labels.append(ax.text(0.5, 2.1, "insider = knows the person is a patient, so a row unique in the EXPORT is enough", fontsize=9, color=INSIDER))
labels.append(ax.text(0.5, 1.3, "stranger = needs the row unique in the whole TOWN", fontsize=9, color=STRANGER))
W, GAP, X0 = 14.6, 1.9, 0.5
for i, (lv, title, sample) in enumerate(STEPS):
    p = BY[lv]
    x = X0 + i * (W + GAP)
    safe = p["below_k"] == 0
    ax.add_patch(
        FancyBboxPatch((x, 3.2), W, 7.2, boxstyle="round,pad=0.1,rounding_size=0.4", fc="#f4f6f9" if not safe else "#eef7f1", ec=OK if safe else INK, lw=1.8 if safe else 1.0)
    )
    labels.append(ax.text(x + W / 2, 9.75, title, ha="center", va="center", fontsize=10, color=INK, weight="bold"))
    labels.append(ax.text(x + W / 2, 8.45, sample, ha="center", va="center", fontsize=9.5, color=INK, family="DejaVu Sans Mono"))
    labels.append(ax.text(x + 0.6, 7.25, f"{p['classes']:,} classes, smallest {p['k_min']}", fontsize=8.8, color=MUTED))
    labels.append(ax.text(x + 0.6, 6.25, f"insider re-identifies {p['unique_export']:,}", fontsize=9, color=INSIDER))
    labels.append(ax.text(x + 0.6, 5.3, f"stranger re-identifies {p['unique_pop']:,}", fontsize=9, color=STRANGER))
    txt = "k >= 5: every row" if safe else f"k >= 5 needs {p['below_k']:,} rows\nsuppressed ({100 * p['below_k'] / N:.0f}%)"
    labels.append(ax.text(x + 0.6, 4.25, txt, fontsize=8.8, color=OK if safe else BAD, va="center"))
    if i < len(STEPS) - 1:
        ax.annotate("", xy=(x + W + GAP - 0.15, 6.8), xytext=(x + W + 0.15, 6.8), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))

# ---------------------------------------------------------------- P. the sampling sweep
ax = axP
sw = [s for s in R["sweep"] if tuple(s["levels"]) == ("sector (2 digits)", "year", "kept")]
xs = [100 * s["frac"] for s in sw]
ax.plot(xs, [s["unique_export_pct"] for s in sw], "-o", color=INSIDER, lw=2, ms=6, label="unique in the export (what the file shows)")
ax.plot(xs, [s["unique_pop_pct"] for s in sw], "-s", color=STRANGER, lw=2, ms=6, label="unique in the town (what the stranger needs)")
ax.set_xscale("log")
ax.set_xticks(xs, [f"{x:g}%" for x in xs])
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 20))
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
s5 = next(s for s in sw if s["frac"] == 0.05)
labels.append(
    ax.annotate(
        f"5% export: {s5['unique_export_pct']:.0f}% unique in the file,\n{s5['unique_pop_pct']:.2f}% unique in the town",
        xy=(5, s5["unique_export_pct"]),
        xytext=(9, 72),
        fontsize=9,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9),
    )
)
labels.append(ax.set_xlabel("share of the town in the export (log scale)", color=INK))
labels.append(ax.set_ylabel("rows unique on sector + year + sex", color=INK))
ax.legend(frameon=False, loc="lower left", fontsize=8.8)
labels.append(
    ax.set_title("Unique in the export is an upper bound, loose until the export is the town", loc="left", fontsize=11, color=INK, weight="bold", pad=8)
)

# ---------------------------------------------------------------- S. linkage
ax = axS
rows = [
    (("6 digits", "full date", "kept"), 1.0, "full key,\nregister complete"),
    (("6 digits", "full date", "kept"), 0.8, "full key,\nregister missing 20%"),
    (("6 digits", "year", "kept"), 1.0, "year of birth,\nregister complete"),
    (("6 digits", "year", "kept"), 0.8, "year of birth,\nregister missing 20%"),
]
link = {(tuple(k["levels"]), k["coverage"]): k for k in R["linkage"]}
ys = list(range(len(rows)))
left = [0] * len(rows)
for part, col, name in (("right", OK, "right person"), ("wrong", BAD, "WRONG person, confidently"), ("ambiguous", NEUTRAL, "several candidates"), ("none", "#e6e9ee", "not in register")):
    vals = [link[(lv, cov)][part] for lv, cov, _ in rows]
    ax.barh(ys, vals, left=left, height=0.6, color=col, label=name, edgecolor=BG, lw=0.6)
    for i, v in enumerate(vals):
        if part == "wrong" and v:
            labels.append(ax.text(left[i] + v + 25, i, f"{v} wrong", va="center", fontsize=9, color=BAD, weight="bold"))
        elif part == "right" and v > 300:
            labels.append(ax.text(left[i] + v / 2, i, f"{v:,}", va="center", ha="center", fontsize=9, color="white"))
    left = [a + b for a, b in zip(left, vals)]
ax.set_yticks(ys, [name for _, _, name in rows], fontsize=9, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, N)
ax.set_xticks(range(0, N + 1, 1000))
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
labels.append(ax.set_xlabel(f"the {N:,} export rows, looked up in a public register by their key", color=INK))
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2, fontsize=8.8)
labels.append(
    ax.set_title("A complete register is never wrong; one missing 20% is wrong 101 times", loc="left", fontsize=11, color=INK, weight="bold", pad=8)
)


def check_geometry() -> None:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W_, H_ = fig.bbox.width, fig.bbox.height
    for t in axP.get_yticklabels() + axP.get_xticklabels() + axS.get_xticklabels() + axS.get_yticklabels():
        bb = t.get_window_extent(rend)
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= W_ and bb.y1 <= H_, f"tick off canvas: {t.get_text()!r}"
    boxes = []
    for t in labels:  # includes the figure-level title and subtitle: the first render clipped the subtitle
        bb = t.get_window_extent(rend)
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= W_ and bb.y1 <= H_, f"off canvas: {t.get_text()!r}"
        boxes.append((t.get_text(), bb))
    for i, (ta, a) in enumerate(boxes):
        for tb, b in boxes[i + 1 :]:
            assert not (a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1), f"overlap {ta!r} / {tb!r}"
    for ax_ in (axP, axS):
        lb = ax_.get_legend().get_window_extent(rend)
        for t in labels:
            if t.axes is ax_:
                b = t.get_window_extent(rend)
                assert not (lb.x0 < b.x1 and b.x0 < lb.x1 and lb.y0 < b.y1 and b.y0 < lb.y1), f"legend on {t.get_text()!r}"
        assert lb.x0 >= 0 and lb.y0 >= 0 and lb.x1 <= W_ and lb.y1 <= H_, "legend off canvas"
    lb = axP.get_legend().get_window_extent(rend)
    for ln in axP.get_lines():
        for px, py in ln.get_transform().transform(ln.get_xydata()):
            assert not (lb.x0 <= px <= lb.x1 and lb.y0 <= py <= lb.y1), "legend covers a line point"
    lb = axS.get_legend().get_window_extent(rend)
    for p in axS.patches:
        b = p.get_window_extent(rend)
        assert not (lb.x0 < b.x1 and b.x0 < lb.x1 and lb.y0 < b.y1 and b.y0 < lb.y1), "legend covers a bar"
    # every ladder label sits inside its own box
    for patch in axM.patches:
        pb = patch.get_window_extent(rend)
        for t in labels:
            if t.axes is axM:
                tb = t.get_window_extent(rend)
                inter = pb.x0 < tb.x1 and tb.x0 < pb.x1 and pb.y0 < tb.y1 and tb.y0 < pb.y1
                if inter and t.get_text() not in (labels[0].get_text(), labels[1].get_text()):
                    assert tb.x0 >= pb.x0 and tb.x1 <= pb.x1 and tb.y0 >= pb.y0 and tb.y1 <= pb.y1, f"spills out of its box: {t.get_text()!r}"


if __name__ == "__main__":
    check_geometry()
    fig.savefig("audit.png", dpi=150, facecolor=BG)
    fig.savefig("audit.svg", facecolor=BG)
    print("wrote audit.png, audit.svg")
