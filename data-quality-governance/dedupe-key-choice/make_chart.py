"""The dedupe-key-choice figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Top: the mechanism drawn - one typed name fanned out to the key each scheme makes for it, and the rows each key
joins to it, with the same person in green and other people in red. Bottom left: false splits and false merges per
key. Bottom right: the frontier of the two pairwise keys, every threshold judged pair by pair, with the chosen
threshold and the clustered result marked; the zero-error corner is empty.
Geometry self-check runs BEFORE savefig: every text inside the canvas, no two labels overlapping, no legend on data.
"""

from __future__ import annotations

import json
from typing import List

import matplotlib

matplotlib.use("Agg")
import explore as E
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["text.parse_math"] = False
plt.rcParams["font.family"] = ["DejaVu Sans"]

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
SPLIT, MERGE, OK, BAD = "#2f6fdb", "#c8562b", "#2e8b57", "#c8562b"

R = json.load(open("results.json"))["main"]
NAMES = E.NAMES
N = R["n"]
TP = R["ceiling"]["true_pairs"]
S = "Jonathan Smith"
SI = [s for s, _, _ in E.CORPUS].index(S)
TRUTH = E.CORPUS[SI][1]

fig = plt.figure(figsize=(16, 11.5), facecolor=BG)
axM = fig.add_axes([0.03, 0.57, 0.94, 0.32])
axP = fig.add_axes([0.25, 0.09, 0.33, 0.38])
axS = fig.add_axes([0.68, 0.09, 0.29, 0.37])
labels: List = []

fig.text(
    0.03,
    0.945,
    "Same customer, three rows: the match key decides how many customers you have",
    fontsize=15,
    color=INK,
    weight="bold",
)
fig.text(
    0.03,
    0.918,
    f"{N} CRM rows, {R['true_customers']} real customers, {TP} true duplicate pairs. The six keys leave between "
    f"{min(p['customers'] for p in R['per'].values())} and {max(p['customers'] for p in R['per'].values())} "
    "customers, and no threshold on either fuzzy key reaches zero errors.",
    fontsize=11,
    color=MUTED,
)

# ---------------------------------------------------------------- M. one name, six keys
ax = axM
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 12)
ax.add_patch(
    FancyBboxPatch((2, 5), 17, 2, boxstyle="round,pad=0.1,rounding_size=0.5", fc="#f4f6f9", ec=INK, lw=1.6)
)
labels.append(ax.text(10.5, 6, S, ha="center", va="center", fontsize=17, color=INK, family="DejaVu Sans Mono"))
labels.append(
    ax.text(10.5, 3.9, "one customer, typed four ways;\ntwo other Smiths in the table", ha="center", va="top", fontsize=10, color=MUTED)
)
for i, n in enumerate(NAMES):
    y = 11.0 - i * 1.9
    lab = R["per"][n]["labels"]
    key = lab[SI] if n in E.ROW_KEYS else "cluster"
    joined = [j for j in range(N) if j != SI and lab[j] == lab[SI]]
    good = [E.CORPUS[j][0] for j in joined if E.CORPUS[j][1] == TRUTH]
    bad = [E.CORPUS[j][0] for j in joined if E.CORPUS[j][1] != TRUTH]
    hot = bool(bad)
    ax.annotate(
        "",
        xy=(29, y),
        xytext=(19.6, 6),
        arrowprops=dict(arrowstyle="-|>", color=BAD if hot else MUTED, lw=1.4 if hot else 0.9),
    )
    labels.append(ax.text(29.8, y, n, va="center", fontsize=10.5, color=INK, weight="bold"))
    labels.append(ax.text(46, y, key, va="center", fontsize=10.5, color=INK, family="DejaVu Sans Mono"))
    txt = f"joins {len(good)} of {sum(1 for _, t, _ in E.CORPUS if t == TRUTH) - 1} same-person rows"
    labels.append(ax.text(60, y, txt, va="center", fontsize=9.5, color=OK))
    if bad:
        labels.append(
            ax.text(
                78.5,
                y,
                f"+ {len(bad)} other {'person' if len(bad) == 1 else 'people'}: " + ", ".join(sorted(set(bad))[:2]) + (" ..." if len(set(bad)) > 2 else ""),
                va="center",
                fontsize=9.5,
                color=BAD,
            )
        )

# ---------------------------------------------------------------- P. splits and merges per key
ax = axP
y = list(range(len(NAMES)))
sp = [len(R["per"][n]["false_splits"]) for n in NAMES]
mg = [len(R["per"][n]["false_merges"]) for n in NAMES]
ax.barh([v - 0.18 for v in y], sp, height=0.34, color=SPLIT, label="false split")
ax.barh([v + 0.18 for v in y], mg, height=0.34, color=MERGE, label="false merge")
for i in y:
    labels.append(ax.text(sp[i] + 0.5, i - 0.18, str(sp[i]), va="center", fontsize=9, color=INK))
    labels.append(ax.text(mg[i] + 0.5, i + 0.18, str(mg[i]), va="center", fontsize=9, color=INK))


def _tick(n: str) -> str:
    p = R["per"][n]
    ch = f", {p['chained_pairs']} chained" if p["chained_pairs"] else ""
    return f"{n}\n{p['customers']} customers (truth {R['true_customers']}){ch}"


ax.set_yticks(y, [_tick(n) for n in NAMES], fontsize=8.8, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, 44)
labels.append(
    ax.set_xlabel(f"row pairs (of {TP} true duplicate pairs); split = one customer becomes two, merge = two become one", color=INK, fontsize=9)
)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower right", fontsize=9)
labels.append(
    ax.set_title("Every key errs; the fuzzy ones trade splits for merges", loc="left", fontsize=11, color=INK, weight="bold", pad=8)
)

# ---------------------------------------------------------------- S. the frontier
ax = axS
for (n, fr), c in zip(R["frontier"].items(), (SPLIT, MERGE)):
    xs = [d["false_merges"] for d in fr]
    ys = [d["false_splits"] for d in fr]
    ax.plot(xs, ys, color=c, lw=2, label=n.split(" >=")[0] + ", every threshold")
    d = R["direct"][n]
    p = R["per"][n]
    ax.plot([d["false_merges"]], [d["false_splits"]], "o", color=c, ms=7)
    ax.plot([len(p["false_merges"])], [len(p["false_splits"])], "s", color=c, ms=7, mfc="white", mew=2)
ax.plot([0], [0], "*", color=OK, ms=14)
labels.append(ax.text(1.5, 1.2, "zero errors: no threshold gets here", color=OK, fontsize=9, va="bottom"))
ax.set_xlim(0, 46)
ax.set_ylim(0, 40)
ax.set_xticks(range(0, 46, 10))
ax.set_yticks(range(0, 41, 10))
labels.append(ax.set_xlabel("false merges (pairs)", color=INK))
labels.append(ax.set_ylabel("false splits (pairs)", color=INK))
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.plot([], [], "o", color=MUTED, label="the chosen threshold, pair by pair")
ax.plot([], [], "s", color=MUTED, mfc="white", mew=2, label="the same threshold, clustered")
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
labels.append(
    ax.set_title("No threshold reaches the corner; clustering moves it", loc="left", fontsize=11, color=INK, weight="bold", pad=8)
)


def check_geometry() -> None:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    for t in axP.get_yticklabels() + axP.get_xticklabels() + axS.get_xticklabels() + axS.get_yticklabels():
        bb = t.get_window_extent(rend)
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= W and bb.y1 <= H, f"tick off canvas: {t.get_text()!r}"
    boxes = []
    for t in labels:
        bb = t.get_window_extent(rend)
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= W and bb.y1 <= H, f"off canvas: {t.get_text()!r}"
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
    lb = axS.get_legend().get_window_extent(rend)
    for ln in axS.get_lines():
        for px, py in ln.get_transform().transform(ln.get_xydata()):
            assert not (lb.x0 <= px <= lb.x1 and lb.y0 <= py <= lb.y1), "legend covers a line point"
    lb = axP.get_legend().get_window_extent(rend)
    for p in axP.patches:
        b = p.get_window_extent(rend)
        assert not (lb.x0 < b.x1 and b.x0 < lb.x1 and lb.y0 < b.y1 and b.y0 < lb.y1), "legend covers a bar"
    # the mechanism panel's arrows must end before the key names start
    for t in labels:
        if t.axes is axM and t.get_text() in NAMES:
            assert t.get_position()[0] >= 29.5


if __name__ == "__main__":
    check_geometry()
    fig.savefig("audit.png", dpi=150, facecolor=BG)
    fig.savefig("audit.svg", facecolor=BG)
    print("wrote audit.png, audit.svg")
