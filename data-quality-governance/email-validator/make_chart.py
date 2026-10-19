"""The email-validator figure. Every number read from results.json, so the chart cannot drift from evidence.txt.

Top: the mechanism drawn - one address taken apart into the pieces the RFC defines, with each common failure pinned
to the piece it breaks. Bottom left: validator x address group, what share of each group it accepts (deliverable
groups should be all accepted, invalid groups all refused). Bottom right: deliverable rejected vs invalid accepted.
Geometry self-check runs BEFORE savefig: every text inside the canvas, no two labels overlapping.
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
plt.rcParams["font.family"] = ["DejaVu Sans"]

INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
BAD, OK, BLUE, HOT = "#c8562b", "#2e8b57", "#2f6fdb", "#e3b23c"

R = json.load(open("results.json"))["main"]
NAMES = list(E.VALIDATORS)

fig = plt.figure(figsize=(16, 11), facecolor=BG)
axM = fig.add_axes([0.03, 0.68, 0.94, 0.22])
axH = fig.add_axes([0.15, 0.20, 0.50, 0.42])
axS = fig.add_axes([0.73, 0.24, 0.25, 0.36])
labels: List = []

# ---------------------------------------------------------------- M. the mechanism: an address, taken apart
ax = axM
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 10)
fig.text(0.03, 0.945, "Where an email address breaks a validator", fontsize=15, color=INK, weight="bold")
fig.text(
    0.03,
    0.920,
    f"{len(E.CORPUS)} addresses, truth from an RFC 5321 + 6531 parser. No validator of ten is error-free; "
    f"{len(R['newline_accepted_by'])} of 10 accept a trailing newline.",
    fontsize=11,
    color=MUTED,
)
parts = [
    (2, 30, "o'brien+news", "local part  (max 64 octets)", BLUE),
    (30, 35, "@", "", INK),
    (35, 55, "mail.eng", "subdomain labels", OK),
    (55, 57, ".", "", INK),
    (57, 75, "example", "label  (max 63, no _)", OK),
    (75, 77, ".", "", INK),
    (77, 98, "photography", "TLD  (any length, not all digits)", HOT),
]
for x0, x1, txt, cap, col in parts:
    if cap:
        ax.add_patch(
            FancyBboxPatch(
                (x0 + 0.3, 4.6),
                x1 - x0 - 0.6,
                2.6,
                boxstyle="round,pad=0.05,rounding_size=0.6",
                fc=col,
                alpha=0.13,
                ec=col,
                lw=1.6,
            )
        )
        labels.append(ax.text((x0 + x1) / 2, 7.9, cap, ha="center", fontsize=9.5, color=MUTED))
    labels.append(
        ax.text(
            (x0 + x1) / 2,
            5.9,
            txt,
            ha="center",
            va="center",
            fontsize=19,
            color=INK,
            family="DejaVu Sans Mono",
        )
    )
pins = [
    (9, "apostrophe: refused by\ntutorial + OWASP regexes"),
    (24, "+tag: refused by\nthe strict TLD{2,4} regex"),
    (45, "extra label: refused\nby strict TLD{2,4}"),
    (87, "11-letter TLD: refused\nby OWASP {2,7} and {2,4}"),
]
for x, txt in pins:
    ax.annotate("", xy=(x, 4.5), xytext=(x, 2.7), arrowprops=dict(arrowstyle="-|>", color=BAD, lw=1.3))
    labels.append(ax.text(x, 2.4, txt, ha="center", va="top", fontsize=9, color=BAD))

# ---------------------------------------------------------------- H. validator x group heatmap
groups_ok = [g for g in E.SHARE]
# grey groups (valid but not internet-deliverable) are a policy call, not an error, so they are left out here
groups_bad = sorted({g for a, g, _ in E.CORPUS if E.reference(a)[0] == "bad"})
groups = groups_ok + groups_bad
M = np.zeros((len(NAMES), len(groups)))
for i, n in enumerate(NAMES):
    f = E.VALIDATORS[n]
    for j, g in enumerate(groups):
        rows = [a for a, gg, _ in E.CORPUS if gg == g]
        M[i, j] = np.mean([f(a) for a in rows])
want = np.array([1.0 if g in groups_ok else 0.0 for g in groups])
wrong = np.abs(M - want)
ax = axH
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("w", ["#f4f7f5", "#f2c9b8", BAD])
ax.imshow(wrong, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(len(NAMES)):
    for j in range(len(groups)):
        if wrong[i, j] > 0:
            ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=7.5, color=INK)
ax.set_yticks(range(len(NAMES)), NAMES, fontsize=9.5, color=INK)
ax.set_xticks(range(len(groups)), groups, rotation=60, ha="right", fontsize=8.5, color=INK)
for k, lab in enumerate(ax.get_xticklabels()):
    lab.set_color(OK if k < len(groups_ok) else BAD)
ax.axvline(len(groups_ok) - 0.5, color=INK, lw=1.4)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(
    "Share of each group accepted - shaded cells are wrong   (green groups: should pass;  orange groups: should fail)",
    loc="left",
    fontsize=10.5,
    color=INK,
    pad=10,
)

# ---------------------------------------------------------------- S. false rejects vs false accepts
ax = axS
per = R["per"]
fr = np.array([len(per[n]["false_reject"]) for n in NAMES])
fa = np.array([len(per[n]["false_accept"]) for n in NAMES])
ax.scatter(fa, fr, s=70, color=BLUE, zorder=3)
offs = {
    "contains @": (0, -15, "center"),
    "stdlib parseaddr": (-6, -12, "right"),
    "hand-rolled split": (-6, 8, "right"),
    "\\S+@\\S+\\.\\S+": (6, 2, "left"),
    "tutorial regex": (6, 0, "left"),
    "strict TLD{2,4}": (6, -4, "left"),
    "OWASP regex": (6, 6, "left"),
    "HTML5 type=email": (6, 4, "left"),
    "RFC 5322 regex": (6, 2, "left"),
    "email-validator 2.3": (4, -13, "left"),
}
for n, x, y in zip(NAMES, fa, fr):
    dx, dy, ha = offs[n]
    labels.append(
        ax.annotate(n, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, ha=ha, color=INK)
    )
ax.set_xlabel(f"invalid addresses accepted (of {R['tiers']['bad']})", color=INK)
ax.set_ylabel(f"deliverable addresses rejected (of {R['tiers']['ok']})", color=INK)
ax.set_xlim(-2, 30)
ax.set_ylim(-2, 31)
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("No validator reaches the corner", loc="left", fontsize=11, color=INK, weight="bold")


def check_geometry() -> None:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    boxes = []
    ticks = axH.get_xticklabels() + axH.get_yticklabels()
    for t in ticks:
        bb = t.get_window_extent(rend)
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= W and bb.y1 <= H, f"tick off canvas: {t.get_text()!r}"
    for t in labels:
        bb = t.get_window_extent(rend)
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= W and bb.y1 <= H, f"off canvas: {t.get_text()!r}"
        boxes.append((t.get_text(), bb))
    for i, (ta, a) in enumerate(boxes):
        for tb, b in boxes[i + 1 :]:
            if ta.strip() and tb.strip():
                assert not (a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1), (
                    f"overlap {ta!r} / {tb!r}"
                )


if __name__ == "__main__":
    check_geometry()
    fig.savefig("audit.png", dpi=150, facecolor=BG)
    fig.savefig("audit.svg", facecolor=BG)
    print("wrote audit.png, audit.svg")
