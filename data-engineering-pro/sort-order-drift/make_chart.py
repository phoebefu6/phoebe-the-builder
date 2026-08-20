"""Six panels, every number computed by collate.py.

Writes sort_order_audit.png (the README figure) and sort_order_demo.png (the
two-panel version used in the app).
"""

from __future__ import annotations

from itertools import combinations
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from collate import (
    CORPUS,
    COLLATIONS,
    PAGE_SIZES,
    Verdict,
    distinct_count,
    drift_matrix,
    finding_counts,
    keyset_pagination,
    offset_pagination,
    positions,
    range_counts,
    tie_groups,
    tied_rows,
    verdict,
)

INK = "#1d2733"
MUTED = "#6b7885"
GRID = "#dfe4ea"
WARM = "#c2571a"
COOL = "#2d5a68"
GREEN = "#2f6b39"
SAND = "#e8d9c0"
PAIRS = len(list(combinations(CORPUS, 2)))

HEAT = LinearSegmentedColormap.from_list("heat", ["#f6f3ee", "#e8d9c0", "#c2571a", "#7d2f0c"])

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

NAMES = [c.key_name for c in COLLATIONS]


def chart_label(name: str) -> str:
    """A label DejaVu Sans can actually draw.

    Fullwidth forms and anything above the BMP have no glyph in the figure
    font, and a chart that renders them as boxes is worse than one that names
    the code point.
    """
    out = []
    for ch in name:
        out.append(ch if ord(ch) < 0x2000 else f"U+{ord(ch):04X}")
    return "".join(out)


def panel_drift(ax: plt.Axes) -> None:
    m = drift_matrix()
    data = np.array([[m[(a, b)] for b in NAMES] for a in NAMES], dtype=float)
    im = ax.imshow(data, cmap=HEAT, vmin=0, vmax=data.max())
    ax.set_xticks(range(len(NAMES)))
    ax.set_yticks(range(len(NAMES)))
    ax.set_xticklabels(NAMES, rotation=55, ha="right", fontsize=6)
    ax.set_yticklabels(NAMES, fontsize=6)
    for i in range(len(NAMES)):
        for j in range(len(NAMES)):
            v = int(data[i, j])
            ax.text(
                j,
                i,
                v if v else "",
                ha="center",
                va="center",
                fontsize=5.4,
                color="white" if v > data.max() * 0.55 else INK,
            )
    ax.set_title(
        f"1. Row pairs returned in the opposite order (of {PAIRS})",
        fontsize=9,
        loc="left",
        pad=8,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def panel_positions(ax: plt.Axes) -> None:
    watch = [2, 5, 21, 26, 27]
    pos = {c.key_name: positions(c) for c in COLLATIONS}
    colors = [WARM, COOL, GREEN, "#8a5410", "#5d4a7e"]
    for k, rid in enumerate(watch):
        ys = [pos[n][rid] for n in NAMES]
        ax.plot(range(len(NAMES)), ys, "-o", ms=3.4, lw=1.2, color=colors[k],
                label=chart_label(CORPUS[rid - 1].name))
    ax.set_xticks(range(len(NAMES)))
    ax.set_xticklabels(NAMES, rotation=55, ha="right", fontsize=6)
    ax.set_ylabel("position in the result (0 = first)")
    ax.invert_yaxis()
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6, frameon=True, framealpha=0.95, edgecolor=GRID,
              ncol=1, loc="center left", bbox_to_anchor=(0.02, 0.42))
    ax.set_title("2. The same five rows, ten collations", fontsize=9, loc="left", pad=8)


def panel_ties(ax: plt.Axes) -> None:
    tied = [tied_rows(c) for c in COLLATIONS]
    groups = [len(tie_groups(c)) for c in COLLATIONS]
    x = np.arange(len(NAMES))
    ax.bar(x - 0.2, tied, 0.4, color=WARM, label="rows inside a tie")
    ax.bar(x + 0.2, groups, 0.4, color=SAND, edgecolor=INK, lw=0.4, label="tie groups")
    for i, c in enumerate(COLLATIONS):
        d = distinct_count(c)
        if d != len({r.name for r in CORPUS}):
            ax.text(i, tied[i] + 0.6, f"DISTINCT\n{d}", ha="center", fontsize=5.6, color=WARM)
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES, rotation=55, ha="right", fontsize=6)
    ax.set_ylabel(f"rows (of {len(CORPUS)})")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6, frameon=False)
    ax.set_title(
        "3. Ties: rows whose order is the plan's choice, not the query's",
        fontsize=9,
        loc="left",
        pad=8,
    )


def panel_pagination(ax: plt.Axes) -> None:
    data = np.zeros((len(NAMES), len(PAGE_SIZES)))
    for i, c in enumerate(COLLATIONS):
        for j, n in enumerate(PAGE_SIZES):
            data[i, j] = len(offset_pagination(c, n).lost)
    im = ax.imshow(data, cmap=HEAT, vmin=0, vmax=max(1.0, data.max()))
    ax.set_xticks(range(len(PAGE_SIZES)))
    ax.set_xticklabels([str(n) for n in PAGE_SIZES], fontsize=6)
    ax.set_yticks(range(len(NAMES)))
    ax.set_yticklabels(NAMES, fontsize=6)
    ax.set_xlabel("page size")
    for i in range(len(NAMES)):
        for j in range(len(PAGE_SIZES)):
            v = int(data[i, j])
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=5.6,
                        color="white" if v > data.max() * 0.55 else INK)
    ax.set_title(
        "4. Rows OFFSET paging never returns (0 everywhere once you add `, id`)",
        fontsize=9,
        loc="left",
        pad=8,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def panel_range(ax: plt.Axes) -> None:
    counts = range_counts()
    vals = [counts[n] for n in NAMES]
    colors = [COOL if v == max(vals) else WARM if v == min(vals) else SAND for v in vals]
    ax.barh(range(len(NAMES)), vals, color=colors, edgecolor=INK, lw=0.4)
    ax.set_yticks(range(len(NAMES)))
    ax.set_yticklabels(NAMES, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel(f"rows matching (of {len(CORPUS)})")
    for i, v in enumerate(vals):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=6, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        "5. WHERE name >= 'A' AND name < 'N' - same table, same predicate",
        fontsize=9,
        loc="left",
        pad=8,
    )


def panel_keyset(ax: plt.Axes) -> None:
    strict_lost: List[int] = []
    loose_dup: List[int] = []
    stalls: List[int] = []
    for c in COLLATIONS:
        strict_lost.append(sum(len(keyset_pagination(c, n, strict=True).lost) for n in PAGE_SIZES))
        loose_dup.append(
            sum(len(keyset_pagination(c, n, strict=False).duplicated) for n in PAGE_SIZES)
        )
        stalls.append(sum(1 for n in PAGE_SIZES if keyset_pagination(c, n, strict=False).stalled))
    x = np.arange(len(NAMES))
    ax.bar(x - 0.2, loose_dup, 0.4, color=COOL, label="rows repeated by `>=`")
    ax.bar(x + 0.2, strict_lost, 0.4, color=WARM, label="rows lost by `>`")
    for i, s in enumerate(stalls):
        if s:
            ax.text(i - 0.2, loose_dup[i] + 1.2, f"{s}/{len(PAGE_SIZES)}\nstall",
                    ha="center", fontsize=5.4, color=COOL)
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES, rotation=55, ha="right", fontsize=6)
    ax.set_ylabel(f"rows, summed over page sizes {PAGE_SIZES[0]}-{PAGE_SIZES[-1]}")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6, frameon=False)
    ax.set_title(
        "6. Keyset paging: `>` loses the tie group, `>=` never terminates",
        fontsize=9,
        loc="left",
        pad=8,
    )


def figure() -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 15.2))
    panel_drift(axes[0][0])
    panel_positions(axes[0][1])
    panel_ties(axes[1][0])
    panel_pagination(axes[1][1])
    panel_range(axes[2][0])
    panel_keyset(axes[2][1])
    counts = finding_counts()
    v = {k.value: 0 for k in Verdict}
    for c in COLLATIONS:
        v[verdict(c).value] += 1
    fig.suptitle(
        "ORDER BY name is a collation, not an order\n"
        f"{len(CORPUS)} rows x {len(COLLATIONS)} collations - "
        f"{v['tied']} tied, {v['merging']} merging, {v['stable-total']} stable-total, "
        f"{v['total']} total - "
        f"{counts['blocking']} blocking / {counts['silent']} silent / "
        f"{counts['advisory']} advisory findings",
        fontsize=12.5,
        y=0.995,
        ha="center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig


def demo_figure() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    panel_positions(axes[0])
    panel_ties(axes[1])
    fig.tight_layout()
    return fig


def main() -> None:
    figure().savefig("sort_order_audit.png", dpi=190, bbox_inches="tight")
    demo_figure().savefig("sort_order_demo.png", dpi=190, bbox_inches="tight")
    print("wrote sort_order_audit.png, sort_order_demo.png")


if __name__ == "__main__":
    main()
