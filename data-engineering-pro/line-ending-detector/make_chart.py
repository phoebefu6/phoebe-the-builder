"""Six panels, every number computed by lineends.py.

Writes eol_audit.png (the README figure) and eol_demo.png (the two-panel
version used in the app).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from lineends import (
    CORPUS,
    SPLITTERS,
    Verdict,
    chunk_drift,
    cr_contamination,
    detect_first,
    detect_majority,
    detect_strict,
    diff_blast,
    eol_histogram,
    finding_counts,
    line_count,
    roundtrip_table,
    verdict,
    verdict_counts,
)
from matplotlib.colors import LinearSegmentedColormap

INK = "#1d2733"
MUTED = "#6b7885"
GRID = "#dfe4ea"
WARM = "#c2571a"
COOL = "#2d5a68"
GREEN = "#2f6b39"
SAND = "#e8d9c0"
PLUM = "#5d4a7e"

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

FILES = [b.label for b in CORPUS]
KEYS = [s.key for s in SPLITTERS]


def panel_counts(ax: plt.Axes) -> None:
    data = np.array([[line_count(b, s) for s in SPLITTERS] for b in CORPUS], dtype=float)
    spread = data.max(axis=1) - data.min(axis=1)
    im = ax.imshow(
        np.repeat(spread[:, None], len(KEYS), axis=1), cmap=HEAT, vmin=0, vmax=max(1, spread.max())
    )
    for i in range(len(FILES)):
        for j in range(len(KEYS)):
            ax.text(j, i, int(data[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if spread[i] > spread.max() * 0.6 else INK)
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels(KEYS, rotation=55, ha="right", fontsize=6)
    ax.set_yticks(range(len(FILES)))
    ax.set_yticklabels(FILES, fontsize=6)
    ax.set_title(
        "1. Line count, shaded by how far the ten disagree",
        fontsize=9, loc="left", pad=8,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="max - min")


def panel_verdicts(ax: plt.Axes) -> None:
    colors = {
        Verdict.AGREED: GREEN,
        Verdict.CONTENT_DRIFT: SAND,
        Verdict.COUNT_DRIFT: WARM,
        Verdict.DATA_SPLIT: "#a5291c",
    }
    vals, cols = [], []
    for b in CORPUS:
        counts = [line_count(b, s) for s in SPLITTERS]
        vals.append(max(counts) - min(counts))
        cols.append(colors[verdict(b)])
    ax.barh(range(len(FILES)), [max(v, 0.12) for v in vals], color=cols, edgecolor=INK, lw=0.4)
    ax.set_yticks(range(len(FILES)))
    ax.set_yticklabels(FILES, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("spread in line count (max - min)")
    for i, b in enumerate(CORPUS):
        ax.text(max(vals[i], 0.12) + 0.05, i, verdict(b).value, va="center", fontsize=5.6,
                color=INK)
    ax.set_xlim(0, max(vals) + 1.6)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    vc = verdict_counts()
    ax.set_title(
        f"2. Verdicts - {vc[Verdict.AGREED]} agreed, {vc[Verdict.CONTENT_DRIFT]} content-drift, "
        f"{vc[Verdict.COUNT_DRIFT]} count-drift, {vc[Verdict.DATA_SPLIT]} data-split",
        fontsize=9, loc="left", pad=8,
    )


def panel_cr(ax: plt.Axes) -> None:
    cr = cr_contamination()
    vals = [cr[k] for k in KEYS]
    ax.bar(range(len(KEYS)), vals, color=[WARM if v else GREEN for v in vals],
           edgecolor=INK, lw=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=6, color=INK)
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels(KEYS, rotation=55, ha="right", fontsize=6)
    ax.set_ylabel("lines handed back with a trailing CR")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        "3. The carriage return that is still in the value", fontsize=9, loc="left", pad=8
    )


def panel_diff(ax: plt.Axes) -> None:
    rows = diff_blast()
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [a for _b, a, _c in rows], 0.4, color=COOL, label="the edit alone")
    ax.bar(x + 0.2, [c for _b, _a, c in rows], 0.4, color=WARM,
           label="the same edit, endings normalised")
    ax.set_xticks(x)
    ax.set_xticklabels([b.label for b, _a, _c in rows], rotation=55, ha="right", fontsize=6)
    ax.set_ylabel("lines a line-diff calls changed")
    ax.legend(fontsize=6, frameon=False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("4. One field edited, and what the diff says", fontsize=9, loc="left", pad=8)


def panel_roundtrip(ax: plt.Axes) -> None:
    rts = roundtrip_table()
    changed = [sum(1 for r in rts if r.splitter == k and r.changed) for k in KEYS]
    inside = [sum(1 for r in rts if r.splitter == k and r.inside_value) for k in KEYS]
    ax.bar(range(len(KEYS)), changed, color=SAND, edgecolor=INK, lw=0.4,
           label="bytes changed by read-then-write")
    ax.bar(range(len(KEYS)), inside, color="#a5291c", label="row count changed too")
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels(KEYS, rotation=55, ha="right", fontsize=6)
    ax.set_ylabel(f"files (of {len(CORPUS)})")
    ax.legend(fontsize=6, frameon=False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("5. Read it, write it back - what moved", fontsize=9, loc="left", pad=8)


def panel_detect(ax: plt.Axes) -> None:
    crlf = [eol_histogram(b.data)["CRLF"] for b in CORPUS]
    lf = [eol_histogram(b.data)["LF"] for b in CORPUS]
    cr = [eol_histogram(b.data)["CR"] for b in CORPUS]
    x = np.arange(len(CORPUS))
    ax.bar(x, crlf, 0.62, color=COOL, label="CRLF")
    ax.bar(x, lf, 0.62, bottom=crlf, color=SAND, edgecolor=INK, lw=0.3, label="LF")
    ax.bar(x, cr, 0.62, bottom=np.add(crlf, lf), color=WARM, label="CR")
    for i, b in enumerate(CORPUS):
        if detect_strict(b.data) is None:
            tag = "no single\nanswer" if detect_first(b.data) == detect_majority(b.data) else \
                f"{detect_first(b.data)} vs\n{detect_majority(b.data)}"
            ax.text(i, crlf[i] + lf[i] + cr[i] + 0.25, tag, ha="center", fontsize=5,
                    color=PLUM)
    ax.set_xticks(x)
    ax.set_xticklabels(FILES, rotation=55, ha="right", fontsize=6)
    ax.set_ylabel("terminators in the file")
    ax.set_ylim(0, max(np.add(np.add(crlf, lf), cr)) + 2.2)
    ax.legend(fontsize=6, frameon=False, ncol=3)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        "6. 'Detect the line ending' - first-seen vs majority vs strict",
        fontsize=9, loc="left", pad=8,
    )


def figure() -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 15.2))
    panel_counts(axes[0][0])
    panel_verdicts(axes[0][1])
    panel_cr(axes[1][0])
    panel_diff(axes[1][1])
    panel_roundtrip(axes[2][0])
    panel_detect(axes[2][1])
    fc = finding_counts()
    vc = verdict_counts()
    fig.suptitle(
        "A file has no lines in it - a splitter makes them\n"
        f"{len(CORPUS)} byte blobs x {len(SPLITTERS)} real splitters - "
        f"{vc[Verdict.AGREED]} of {len(CORPUS)} files read identically by all ten - "
        f"{len(chunk_drift())} chunk-boundary failures - "
        f"{fc['blocking']} blocking / {fc['silent']} silent / {fc['advisory']} advisory",
        fontsize=12.5, y=0.995, ha="center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig


def demo_figure() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    panel_cr(axes[0])
    panel_diff(axes[1])
    fig.tight_layout()
    return fig


def main() -> None:
    figure().savefig("eol_audit.png", dpi=190, bbox_inches="tight")
    demo_figure().savefig("eol_demo.png", dpi=190, bbox_inches="tight")
    print("wrote eol_audit.png, eol_demo.png")


if __name__ == "__main__":
    main()
