"""Six panels, every number computed live from boolparse.

    python make_chart.py    ->  boolean_audit.png (300 DPI) + .svg
"""

from __future__ import annotations

import collections

import matplotlib

matplotlib.use("Agg")

import boolparse as B
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

INK = "#1d1a17"
MUTED = "#8a8178"
GRID = "#e3ddd5"
PAPER = "#faf7f2"
ACCENT = "#c8553d"
COOL = "#2f6f8f"
WARM = "#e0a458"
GREEN = "#4f7942"

#: One colour per verdict, used identically in every panel.
VERDICT_COLOUR = {
    B.TRUE: COOL,
    B.FALSE: WARM,
    B.REFUSED: GREEN,
    B.NOTBOOL: "#cfc7bd",
}
VERDICT_LABEL = {
    B.TRUE: "read as true",
    B.FALSE: "read as false",
    B.REFUSED: "refused (told you)",
    B.NOTBOOL: "not a boolean (deferred)",
}

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
})

READER_NAMES = [r.name for r in B.READERS]


def safe(text: str) -> str:
    """`B.show`, then swap glyphs DejaVu Sans has no coverage for.

    The corpus contains fullwidth Latin, which the chart font renders as
    tofu; folding it to ASCII and flagging the fold beats drawing empty
    boxes. Nothing else in the corpus needs this - DejaVu does cover
    LATIN SMALL LETTER LONG S.
    """
    shown = B.show(text)
    folded = "".join(
        chr(ord(ch) - 0xFEE0) if 0xFF01 <= ord(ch) <= 0xFF5E else ch for ch in shown
    )
    return f"[fullwidth]{folded}" if folded != shown else shown


LABELS = [safe(s.text) for s in B.CORPUS]


def strip(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=0)


def panel_grid(ax):
    """45 strings x 16 readers. Every cell is one live reading."""
    order = [B.TRUE, B.FALSE, B.REFUSED, B.NOTBOOL]
    idx = {v: i for i, v in enumerate(order)}
    data = np.array([[idx[B.grid()[n][j].verdict] for n in READER_NAMES]
                     for j in range(len(B.CORPUS))])
    cmap = matplotlib.colors.ListedColormap([VERDICT_COLOUR[v] for v in order])
    ax.imshow(data, cmap=cmap, aspect="auto", vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.set_xticks(range(len(READER_NAMES)))
    ax.set_xticklabels(READER_NAMES, rotation=90, fontsize=6)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS, fontsize=5.5)
    for x in range(1, len(READER_NAMES)):
        ax.axvline(x - 0.5, color=PAPER, lw=0.5)
    for y in range(1, len(LABELS)):
        ax.axhline(y - 0.5, color=PAPER, lw=0.35)
    strip(ax, keep=())
    ax.set_title(f"1  ·  {len(B.CORPUS)} strings × {len(B.READERS)} readers = "
                 f"{len(B.CORPUS) * len(B.READERS)} readings\n"
                 f"not one row is a single colour", loc="left")
    ax.legend(
        handles=[mpatches.Patch(color=VERDICT_COLOUR[v], label=VERDICT_LABEL[v]) for v in order],
        loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=6.5,
    )


def panel_false(ax):
    """The title bug, reader by reader."""
    s = next(x for x in B.CORPUS if x.text == "false")
    v = B.verdicts_for(s)
    names = READER_NAMES[::-1]
    colours = [VERDICT_COLOUR[v[n].verdict] for n in names]
    ax.barh(range(len(names)), [1] * len(names), color=colours, height=0.72)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xticks([])
    for i, n in enumerate(names):
        if v[n].verdict == B.TRUE:
            ax.text(0.5, i, "true", ha="center", va="center",
                    color="white", fontsize=7, fontweight="bold")
    n_true = sum(1 for r in v.values() if r.verdict == B.TRUE)
    strip(ax, keep=())
    ax.set_title(f'2  ·  the string "false"\n'
                 f"{n_true} of {len(B.READERS)} readers call it true", loc="left")


def panel_policy(ax):
    """Refusals against silent errors - the two columns are exclusive."""
    wrong = collections.Counter(name for _, name, _ in B.silently_wrong())
    refuse = B.refusal_counts()
    defer = B.notbool_counts()
    order = sorted(READER_NAMES, key=lambda n: (refuse[n] + defer[n]))
    y = np.arange(len(order))
    ax.barh(y, [-wrong[n] for n in order], color=ACCENT, height=0.66,
            label="confidently opposite the author's intent")
    ax.barh(y, [refuse[n] for n in order], color=GREEN, height=0.66, label="refused")
    ax.barh(y, [defer[n] for n in order], left=[refuse[n] for n in order],
            color=VERDICT_COLOUR[B.NOTBOOL], height=0.66, label="deferred (not a boolean)")
    ax.axvline(0, color=INK, lw=0.8)
    # Labels sit against the zero line rather than the far spine, so each
    # name stays next to the bar it belongs to.
    ax.set_yticks([])
    span = max(max(wrong.values()), max(refuse[n] + defer[n] for n in order))
    for i, n in enumerate(order):
        left = wrong[n] > 0
        ax.text(-span * 0.02 if left else span * 0.02, i, n,
                ha="right" if left else "left", va="center", fontsize=6.5, color=INK)
    ax.set_xlim(-span * 1.12, span * 1.12)
    ax.set_xlabel("← wrong quietly          told you →", fontsize=7)
    ax.set_xticks([-20, -10, 0, 10, 20, 30, 40])
    ax.set_xticklabels(["20", "10", "0", "10", "20", "30", "40"], fontsize=6.5)
    strip(ax, keep=("bottom",))
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax.set_title("3  ·  no reader is both permissive and safe\n"
                 "the two sides never both light up", loc="left")


def panel_roundtrip(ax):
    """Written for one reader, read by another."""
    rt = B.round_trip()
    data = np.array([[rt[(w, r)] for r in READER_NAMES] for w in READER_NAMES], dtype=float)
    im = ax.imshow(data, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=len(B.CORPUS))
    ax.set_xticks(range(len(READER_NAMES)))
    ax.set_xticklabels(READER_NAMES, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(READER_NAMES)))
    ax.set_yticklabels(READER_NAMES, fontsize=5.5)
    ax.set_xlabel("read by", fontsize=7)
    ax.set_ylabel("config written for", fontsize=7)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=6, length=0)
    cb.outline.set_visible(False)
    cb.set_label("strings lost", fontsize=6.5)
    strip(ax, keep=())
    ax.set_title("4  ·  the matrix is not symmetric\n"
                 "migration is safe towards strictness, not away", loc="left")


def panel_sqlite(ax):
    """Every word-shaped spelling of true is false in a SQLite WHERE."""
    verdicts = [B.verdict("sqlite_where", s).verdict for s in B.CORPUS]
    n_true = sum(1 for v in verdicts if v == B.TRUE)
    cols = 9
    rows = int(np.ceil(len(B.CORPUS) / cols))
    for i, (s, v) in enumerate(zip(B.CORPUS, verdicts)):
        r, c = divmod(i, cols)
        ax.add_patch(plt.Rectangle((c, rows - r - 1), 0.94, 0.94,
                                   color=VERDICT_COLOUR[v], lw=0))
        ax.text(c + 0.47, rows - r - 0.53, safe(s.text), ha="center", va="center",
                fontsize=5.2, color="white" if v == B.TRUE else INK)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    strip(ax, keep=())
    ax.set_title(f"5  ·  WHERE flag on a TEXT column\n"
                 f"{n_true} of {len(B.CORPUS)} strings select any rows at all", loc="left")


def panel_normalisation(ax):
    """One accept table, five normalisations."""
    strings = ["TRUE", "tRuE", " true", "true\r", "yeſ", "FALſE", "ＴＲＵＥ"]
    keys = list(B.normalisations("x").keys())
    data = np.array([[1 if B.accepted_after(t)[k] else 0 for k in keys] for t in strings])
    cmap = matplotlib.colors.ListedColormap([GRID, GREEN])
    ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(strings)))
    ax.set_yticklabels([safe(t) for t in strings], fontsize=7)
    for x in range(1, len(keys)):
        ax.axvline(x - 0.5, color=PAPER, lw=1.2)
    for y in range(1, len(strings)):
        ax.axhline(y - 0.5, color=PAPER, lw=1.2)
    strip(ax, keep=())
    ax.set_title("6  ·  the same accept table, five normalisations\n"
                 "casefold is not a stricter lower(), it is a different function", loc="left")


def main() -> None:
    fig = plt.figure(figsize=(15.5, 15.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.55, 1.0, 0.85],
                          hspace=0.42, wspace=0.30,
                          left=0.055, right=0.955, top=0.925, bottom=0.045)
    panel_grid(fig.add_subplot(gs[0, 0]))
    panel_policy(fig.add_subplot(gs[0, 1]))
    panel_false(fig.add_subplot(gs[1, 0]))
    panel_roundtrip(fig.add_subplot(gs[1, 1]))
    panel_sqlite(fig.add_subplot(gs[2, 0]))
    panel_normalisation(fig.add_subplot(gs[2, 1]))

    fig.suptitle("A string does not contain a boolean - a reader assigns one",
                 x=0.055, y=0.975, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.055, 0.951,
             f"Day 154 · {len(B.CORPUS)} strings through {len(B.READERS)} real readers · "
             f"{len(B.unanimous())} strings are read the same way by all of them · "
             f"{len(B.never_refuse())} of {len(B.READERS)} readers can never refuse",
             ha="left", fontsize=9, color=MUTED)

    fig.savefig("boolean_audit.png", dpi=300, facecolor=PAPER)
    fig.savefig("boolean_audit.svg", facecolor=PAPER)
    print("wrote boolean_audit.png and boolean_audit.svg")


if __name__ == "__main__":
    main()
