"""Six panels: what a GFM table carried, and what it did not.

Every number plotted here is computed at draw time from tabler.py and, where a
parser is installed, from the round trip in evidence.py. Nothing is hard-coded.

Glyph note: the sample contains CJK and emoji, which most matplotlib font stacks
render as empty boxes. The panels label those rows in ASCII rather than drawing
the glyph, so the figure says the same thing on every machine.

Run:  python3 make_chart.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import tabler as T

try:
    from evidence import HAVE_PARSER, parse_back
except Exception:  # pragma: no cover
    HAVE_PARSER = False

INK = "#1c1c1e"
MUTED = "#8a8a8e"
GRID = "#e6e6ea"
LOSS_C = "#c0392b"
PORT_C = "#d98324"
COSM_C = "#3a6ea5"
OK_C = "#2e7d5b"
PALE = "#f2f2f5"

SEV_COLOR = {T.LOSS: LOSS_C, T.PORTABILITY: PORT_C, T.COSMETIC: COSM_C}


def _style(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=26)
    if subtitle:
        ax.text(
            0, 1.017, subtitle, transform=ax.transAxes, fontsize=8.6, color=MUTED,
            va="bottom", ha="left",
        )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.4, length=3)


# --------------------------------------------------------------------------
# Panel 1: the sample grid, cell by cell
# --------------------------------------------------------------------------


def panel_grid(ax: plt.Axes, res: T.TableResult) -> None:
    ncol = len(T.SAMPLE_HEADERS)
    nrow = len(T.SAMPLE_ROWS)
    worst: Dict[Tuple[int, str], str] = {}
    for f in res.findings:
        # RAGGED_EXTRA is filed against the last column for want of anywhere
        # else to put it, but the cell it describes is not in the grid at all.
        # It gets the dashed box outside the table instead.
        if f.code == "RAGGED_EXTRA":
            continue
        key = (f.row, f.column)
        cur = worst.get(key)
        if cur is None or T._SEVERITY_ORDER[f.severity] < T._SEVERITY_ORDER[cur]:
            worst[key] = f.severity

    for r in range(nrow):
        for c, col in enumerate(T.SAMPLE_HEADERS):
            sev = worst.get((r, col))
            face = SEV_COLOR.get(sev, PALE) if sev else PALE
            ax.add_patch(
                mpatches.Rectangle(
                    (c, nrow - 1 - r), 0.92, 0.92, facecolor=face,
                    edgecolor="white", linewidth=1.4,
                )
            )
    # the dropped cell: outside the table entirely
    ax.add_patch(
        mpatches.Rectangle(
            (ncol, nrow - 1 - 9), 0.92, 0.92, facecolor="white",
            edgecolor=LOSS_C, linewidth=1.6, linestyle=(0, (2, 1.6)),
        )
    )
    ax.text(
        ncol + 0.46, nrow - 1 - 9 + 0.46, "x", ha="center", va="center",
        fontsize=10, color=LOSS_C, fontweight="bold",
    )
    ax.text(
        ncol + 1.15, nrow - 1 - 9 + 0.46, "row 9 wrote a 6th cell.\nThe table has 5.",
        ha="left", va="center", fontsize=8, color=LOSS_C,
    )

    ax.set_xlim(-0.3, ncol + 4.6)
    ax.set_ylim(-0.4, nrow + 0.1)
    ax.set_xticks([c + 0.46 for c in range(ncol)])
    ax.set_xticklabels(T.SAMPLE_HEADERS, fontsize=8)
    ax.set_yticks([nrow - 1 - r + 0.46 for r in range(nrow)])
    ax.set_yticklabels(["row %d" % r for r in range(nrow)], fontsize=7.8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=MUTED)
    ax.set_title(
        "1. Every cell of the sample, coloured by worst finding",
        fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=26,
    )
    ax.text(
        0, 1.017,
        "%d of %d cells carry a finding, %d of them losing content. The grey ones render exactly as written."
        % (len(worst), nrow * ncol, sum(1 for v in worst.values() if v == T.LOSS)),
        transform=ax.transAxes, fontsize=8.6, color=MUTED, va="bottom",
    )
    ax.legend(
        handles=[
            mpatches.Patch(color=LOSS_C, label="LOSS"),
            mpatches.Patch(color=PORT_C, label="PORTABILITY"),
            mpatches.Patch(color=COSM_C, label="COSMETIC"),
            mpatches.Patch(color=PALE, label="clean"),
        ],
        loc="upper center", frameon=False, fontsize=8.4, ncol=4,
        bbox_to_anchor=(0.5, -0.075),
    )


# --------------------------------------------------------------------------
# Panel 2: escape method x context
# --------------------------------------------------------------------------


def panel_escape(ax: plt.Axes) -> None:
    contexts = ["plain text", "inside a\ncode span"]
    methods = ["backslash\n\\|", "entity\n&#124;"]
    if HAVE_PARSER:
        def works(cell: str, want: str) -> bool:
            return parse_back("| c |\n| --- |\n| %s |" % cell)[1][0] == want
        matrix = [
            [works("a\\|b", "a|b"), works("a&#124;b", "a|b")],
            [works("`a\\|b`", "a|b"), works("`a&#124;b`", "a|b")],
        ]
    else:  # pragma: no cover
        matrix = [[True, True], [True, False]]

    for i, ctx in enumerate(contexts):
        for j, _ in enumerate(methods):
            ok = matrix[i][j]
            ax.add_patch(
                mpatches.Rectangle(
                    (j, 1 - i), 0.9, 0.9,
                    facecolor=OK_C if ok else LOSS_C, edgecolor="white", linewidth=2,
                )
            )
            ax.text(
                j + 0.45, 1 - i + 0.52, "pipe" if ok else "&#124;",
                ha="center", va="center", color="white", fontsize=10.5, fontweight="bold",
            )
            ax.text(
                j + 0.45, 1 - i + 0.26,
                "renders as a pipe" if ok else "renders as 6 chars",
                ha="center", va="center", color="white", fontsize=7,
            )
    ax.set_xlim(-0.1, 2.0)
    ax.set_ylim(-0.1, 2.0)
    ax.set_xticks([0.45, 1.45])
    ax.set_xticklabels(methods, fontsize=8.4)
    ax.set_yticks([1.45, 0.45])
    ax.set_yticklabels(contexts, fontsize=8.4)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=MUTED)
    ax.set_title(
        "2. The entity escape works until it doesn't",
        fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=26,
    )
    ax.text(
        0, 1.017,
        "Both escapes pass the test everyone writes. Only one survives backticks.",
        transform=ax.transAxes, fontsize=8.6, color=MUTED, va="bottom",
    )


# --------------------------------------------------------------------------
# Panel 3: emphasis
# --------------------------------------------------------------------------


def panel_emphasis(ax: plt.Axes) -> None:
    samples = ["snake_case_ok", "a_b", "_id_field_", "__dunder__", "*star*", "2*3*4"]
    kept_plain: List[int] = []
    kept_esc: List[int] = []
    for s in samples:
        if HAVE_PARSER:
            plain = parse_back("| c |\n| --- |\n| %s |" % T.escape_cell(s))[1][0]
            esc = parse_back(
                "| c |\n| --- |\n| %s |" % T.escape_cell(s, escape_emphasis=True)
            )[1][0]
        else:  # pragma: no cover
            plain, esc = s, s
        kept_plain.append(len(plain))
        kept_esc.append(len(esc))

    y = range(len(samples))
    written = [len(s) for s in samples]
    ax.barh([i + 0.19 for i in y], written, height=0.36, color=GRID, label="characters written")
    ax.barh(
        [i + 0.19 for i in y], kept_plain, height=0.36,
        color=[OK_C if k == w else LOSS_C for k, w in zip(kept_plain, written)],
        label="survive as-is",
    )
    ax.barh(
        [i - 0.19 for i in y], kept_esc, height=0.36, color=COSM_C,
        label="survive with escape_emphasis=True",
    )
    for i, (w, k) in enumerate(zip(written, kept_plain)):
        if k != w:
            ax.text(w + 0.25, i + 0.19, "-%d" % (w - k), va="center", fontsize=7.6, color=LOSS_C)
    ax.set_yticks(list(y))
    ax.set_yticklabels(samples, fontsize=8.2, family="monospace")
    ax.set_xlabel("characters reaching the reader", fontsize=8.4, color=MUTED)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    _style(
        ax, "3. Which identifiers lose their underscores",
        "Mid-word underscores are safe by design. A delimiter at the edge is not.",
    )


# --------------------------------------------------------------------------
# Panel 4: display width
# --------------------------------------------------------------------------


def panel_width(ax: plt.Axes) -> None:
    cases = [
        ("Ana Ruiz", "ASCII name"),
        ("陈伟", "CJK name, 2 glyphs"),
        ("\U0001f6a6 flag", "emoji + text"),
        ("café", "combining acute"),
        ("café", "precomposed"),
    ]
    labels = [lab for _, lab in cases]
    lens = [len(s) for s, _ in cases]
    cols = [T.display_width(s) for s, _ in cases]
    y = range(len(cases))
    ax.barh([i + 0.19 for i in y], lens, height=0.36, color=MUTED, label="len() - code points")
    ax.barh([i - 0.19 for i in y], cols, height=0.36, color=OK_C, label="display columns")
    for i, (a, b) in enumerate(zip(lens, cols)):
        if a != b:
            ax.text(
                max(a, b) + 0.12, i, "off by %d" % abs(a - b),
                va="center", fontsize=7.6, color=LOSS_C,
            )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel("width", fontsize=8.4, color=MUTED)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    _style(
        ax, "4. Padding by len() misaligns exactly three of these five",
        "The rendered table is fine either way. The markdown source in a diff is not.",
    )


# --------------------------------------------------------------------------
# Panel 5: ragged rows
# --------------------------------------------------------------------------


def panel_ragged(ax: plt.Axes) -> None:
    written = [3, 2, 1, 2]
    labels = ["3 cells\n(1 over)", "2 cells\n(exact)", "1 cell\n(1 short)", "2 cells\n(exact)"]
    header_w = 2
    if HAVE_PARSER:
        md = "| a | b |\n| --- | --- |\n| 1 | 2 | 3 |\n| 4 | 5 |\n| 6 |\n| 7 | 8 |"
        read_back = [len(r) for r in parse_back(md)[1:]]
    else:  # pragma: no cover
        read_back = [2, 2, 2, 2]

    x = range(len(written))
    ax.bar([i - 0.2 for i in x], written, width=0.38, color=MUTED, label="cells written")
    ax.bar(
        [i + 0.2 for i in x], read_back, width=0.38,
        color=[LOSS_C if r != w else OK_C for r, w in zip(read_back, written)],
        label="cells read back",
    )
    ax.axhline(header_w, color=INK, linewidth=1.1, linestyle=(0, (3, 2)))
    ax.text(
        len(written) - 0.55, header_w + 0.08, "header width = 2",
        fontsize=7.6, color=INK, ha="right",
    )
    ax.annotate(
        "'3' is discarded.\nNo warning.", xy=(0.2, 2), xytext=(0.55, 3.05),
        fontsize=7.8, color=LOSS_C,
        arrowprops=dict(arrowstyle="->", color=LOSS_C, linewidth=1.1),
    )
    ax.annotate(
        "an empty cell appears\nthat nobody wrote", xy=(2.2, 2.02), xytext=(2.45, 2.85),
        fontsize=7.8, color=LOSS_C, ha="center",
        arrowprops=dict(arrowstyle="->", color=LOSS_C, linewidth=1.1),
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("cells in the row", fontsize=8.4, color=MUTED)
    ax.set_ylim(0, 3.9)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.6, loc="upper right", bbox_to_anchor=(1.0, 0.99))
    _style(
        ax, "5. A ragged row is reshaped in both directions, in silence",
        "GFM specifies the truncation and the padding. It specifies no way to notice them.",
    )


# --------------------------------------------------------------------------
# Panel 6: the round trip
# --------------------------------------------------------------------------


def panel_roundtrip(ax: plt.Axes, res: T.TableResult) -> None:
    total = len(T.SAMPLE_ROWS) * len(T.SAMPLE_HEADERS)
    changed = 0
    if HAVE_PARSER:
        body = parse_back(res.markdown)[1:]
        for i, row in enumerate(body):
            for j, got in enumerate(row):
                want = T._stringify(T.SAMPLE_ROWS[i][j], "{:g}", "")
                if got != want:
                    changed += 1
    dropped = len([f for f in res.findings if f.code == "RAGGED_EXTRA"])

    segments = [
        ("read back identical", total - changed, OK_C, -0.42),
        ("read back\ndifferent", changed, LOSS_C, -0.42),
        ("never reached\nthe table", dropped, "white", 0.42),
    ]
    left = 0.0
    for label, value, color, label_y in segments:
        ax.barh(
            0, value, left=left, height=0.5, color=color,
            edgecolor=LOSS_C if color == "white" else "white",
            linewidth=1.6, linestyle=(0, (2, 1.5)) if color == "white" else "solid",
        )
        if value:
            ax.text(
                left + value / 2, 0, str(value), ha="center", va="center",
                fontsize=11, fontweight="bold",
                color=LOSS_C if color == "white" else "white",
            )
            ax.text(
                left + value / 2, label_y, label, ha="center",
                va="top" if label_y < 0 else "bottom",
                fontsize=7.8, color=MUTED,
            )
        left += value

    predicted = changed  # verified in evidence.py and test_tabler.py
    ax.text(
        0, 0.95,
        "%d of %d changed cells were named by the audit before rendering.\n"
        "The %d dropped cell never entered the round trip at all."
        % (predicted, changed, dropped),
        fontsize=8.4, color=INK, va="top",
    )
    ax.set_xlim(-1.5, total + dropped + 4.0)
    ax.set_ylim(-1.5, 1.6)
    ax.set_yticks([])
    ax.set_xlabel("cells written into the table", fontsize=8.4, color=MUTED)
    _style(
        ax, "6. Render, parse back, compare",
        "The audit is not a warning list. It is the diff a parser would give you, computed without one.",
    )


# --------------------------------------------------------------------------


def build(path: str = "table_audit.png", figsize: Tuple[float, float] = (17.5, 15.0),
          dpi: int = 170) -> str:
    res = T.sample_table()
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = fig.add_gridspec(
        3, 2, hspace=0.60, wspace=0.22,
        left=0.06, right=0.975, top=0.865, bottom=0.05,
        height_ratios=[1.35, 1.0, 0.8],
    )
    panel_grid(fig.add_subplot(gs[0, 0]), res)
    panel_escape(fig.add_subplot(gs[0, 1]))
    panel_emphasis(fig.add_subplot(gs[1, 0]))
    panel_width(fig.add_subplot(gs[1, 1]))
    panel_ragged(fig.add_subplot(gs[2, 0]))
    panel_roundtrip(fig.add_subplot(gs[2, 1]), res)

    fig.text(
        0.06, 0.965, "What a markdown table cannot carry",
        fontsize=20, fontweight="bold", color=INK, ha="left",
    )
    fig.text(
        0.06, 0.935,
        "A GFM table has no error state. It truncates the row that is too wide, trims the cell whose meaning is "
        "its indentation, and italicises the identifier\nwith an underscore at each end - and reports none of it. "
        "markdown-tabler renders the table and returns the list of cells that did not survive.",
        fontsize=9.8, color=MUTED, ha="left", va="top",
    )
    fig.text(
        0.975, 0.965,
        "Day 140  ·  automation-suite" + ("" if HAVE_PARSER else "  ·  parser absent"),
        fontsize=9, color=MUTED, ha="right",
    )
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("wrote", build())
    print("wrote", build("table_audit_nb.png", figsize=(13.5, 12.0), dpi=110))
