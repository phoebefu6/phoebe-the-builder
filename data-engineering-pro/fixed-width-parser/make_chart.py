"""Six-panel audit figure. ``python3 make_chart.py`` writes both PNGs."""

from __future__ import annotations

from decimal import Decimal
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evidence import (
    _magnitude_only,
    _raw_column,
    _strip_to_digits,
    exp_byte_vs_char,
    exp_framing,
    exp_index_base,
)
from fwf import (
    BALANCE_SPEC,
    CUSTOMER_SPEC,
    build_balance_file,
    build_customer_file,
    frame_records,
    parse,
    parse_naive,
)
from matplotlib.patches import Patch

INK = "#1c1c1c"
MUTED = "#8a8a8a"
GOOD = "#2f6f52"
BAD = "#b23a3a"
WARM = "#c98a2b"
COOL = "#3a6ea5"
PAPER = "#faf8f4"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)


# --------------------------------------------------------------------------


def panel_encoding(ax: plt.Axes) -> None:
    """Which cells a character-slicing reader gets wrong, row by row."""
    data = build_customer_file()
    right = parse(data, CUSTOMER_SPEC).rows
    naive = parse_naive(data, CUSTOMER_SPEC)
    cols = ["country", "status", "qty", "list_price"]

    grid = np.zeros((len(right), len(cols)))
    for i in range(len(right)):
        for j, c in enumerate(cols):
            cv, nv = right[i][c], naive[i][c]
            if c == "list_price":
                ok = nv is not None and cv is not None and int(cv * 100) == nv
            else:
                ok = str(cv) == str(nv)
            grid[i, j] = 0 if ok else 1

    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap([GOOD, BAD]), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=20, ha="right", fontsize=7)
    ax.set_yticks(range(len(right)))
    # Rows are labelled by cust_id, not by name: the names are the point of the
    # panel and half of them have no glyph in a portable matplotlib font, so
    # drawing them here would render as boxes on any machine without a CJK face.
    labels = [
        f"{'*' if r['name'] and not r['name'].isascii() else ' '} {r['cust_id']}" for r in right
    ]
    ax.set_yticklabels(labels, fontsize=6.5, fontfamily="monospace")
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(right), 1), minor=True)
    ax.grid(which="minor", color=PAPER, lw=1.1)
    ax.set_title("A  character offsets vs byte offsets", fontsize=9.5, color=INK, loc="left", pad=16)
    ax.text(
        0.0,
        -0.19,
        "* record contains a character wider than one byte (names omitted:\nno portable font covers them). Rows 1/2 and 3/4 are the same\n"
        "customer, one accent apart; the reader gets one right, one wrong.",
        transform=ax.transAxes,
        fontsize=6.8,
        color=MUTED,
        va="top",
    )
    ax.legend(
        handles=[Patch(color=GOOD, label="agrees"), Patch(color=BAD, label="wrong")],
        fontsize=6.5,
        loc="lower right",
        bbox_to_anchor=(1.02, 1.005),
        ncol=2,
        frameon=False,
    )
    ax.tick_params(length=0)


def panel_index_base(ax: plt.Axes) -> None:
    r = exp_index_base(verbose=False)
    vals = [float(r["price_correct"]), float(r["price_shifted"])]
    bars = ax.bar(["1-indexed\n(correct)", "0-indexed\n(same numbers)"], vals, color=[GOOD, BAD], width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("sum(list_price)", fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:,.0f}", ha="center", fontsize=7.5, color=INK)
    ax.set_title("B  one-byte shift, no exception", fontsize=9.5, color=INK, loc="left")
    ax.text(
        0.0,
        -0.28,
        f"sum(qty) is exactly {r['qty_shifted'] / r['qty_correct']:.0f}x too large.\n"
        f"Declaring the record length catches it; nothing else does.",
        transform=ax.transAxes,
        fontsize=6.8,
        color=MUTED,
        va="top",
    )
    _style(ax)


def panel_overpunch(ax: plt.Axes) -> None:
    data = build_customer_file()
    right = parse(data, CUSTOMER_SPEC)
    raw = _raw_column(data, CUSTOMER_SPEC, "net_amount")
    correct = float(right.total("net_amount"))
    fix1 = float(sum((_strip_to_digits(r.decode("latin-1")) for r in raw), Decimal(0)))
    fix2 = float(sum((_magnitude_only(r.decode("latin-1")) for r in raw), Decimal(0)))

    names = ["fix #1\nstrip non-digits", "correct\nsign honoured", "fix #2\nsign dropped"]
    vals = [fix1, correct, fix2]
    bars = ax.bar(names, vals, color=[WARM, GOOD, BAD], width=0.55)
    for b, v in zip(bars, vals):
        pct = (v - correct) / correct * 100
        tag = "" if abs(pct) < 1e-9 else f"{pct:+.0f}%"
        ax.text(b.get_x() + b.get_width() / 2, v + correct * 0.03, f"{v:,.0f}\n{tag}",
                ha="center", fontsize=7.2, color=INK)
    ax.set_ylim(0, max(vals) * 1.32)
    ax.set_ylabel("total net_amount", fontsize=8)
    ax.set_title("C  the sign is a letter inside the last digit", fontsize=9.5, color=INK, loc="left")
    ax.text(
        0.0,
        -0.30,
        "Both repairs are wrong, in opposite directions. Fix #2 is the\n"
        "dangerous one: every magnitude is right, refunds just flip.",
        transform=ax.transAxes,
        fontsize=6.8,
        color=MUTED,
        va="top",
    )
    _style(ax)


def panel_implied(ax: plt.Axes) -> None:
    data = build_customer_file()
    right = parse(data, CUSTOMER_SPEC)
    raw = _raw_column(data, CUSTOMER_SPEC, "list_price")
    vals = [float(right.total("list_price")), float(sum(int(r) for r in raw))]
    bars = ax.bar(["PIC 9(7)V99", "read as int"], vals, color=[GOOD, BAD], width=0.5)
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.2, f"{v:,.0f}", ha="center", fontsize=7.5, color=INK)
    ax.set_ylabel("sum(list_price)", fontsize=8)
    ax.set_title("D  implied decimal: exactly 100x", fontsize=9.5, color=INK, loc="left")
    ax.text(
        0.0,
        -0.28,
        "The scale is not in the file. Both readings are positive\n"
        "integers of the right width and pass every dtype check.",
        transform=ax.transAxes,
        fontsize=6.8,
        color=MUTED,
        va="top",
    )
    _style(ax)


def panel_framing(ax: plt.Axes) -> None:
    """Where the separator bytes actually sit inside the packed file."""
    data = build_balance_file()
    recs, _, _ = frame_records(data, BALANCE_SPEC, "block")
    n, w = len(recs), BALANCE_SPEC.record_length
    grid = np.zeros((n, w))
    for i, rec in enumerate(recs):
        for j, byte in enumerate(rec):
            if byte in (0x0A, 0x0D):
                grid[i, j] = 2
            elif byte < 0x20 or byte > 0x7E:
                grid[i, j] = 1
    cmap = matplotlib.colors.ListedColormap(["#e8e4dc", COOL, BAD])
    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=2, interpolation="nearest")
    for name in ("balance", "adjustment"):
        s, e = BALANCE_SPEC.slice_of(name)
        ax.axvline(s - 0.5, color=INK, lw=0.7, ls=":")
        ax.axvline(e - 0.5, color=INK, lw=0.7, ls=":")
        ax.text((s + e) / 2 - 0.5, -0.85, name, ha="center", fontsize=6.2, color=INK)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[:10].decode("latin-1") for r in recs], fontsize=6.3, fontfamily="monospace")
    ax.set_xlabel("byte offset in record", fontsize=8)
    ax.set_title(
        "E  a record separator that is also a number", fontsize=9.5, color=INK, loc="left", pad=32
    )
    ax.legend(
        handles=[
            Patch(color="#e8e4dc", label="printable"),
            Patch(color=COOL, label="packed"),
            Patch(color=BAD, label="0x0A / 0x0D"),
        ],
        fontsize=6.3,
        loc="lower right",
        bbox_to_anchor=(1.02, 1.135),
        ncol=3,
        frameon=False,
    )
    fr = exp_framing(verbose=False)
    ax.text(
        0.0,
        -0.30,
        f"Line framing recovers {fr['n_lines']} record of {fr['n_block']}, with "
        f"{fr['errors_lines']} parse errors.\n"
        "COMP-3 sign nibble 0xD + final digit 0 = carriage return.",
        transform=ax.transAxes,
        fontsize=6.8,
        color=MUTED,
        va="top",
    )
    ax.tick_params(length=0)


def panel_ledger(ax: plt.Axes) -> None:
    ax.axis("off")
    a = exp_byte_vs_char(verbose=False)
    rows: List[tuple] = [
        ("character offsets", f"{len(a['wrong_rows'])}/{a['records']} rows scrambled"),
        ("index base off by one", "9.6x totals"),
        ("overpunch sign dropped", "+36,012 revenue (+9.2%)"),
        ("implied decimal ignored", "100x totals"),
        ("line framing on RECFM=F", "1 record of 6"),
        ("packed read as text", "12 amounts to mojibake"),
    ]
    ax.set_title("F  damage ledger - none of these raise", fontsize=9.5, color=INK, loc="left")
    y = 0.86
    ax.text(0.02, 0.97, "failure mode", fontsize=7, color=MUTED, transform=ax.transAxes)
    ax.text(0.56, 0.97, "effect on this sample", fontsize=7, color=MUTED, transform=ax.transAxes)
    ax.plot([0.02, 0.98], [0.93, 0.93], color=MUTED, lw=0.6, transform=ax.transAxes)
    for name, effect in rows:
        ax.text(0.02, y, name, fontsize=7.6, color=INK, transform=ax.transAxes)
        ax.text(0.56, y, effect, fontsize=7.6, color=BAD, transform=ax.transAxes)
        y -= 0.115
    ax.plot([0.02, 0.98], [y + 0.06, y + 0.06], color=MUTED, lw=0.6, transform=ax.transAxes)
    ax.text(
        0.02,
        y - 0.04,
        "Every one produces a plausible answer. Four produce a stable\n"
        "one, so month-on-month reconciliation agrees with itself.\n"
        "audit() runs on the bytes, before the load, and names all six.",
        fontsize=6.9,
        color=MUTED,
        transform=ax.transAxes,
        va="top",
    )


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5), facecolor=PAPER)
    fig.suptitle(
        "Fixed-width is a byte format. Six ways a character-shaped reader gets it wrong.",
        fontsize=13,
        color=INK,
        x=0.012,
        ha="left",
        y=0.985,
    )
    panel_encoding(axes[0][0])
    panel_index_base(axes[0][1])
    panel_overpunch(axes[0][2])
    panel_implied(axes[1][0])
    panel_framing(axes[1][1])
    panel_ledger(axes[1][2])
    fig.tight_layout(rect=(0, 0.02, 1, 0.955))
    fig.subplots_adjust(hspace=0.62, wspace=0.30)
    fig.savefig("fwf_audit.png", dpi=300, facecolor=PAPER)
    fig.savefig("fwf_audit_nb.png", dpi=110, facecolor=PAPER)
    print("wrote fwf_audit.png (300 dpi) and fwf_audit_nb.png (110 dpi)")


if __name__ == "__main__":
    main()
