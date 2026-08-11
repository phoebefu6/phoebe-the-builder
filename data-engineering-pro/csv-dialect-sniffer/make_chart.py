"""Six-panel audit figure. Every number is computed from sniff.py, not typed in.

    python3 make_chart.py            -> sniff_audit.png
    python3 make_chart.py nb         -> sniff_audit_nb.png (notebook variant)
"""

from __future__ import annotations

import sys
from typing import Dict, List

import matplotlib
if __name__ == "__main__":  # importing this from a notebook must not steal the backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sniff

INK = "#1d2330"
MUTED = "#8b93a7"
OK = "#2f8f6b"
WARN = "#d98324"
BAD = "#b8434f"
GRID = "#e3e6ee"

FILES = sniff.sample_files()
TEXT = {n: r.decode("utf-8", errors="replace") for n, r in FILES.items()}


def style(ax: plt.Axes, title: str, sub: str = "", integer_y: bool = False) -> None:
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=30)
    if sub:
        ax.text(0, 1.018, sub, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    if integer_y:
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(labelsize=8, colors=MUTED, length=3)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #


def panel_candidates(ax: plt.Axes) -> None:
    v = sniff.classify_delimiter(TEXT["sensor.csv"])
    shapes = [s for s in v.all_shapes if s.quotechar == '"']
    shapes.sort(key=lambda s: -s.modal)
    labels = [s.label.split(" /")[0] for s in shapes]
    vals = [s.modal for s in shapes]
    colors = [BAD if s.viable else MUTED for s in shapes]
    ax.bar(range(len(vals)), vals, color=colors, width=0.62)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("fields per record", fontsize=8, color=MUTED)
    for i, s in enumerate(shapes):
        if s.viable:
            ax.text(i, s.modal + 0.12, "clean", ha="center", fontsize=7.5,
                    color=BAD, fontweight="bold")
    style(ax, "1  sensor.csv: two clean parses",
          "red = every record the same width. 3 columns or 4, and no exception either way",
          integer_y=True)


def panel_header_tie(ax: plt.Axes) -> None:
    names = ["sensor.csv\n(no header)", "sales_eu.csv\n(header row)"]
    counts = []
    for n in ("sensor.csv", "sales_eu.csv"):
        counts.append(len(sniff.classify_delimiter(TEXT[n]).viable))
    colors = [BAD if c > 1 else OK for c in counts]
    ax.bar(range(2), counts, color=colors, width=0.5)
    ax.set_xticks(range(2))
    ax.set_xticklabels(names)
    ax.set_yticks([0, 1, 2])
    ax.set_ylabel("viable dialects", fontsize=8, color=MUTED)
    for i, c in enumerate(counts):
        ax.text(i, c + 0.05, "contested" if c > 1 else "unambiguous", ha="center",
                fontsize=7.5, color=colors[i], fontweight="bold")
    style(ax, "2  A header row decides it",
          "same data, 24 more bytes: the comma parse becomes ragged and drops out")


def panel_encoding(ax: plt.Axes) -> None:
    order = ["cp1252.csv", "utf8_umlaut.csv", "bom.csv", "sensor.csv"]
    decodes, credible = [], []
    for n in order:
        r = sniff.probe_encoding(FILES[n])
        decodes.append(len(r.survived))
        credible.append(len(r.plausible))
    x = np.arange(len(order))
    ax.bar(x - 0.19, decodes, width=0.36, color=MUTED, label="decode without raising")
    ax.bar(x + 0.19, credible, width=0.36, color=OK, label="credible after the C1 test")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(".csv", "") for n in order], rotation=20, ha="right")
    ax.set_ylabel("encodings (of 6 tried)", fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    style(ax, "3  Decoding proves less than it looks",
          "latin-1 cannot fail on any input, so its success is never evidence")


def panel_c1_coverage(ax: plt.Axes) -> None:
    letters = [chr(c) for c in range(0xC0, 0x180)]
    caught = sum(1 for c in letters
                 if any(0x80 <= b <= 0x9F for b in c.encode("utf-8")[1:]))
    missed = len(letters) - caught
    ax.barh([1, 0], [caught, missed], color=[OK, BAD], height=0.5)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["caught\n(U+00C0-DF, upper case)", "missed\n(U+00E0-FF, lower case)"],
                       fontsize=8)
    ax.set_xlabel("accented Latin letters", fontsize=8, color=MUTED)
    ax.text(caught - 4, 1, str(caught), va="center", ha="right", color="white",
            fontsize=9, fontweight="bold")
    ax.text(missed - 4, 0, str(missed), va="center", ha="right", color="white",
            fontsize=9, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.7)
    style(ax, "4  The C1 test is exactly half blind",
          "second utf-8 byte is 0x80|(cp & 0x3f), so only half land in 0x80-0x9f")
    ax.grid(axis="y", visible=False)


def panel_sample_size(ax: plt.Axes) -> None:
    text = TEXT["late.csv"]
    sizes = [64, 128, 256, 512, 1024, len(text)]
    cols: List[float] = []
    labels: List[str] = []
    for n in sizes:
        pick = sniff.sniffer_says(text[:n])
        cols.append(sniff.shape_of(text[:n], pick).modal if pick else 0)
        labels.append("{0}\n{1}".format(n, repr(pick) if pick else "raises"))
    colors = [OK if c == 3 else BAD for c in cols]
    ax.bar(range(len(cols)), cols, color=colors, width=0.6)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("columns Sniffer implies", fontsize=8, color=MUTED)
    ax.axhline(3, color=INK, lw=0.9, ls=":")
    ax.text(len(cols) - 0.4, 3.08, "correct", fontsize=7, color=INK, ha="right")
    style(ax, "5  Sniffing a prefix, byte budget vs answer",
          "at 128 B it returns 'i', a letter from the word Widget, and does not raise",
          integer_y=True)


def panel_line_counts(ax: plt.Axes) -> None:
    """How many records does this file have? Three answers per file."""
    order = ["quoted.csv", "mac.csv", "dutch.csv", "sales_eu.csv"]
    naive, best, alt = [], [], []
    for n in order:
        v = sniff.classify_delimiter(TEXT[n])
        p = v.preferred
        d = p.delimiter if p else ","
        q = p.quotechar if p else '"'
        naive.append(sniff.probe_terminator(TEXT[n], d, q).naive_lines)
        best.append(p.records if p else 0)
        others = [s for s in v.viable if s is not p]
        alt.append(min(s.records for s in others) if others else (p.records if p else 0))
    x = np.arange(len(order))
    ax.bar(x - 0.26, naive, width=0.25, color=WARN, label="str.split('\\n')")
    ax.bar(x, best, width=0.25, color=INK, label="csv, preferred dialect")
    ax.bar(x + 0.26, alt, width=0.25, color=BAD, label="csv, runner-up dialect")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(".csv", "") for n in order], rotation=20, ha="right")
    ax.set_ylabel("records", fontsize=8, color=MUTED)
    ax.set_ylim(0, max(naive + best + alt) * 1.22)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), columnspacing=1.2, handlelength=1.2)
    for i in range(len(order)):
        if naive[i] != best[i]:
            ax.text(i - 0.26, naive[i] + 0.08, "{0:+d}".format(naive[i] - best[i]),
                    ha="center", fontsize=7.5, color=WARN, fontweight="bold")
        if alt[i] != best[i]:
            ax.text(i + 0.26, alt[i] + 0.08, "{0:+d}".format(alt[i] - best[i]),
                    ha="center", fontsize=7.5, color=BAD, fontweight="bold")
    style(ax, "6  How many records does this file have?",
          "naive counting is wrong both ways; a wrong quotechar deletes a row at full width",
          integer_y=True)


def main() -> None:
    nb = len(sys.argv) > 1 and sys.argv[1] == "nb"
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
    fig.patch.set_facecolor("white")
    panels = [panel_candidates, panel_header_tie, panel_encoding,
              panel_c1_coverage, panel_sample_size, panel_line_counts]
    for ax, fn in zip(axes.ravel(), panels):
        ax.set_facecolor("white")
        fn(ax)

    undecided = [n for n, r in FILES.items() if not sniff.audit(r, n).decided]
    fig.suptitle("CSV dialect detection: what the bytes decide, and what they do not",
                 fontsize=15, fontweight="bold", color=INK, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.947,
             "{0} sample files, {1} of them not determined by their own contents ({2}). "
             "Every value computed by sniff.py; standard library only.".format(
                 len(FILES), len(undecided), ", ".join(sorted(undecided))),
             fontsize=9.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.subplots_adjust(hspace=0.52, wspace=0.28)
    out = "sniff_audit_nb.png" if nb else "sniff_audit.png"
    fig.savefig(out, dpi=150 if nb else 200, facecolor="white")
    print("wrote " + out)


if __name__ == "__main__":
    main()
