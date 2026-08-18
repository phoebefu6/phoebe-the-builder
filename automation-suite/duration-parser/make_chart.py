"""Figures for the duration audit. Every value is read from the engine.

`duration_audit.png` - six panels, the README hero
`duration_demo.png`  - two panels, the notebook's chart
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from durations import (
    DAY24,
    DEFAULT_ANCHORS,
    GRAMMARS,
    PARSERS,
    REFERENCE_ANCHOR,
    SEVERITY_OF,
    Verdict,
    audit,
    audit_corpus,
    parse_iso,
)

INK = "#141414"
MUTED = "#8a8a8a"
GRID = "#e4e2dd"
PAPER = "#faf8f4"
BLOCKING = "#c0392b"
SILENT = "#d98324"
ADVISORY = "#4a7c8c"
OK = "#4b7f52"
SEV_COLOR = {"blocking": BLOCKING, "silent": SILENT, "advisory": ADVISORY}
VERDICT_COLOR = {
    "exact": OK,
    "anchored": ADVISORY,
    "ambiguous": SILENT,
    "rejected": BLOCKING,
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
})

REP = audit_corpus()


def title(ax, n: int, text: str, sub: str = "") -> None:
    ax.set_title(f"{n}.  {text}", loc="left", fontsize=10, fontweight="bold",
                 pad=22 if sub else 6)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=7.6, color=MUTED, va="bottom")


# ---------------------------------------------------------------------------


def panel_spread(ax) -> None:
    rows = [(a.text, a.spread_ratio or 1.0, a.verdict.value) for a in REP.audits
            if a.verdict is not Verdict.REJECTED]
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))
    for yi, (_, ratio, verdict) in zip(y, rows):
        if ratio > 1.001:
            ax.barh(yi, ratio - 1.0, left=1.0, color=VERDICT_COLOR[verdict], height=0.68)
        else:
            # A unanimous row has no bar to draw on a ratio axis, so it gets a
            # mark at 1.0 rather than disappearing.
            ax.plot([1.0], [yi], marker="o", ms=4.5, color=VERDICT_COLOR[verdict], lw=0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7, fontfamily="DejaVu Sans Mono")
    ax.set_xscale("log")
    ax.set_xlim(0.9, 2e5)
    ax.axvline(1.0, color=INK, lw=0.8)
    ax.set_xlabel("highest reading / lowest reading  (log)")
    for yi, (_, ratio, _) in zip(y, rows):
        if ratio > 1.001:
            ax.text(ratio * 1.25, yi, f"{ratio:,.0f}x" if ratio >= 10 else f"{ratio:.2f}x",
                    va="center", fontsize=6.6, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    title(ax, 1, "How far apart two correct parsers land",
          "one bar per string; 1x means every parser that accepted it agreed")
    ax.legend(handles=[Patch(facecolor=VERDICT_COLOR[k], label=k) for k in
                       ("exact", "anchored", "ambiguous")],
              loc="lower right", frameon=False, fontsize=7)


def panel_grammars(ax) -> None:
    names = [g.name for g in GRAMMARS]
    acc = [REP.accepted_by[n] for n in names]
    wrong = []
    for n in names:
        w = 0
        for a in REP.audits:
            mine = next((r for r in a.accepted if r.grammar == n), None)
            if mine is None:
                continue
            others = [r for r in a.accepted if r.grammar != n]
            if any(round(o.resolve(REFERENCE_ANCHOR), 6) != round(mine.resolve(REFERENCE_ANCHOR), 6)
                   for o in others):
                w += 1
        wrong.append(w)
    x = np.arange(len(names))
    ax.bar(x, acc, color=ADVISORY, width=0.66, label="accepted")
    ax.bar(x, wrong, color=SILENT, width=0.66, label="and read differently by a peer")
    ax.axhline(REP.total, color=INK, lw=0.9, ls=(0, (4, 2)))
    ax.text(len(names) - 0.4, REP.total + 0.4, f"all {REP.total} strings", ha="right",
            fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("strings")
    ax.set_ylim(0, REP.total + 3)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    title(ax, 2, "Pick one library and this is your coverage",
          "the orange part returned a number another parser contradicts")


def panel_pairs(ax) -> None:
    names = [g.name for g in GRAMMARS]
    n = len(names)
    m = np.full((n, n), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            key = tuple(sorted((a, b)))
            m[i, j] = REP.disagreements.get(key, 0)
    ax.imshow(m, cmap="YlOrRd", vmin=0, vmax=max(REP.disagreements.values()))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", color=MUTED, fontsize=7)
            elif m[i, j] > 0:
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] > 4 else INK)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=PAPER, lw=1.4)
    ax.tick_params(which="minor", length=0)
    title(ax, 3, "Which pairs return different numbers",
          "blank means they never both accepted the same string")


def panel_anchor(ax) -> None:
    items = [("P1D", "iso8601"), ("P1W", "iso8601"), ("P1M", "iso8601"), ("P1Y", "iso8601")]
    labels, lows, highs = [], [], []
    for text, _ in items:
        r = parse_iso(text)
        vals = [r.resolve(a) for a in DEFAULT_ANCHORS]
        labels.append(text)
        lows.append(min(vals) / DAY24)
        highs.append(max(vals) / DAY24)
    y = np.arange(len(labels))
    for yi, lo, hi in zip(y, lows, highs):
        ax.plot([lo, hi], [yi, yi], color=ADVISORY, lw=7, solid_capstyle="butt")
        ax.text(lo * 0.88, yi + 0.02, f"{lo:.4g}d - {hi:.4g}d", fontsize=6.8, color=INK,
                ha="right", va="center")
    fixed = [("prometheus 1d", PARSERS["prometheus"]("1d").exact_s, 0),
             ("systemd 1w", PARSERS["systemd"]("1w").exact_s, 1),
             ("systemd 1M", PARSERS["systemd"]("1M").exact_s, 2),
             ("prometheus 1y", PARSERS["prometheus"]("1y").exact_s, 3),
             ("jira 1d", PARSERS["jira"]("1d").exact_s, 0)]
    for k, (label, seconds, yi) in enumerate(fixed):
        ax.plot([seconds / DAY24], [yi], marker="D", ms=5.5, color=BLOCKING, zorder=5)
        ax.text(seconds / DAY24, yi + (0.30 if k % 2 else -0.34), label, fontsize=6.6,
                color=BLOCKING, ha="center",
                va="bottom" if k % 2 else "top")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontfamily="DejaVu Sans Mono", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("elapsed length in days (log)")
    ax.set_xlim(0.18, 900)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    title(ax, 4, "Calendar units have a range, not a value",
          "bars: ISO reading over 17 anchors. diamonds: the fixed number other grammars substitute")


def panel_findings(ax) -> None:
    rows = sorted(((c, n, SEVERITY_OF[c]) for c, n in REP.finding_counts.items()),
                  key=lambda r: (["blocking", "silent", "advisory"].index(r[2]), -r[1]))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[SEV_COLOR[r[2]] for r in rows], height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.4, fontfamily="DejaVu Sans Mono")
    ax.invert_yaxis()
    ax.set_xlabel(f"strings in the corpus of {REP.total} that trip it")
    for yi, (_, n, _) in zip(y, rows):
        ax.text(n + 0.3, yi, str(n), va="center", fontsize=6.6, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=SEV_COLOR[k], label=k) for k in
                       ("blocking", "silent", "advisory")],
              loc="lower right", frameon=False, fontsize=7)
    title(ax, 5, "Eighteen mechanisms, every one with evidence",
          "silent = every parser involved returned successfully")


def panel_fan(ax) -> None:
    picks = ["90", "0.5", "1h30", "1:30", "1d", "1w", "3w 2d"]
    ax.set_xscale("log")
    for yi, text in enumerate(picks):
        a = audit(text)
        by_value: Dict[float, List[str]] = {}
        for r in a.accepted:
            v = round(abs(r.resolve(REFERENCE_ANCHOR)) or 1e-3, 6)
            by_value.setdefault(v, []).append(r.grammar)
        for k, (v, names) in enumerate(sorted(by_value.items())):
            ax.plot([v], [yi], marker="o", ms=6, mfc="none", color=INK, lw=0)
            # Alternate above/below: two readings can be close enough on a log
            # axis that one row of labels would sit on top of itself.
            off, va = (0.20, "top") if k % 2 == 0 else (-0.16, "bottom")
            ax.text(v, yi + off, " / ".join(sorted(names)), fontsize=6.1, ha="center",
                    va=va, color=MUTED)
        lo, hi = a.min_s, a.max_s
        if hi > lo:
            ax.plot([lo, hi], [yi, yi], color=SILENT, lw=1.4, zorder=0)
            r = hi / lo
            ax.text(hi * 1.6, yi, f"{r:,.0f}x" if r >= 10 else f"{r:.2f}x",
                    fontsize=7, va="center", color=BLOCKING)
    ax.set_yticks(range(len(picks)))
    ax.set_yticklabels(picks, fontfamily="DejaVu Sans Mono", fontsize=8)
    ax.invert_yaxis()
    for seconds, label in ((60, "1min"), (3600, "1h"), (DAY24, "1d"), (30 * DAY24, "30d")):
        ax.axvline(seconds, color=GRID, lw=0.8, zorder=0)
        ax.text(seconds, 0.985, label, fontsize=6.4, color=MUTED, ha="center", va="top",
                transform=ax.get_xaxis_transform())
    ax.set_xlim(0.2, 1.2e8)
    ax.set_ylim(len(picks) - 0.4, -0.85)
    ax.set_xlabel("seconds (log)")
    title(ax, 6, "The same characters, read by whoever gets there first",
          "one circle per parser that accepted the string")


def build_audit() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 15.6))
    panel_spread(axes[0][0])
    panel_grammars(axes[0][1])
    panel_pairs(axes[1][0])
    panel_anchor(axes[1][1])
    panel_findings(axes[2][0])
    panel_fan(axes[2][1])
    fig.suptitle(
        "A duration string is not a number: 28 strings, 8 conforming parsers",
        x=0.012, y=0.995, ha="left", fontsize=14, fontweight="bold",
    )
    fig.text(
        0.012, 0.9755,
        f"Day 147 - Duration Parser.  {REP.verdicts['exact']} exact, "
        f"{REP.verdicts['anchored']} anchored, {REP.verdicts['ambiguous']} ambiguous, "
        f"{REP.verdicts['rejected']} rejected.  No parser reads more than "
        f"{max(REP.accepted_by.values())} of {REP.total}.",
        fontsize=9, color=MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.968))
    fig.savefig("duration_audit.png", dpi=170)
    print("wrote duration_audit.png")


def build_demo() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    panel_fan(axes[0])
    panel_anchor(axes[1])
    fig.suptitle("Ambiguity, and the anchored range underneath it",
                 x=0.012, y=0.99, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("duration_demo.png", dpi=165)
    print("wrote duration_demo.png")


if __name__ == "__main__":
    build_audit()
    build_demo()
