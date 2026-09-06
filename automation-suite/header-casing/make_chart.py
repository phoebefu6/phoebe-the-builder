"""Figures for the field-name audit. Every value is read from the engine.

`header_audit.png` - six panels, the README hero
`header_demo.png`  - two panels, the notebook's chart
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from headers import (
    CORPUS,
    CORPUS_BY_NAME,
    HOPS,
    LOOKUPS,
    PATHS,
    REGISTRY_NAMES,
    SEVERITY_OF_CODE,
    Verdict,
    audit_corpus,
    canonical_mismatches,
    deliver,
    environ_collisions,
    hpack_names,
    lookup_audit,
    turkish_breakage,
    verdict_counts,
    wire_cost,
)
from matplotlib.patches import Patch

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
    "preserved": OK,
    "renormalized": ADVISORY,
    "lossy": SILENT,
    "rejected": BLOCKING,
}
VERDICT_INDEX = {v.value: i for i, v in enumerate(Verdict)}


def _frame(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    ax.set_facecolor(PAPER)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=0)
    ax.set_title(title, color=INK, fontsize=10.5, fontweight="bold", loc="left",
                 pad=14 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=MUTED, fontsize=7.6,
                va="bottom")


# --- panels ----------------------------------------------------------------


def panel_matrix(ax: plt.Axes) -> None:
    """Every message down every path, coloured by verdict."""
    paths = list(PATHS)
    msgs = [m.name for m in CORPUS]
    grid = np.zeros((len(msgs), len(paths)))
    for i, mn in enumerate(msgs):
        for j, p in enumerate(paths):
            grid[i, j] = VERDICT_INDEX[deliver(CORPUS_BY_NAME[mn], p).verdict().value]
    cmap = matplotlib.colors.ListedColormap(
        [VERDICT_COLOR[v.value] for v in Verdict])
    ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=len(Verdict) - 0.5, aspect="auto")
    ax.set_xticks(range(len(paths)))
    ax.set_xticklabels(paths, rotation=45, ha="right", fontsize=6.6)
    ax.set_yticks(range(len(msgs)))
    ax.set_yticklabels(msgs, fontsize=6.8)
    for x in range(len(paths) + 1):
        ax.axvline(x - 0.5, color=PAPER, lw=1.2)
    for y in range(len(msgs) + 1):
        ax.axhline(y - 0.5, color=PAPER, lw=1.2)
    counts = verdict_counts()
    total = sum(counts.values())
    _frame(ax, "12 messages x 10 paths",
           f"{counts[Verdict.PRESERVED]} of {total} arrive spelled as sent")
    ax.legend(handles=[Patch(facecolor=VERDICT_COLOR[v.value], label=v.value)
                       for v in Verdict],
              loc="upper left", bbox_to_anchor=(0, -0.42), ncol=4, frameon=False,
              fontsize=7)


def panel_lookups(ax: plt.Axes) -> None:
    """Which style of lookup finds the field, per path."""
    m = CORPUS_BY_NAME["browser-get"]
    wanted = "Upgrade-Insecure-Requests"
    names = [n for n, _, _ in LOOKUPS]
    paths = list(PATHS)
    grid = np.zeros((len(paths), len(names)))
    for i, p in enumerate(paths):
        got = lookup_audit(deliver(m, p), wanted)
        for j, n in enumerate(names):
            grid[i, j] = 1 if got[n] else 0
    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(["#efece6", OK]),
              vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=6.4)
    ax.set_yticks(range(len(paths)))
    ax.set_yticklabels(paths, fontsize=6.8)
    for x in range(len(names) + 1):
        ax.axvline(x - 0.5, color=PAPER, lw=1.2)
    for y in range(len(paths) + 1):
        ax.axhline(y - 0.5, color=PAPER, lw=1.2)
    hits = grid.sum(axis=0).astype(int)
    best = names[int(np.argmax(hits))]
    _frame(ax, f"Reading {wanted!r} back",
           f"best lookup finds it on {hits.max()} of {len(paths)} paths: {best}")


def panel_searches(ax: plt.Axes) -> None:
    """The four exhaustive searches over the registry, as counts."""
    inside, outside = hpack_names()
    rows = [
        ("CGI variable collisions", len(environ_collisions())),
        ("canonical respells it", len(canonical_mismatches())),
        ("names with a capital I", len(turkish_breakage())),
        ("not in HPACK table", outside),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, vals, color=[SILENT, ADVISORY, BLOCKING, MUTED], height=0.6)
    ax.axvline(len(REGISTRY_NAMES), color=INK, lw=1, ls=":")
    ax.text(len(REGISTRY_NAMES), -0.75, f" all {len(REGISTRY_NAMES)} names",
            color=INK, fontsize=7, va="bottom")
    for i, v in enumerate(vals):
        ax.text(v + 1, i, str(v), va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.4)
    ax.set_xlim(0, len(REGISTRY_NAMES) + 12)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _frame(ax, "Exhaustive over the registry",
           "searched, not sampled - every name checked against every other")


def panel_severity(ax: plt.Axes) -> None:
    """Findings per hop, stacked by severity."""
    per_hop: Dict[str, Dict[str, int]] = {
        h.name: {"blocking": 0, "silent": 0, "advisory": 0} for h in HOPS}
    for d in audit_corpus().values():
        for e in d.events:
            sev = SEVERITY_OF_CODE.get(e.code, "advisory")
            per_hop[e.hop][sev] += 1
    hops = [h.name for h in HOPS]
    x = np.arange(len(hops))
    bottom = np.zeros(len(hops))
    for sev in ("blocking", "silent", "advisory"):
        vals = np.array([per_hop[h][sev] for h in hops], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=SEV_COLOR[sev], label=sev, width=0.62)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(hops, rotation=35, ha="right", fontsize=7)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    silent_total = sum(v["silent"] for v in per_hop.values())
    _frame(ax, "Who does what to the name",
           f"{silent_total} silent findings - the request still returns 200")


def panel_bytes(ax: plt.Axes) -> None:
    """HTTP/1.1 field-line bytes against modelled HPACK bytes."""
    names = [m.name for m in CORPUS]
    h1 = [wire_cost(m)[0] for m in CORPUS]
    h2 = [wire_cost(m)[1] for m in CORPUS]
    x = np.arange(len(names))
    ax.bar(x - 0.19, h1, width=0.36, color=MUTED, label="HTTP/1.1 lines")
    ax.bar(x + 0.19, h2, width=0.36, color=ADVISORY, label="HPACK (modelled)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6.6)
    ax.set_ylabel("bytes", fontsize=7.5, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    saved = 1 - sum(h2) / sum(h1)
    _frame(ax, "Lowercase is also a size",
           f"{saved:.0%} smaller across the corpus, because static-table names "
           f"cost one byte")


def panel_story(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.set_facecolor(PAPER)
    counts = verdict_counts()
    total = sum(counts.values())
    lossy = counts[Verdict.LOSSY]
    lines = [
        ("RFC 9110 5.1", "field names are case-insensitive", INK),
        ("RFC 9113 8.2.1", "in HTTP/2 they MUST be lowercase, or the", INK),
        ("", "message is malformed", INK),
        ("", "", INK),
        (f"{counts[Verdict.PRESERVED]}/{total}", "arrive spelled as they were sent", OK),
        (f"{counts[Verdict.RENORMALIZED]}/{total}",
         "arrive respelled - same meaning, new bytes", ADVISORY),
        (f"{lossy}/{total}", "arrive changed, with a 200 and no error", SILENT),
        (f"{counts[Verdict.REJECTED]}/{total}", "never arrive at all", BLOCKING),
        ("", "", INK),
        ("The rule", "a field name is an identity, not a string.", INK),
        ("", "Compare with ASCII case-folding. Never `==`.", INK),
    ]
    y = 0.96
    for tag, text, colour in lines:
        if tag:
            ax.text(0.02, y, tag, fontsize=9, fontweight="bold", color=colour,
                    transform=ax.transAxes, va="top")
        ax.text(0.30, y, text, fontsize=8.4, color=colour, transform=ax.transAxes,
                va="top")
        y -= 0.088
    _frame(ax, "What the audit returns", "")
    ax.set_xticks([])
    ax.set_yticks([])


# --- figures ---------------------------------------------------------------


def audit_figure(path: str = "header_audit.png") -> str:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6))
    fig.patch.set_facecolor(PAPER)
    panel_matrix(axes[0][0])
    panel_lookups(axes[0][1])
    panel_searches(axes[0][2])
    panel_severity(axes[1][0])
    panel_bytes(axes[1][1])
    panel_story(axes[1][2])
    fig.suptitle("A field name is not a string", x=0.008, y=0.988, ha="left",
                 fontsize=15, fontweight="bold", color=INK)
    fig.text(0.008, 0.945,
             f"{len(CORPUS)} messages down {len(PATHS)} real delivery paths, "
             f"{len(REGISTRY_NAMES)} field names searched exhaustively",
             ha="left", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(path, dpi=170, facecolor=PAPER)
    plt.close(fig)
    return path


def demo_figure(path: str = "header_demo.png") -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))
    fig.patch.set_facecolor(PAPER)
    panel_matrix(axes[0])
    panel_lookups(axes[1])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160, facecolor=PAPER)
    plt.close(fig)
    return path


def main() -> List[str]:
    out = [audit_figure(), demo_figure()]
    for p in out:
        print(f"wrote {p}")
    return out


if __name__ == "__main__":
    main()
