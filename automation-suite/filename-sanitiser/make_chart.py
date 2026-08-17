"""Six-panel audit figure. Run: python3 make_chart.py -> sanitise_audit.png

One panel per mechanism, in the order the README argues them.
"""

from __future__ import annotations

from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import sanitise as S

# A restrained palette: one hue per outcome, reused in every panel so the
# reader learns it once.
DELIVERED = "#2a7f7f"
OVERWRITTEN = "#d98324"
REJECTED = "#8c8c96"
ACCENT = "#b3402f"
INK = "#1d1d21"
GRID = "#dcdce2"
PAPER = "#faf9f7"

NAMES = S.SAMPLE_NAMES
WIN_DEST = r"C:\data"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.edgecolor": "#b9b9c2",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": "#5c5c66",
        "ytick.color": "#5c5c66",
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
    }
)


def tidy(ax: plt.Axes, grid_axis: str = "x") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------


def panel1_frontier(ax: plt.Axes) -> None:
    """Every source lands in one of three buckets. The bars all sum to 42."""
    rows = S.compare(NAMES, S.WINDOWS, WIN_DEST)
    rows = sorted(rows, key=lambda r: r.delivered)
    y = range(len(rows))
    labels = [r.sanitiser for r in rows]

    d = [r.delivered for r in rows]
    o = [r.overwritten for r in rows]
    j = [r.rejected for r in rows]

    ax.barh(y, d, color=DELIVERED, label="delivered", zorder=3)
    ax.barh(y, o, left=d, color=OVERWRITTEN, label="overwritten", zorder=3)
    ax.barh(y, j, left=[a + b for a, b in zip(d, o)], color=REJECTED,
            label="rejected", zorder=3)

    for i, r in enumerate(rows):
        ax.text(r.delivered / 2, i, str(r.delivered), va="center", ha="center",
                color="white", fontsize=8, fontweight="bold", zorder=4)

    nothing = next(r for r in rows if r.sanitiser == "passthrough")
    ax.axvline(nothing.delivered, color=ACCENT, lw=1.4, ls="--", zorder=5)
    ax.text(nothing.delivered + 0.6, -0.62, "doing nothing", color=ACCENT,
            fontsize=7.5, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, len(NAMES))
    ax.set_xlabel(f"of {len(NAMES)} source names")
    ax.set_title("1  Validity is bought with collisions", loc="left")
    # Below the axis: inside the plot it lands on the bars whichever corner it
    # is given, because every bar spans the full width by construction.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3,
              frameon=False, fontsize=7.5)
    tidy(ax)


def panel2_target_dependence(ax: plt.Axes) -> None:
    """The same function, four targets, opposite sign."""
    targets = [S.WINDOWS, S.MACOS_APFS, S.LINUX_EXT4, S.OBJECT_STORE]
    nothing, sanitised = [], []
    for p in targets:
        dest = WIN_DEST if p.name.startswith("windows") else "/data"
        nothing.append(S.audit(NAMES, p, dest, "passthrough").delivered)
        sanitised.append(S.audit(NAMES, p, dest, "pathvalidate").delivered)

    x = range(len(targets))
    w = 0.38
    ax.bar([i - w / 2 for i in x], nothing, w, color=REJECTED,
           label="no sanitiser", zorder=3)
    ax.bar([i + w / 2 for i in x], sanitised, w, color=DELIVERED,
           label="pathvalidate", zorder=3)

    for i, (a, b) in enumerate(zip(nothing, sanitised)):
        delta = b - a
        ax.annotate(
            f"{delta:+d}",
            xy=(i, max(a, b) + 1.2),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=DELIVERED if delta > 0 else ACCENT,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([p.name.replace("-", "\n") for p in targets], fontsize=7.5)
    ax.set_ylim(0, len(NAMES) + 5)
    ax.set_ylabel("names delivered")
    ax.set_title("2  A sanitiser is only right for one target", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    tidy(ax, "y")


def panel3_length_units(ax: plt.Axes) -> None:
    """Characters, UTF-8 bytes and UTF-16 code units, against one limit."""
    probes = [
        ("90 CJK", "季度销售报告" * 15 + ".csv"),
        ("70 emoji", "\U0001f4c8" * 70 + ".png"),
        ("200 é", "é" * 200 + ".csv"),
        ("300 ASCII", "a" * 300 + ".csv"),
    ]
    x = range(len(probes))
    w = 0.27
    chars = [len(n) for _, n in probes]
    b8 = [len(n.encode("utf-8")) for _, n in probes]
    cu = [len(n.encode("utf-16-le")) // 2 for _, n in probes]

    ax.bar([i - w for i in x], chars, w, color="#9aa7b1", label="characters", zorder=3)
    ax.bar(list(x), cu, w, color=DELIVERED, label="UTF-16 code units", zorder=3)
    ax.bar([i + w for i in x], b8, w, color=OVERWRITTEN, label="UTF-8 bytes", zorder=3)

    ax.axhline(255, color=ACCENT, lw=1.4, zorder=5)
    ax.text(-0.42, 264, "NAME_MAX 255", color=ACCENT, fontsize=7,
            fontweight="bold", ha="left")

    ax.set_xticks(list(x))
    ax.set_xticklabels([lab for lab, _ in probes], fontsize=7.5)
    ax.set_ylabel("length")
    ax.set_ylim(0, 440)
    ax.set_title("3  The limit is not in characters", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    tidy(ax, "y")


def panel4_case_tables(ax: plt.Axes) -> None:
    """Three fold models against what a volume actually does."""
    pairs = [
        ("Straße / STRASSE", "Straße.txt", "STRASSE.txt", False),
        ("ΣΙΣΥΦΟΣ / σισυφοσ", "ΣΙΣΥΦΟΣ", "σισυφοσ", True),
        ("…the same + .txt", "ΣΙΣΥΦΟΣ.txt", "σισυφοσ.txt", True),
        ("İstanbul / istanbul", "İstanbul.txt", "istanbul.txt", False),
        ("Q3 Report / Q3 report", "Q3 Report.csv", "Q3 report.csv", True),
    ]
    models = ["py_lower", "py_casefold", "simple_upper"]
    cols = models + ["volume"]

    for r, (label, a, b, truth) in enumerate(pairs):
        for c, key in enumerate(cols):
            if key == "volume":
                merged = truth
                wrong = False
            else:
                merged = S.FOLDS[key](a) == S.FOLDS[key](b)
                wrong = merged != truth
            face = DELIVERED if merged else PAPER
            ax.add_patch(
                plt.Rectangle(
                    (c - 0.46, r - 0.42), 0.92, 0.84,
                    facecolor=face,
                    edgecolor=ACCENT if wrong else "#b9b9c2",
                    lw=2.0 if wrong else 0.8,
                    zorder=3,
                )
            )
            txt = "merge" if merged else "keep"
            ax.text(c, r, txt, ha="center", va="center", fontsize=7,
                    color="white" if merged else "#5c5c66",
                    fontweight="bold" if wrong else "normal", zorder=4)
            if wrong:
                ax.text(c + 0.34, r + 0.26, "✗", ha="center", va="center",
                        fontsize=9, color=ACCENT, fontweight="bold", zorder=5)

    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(len(pairs) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["lower()", "casefold()", "simple\nupper", "NTFS\nAPFS"],
                       fontsize=7.5)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([p[0] for p in pairs], fontsize=7.5)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("4  No stdlib fold is a filesystem's case table", loc="left")


def panel5_destination_depth(ax: plt.Axes) -> None:
    """Same names, same target. Only the destination moves."""
    dests: List[str] = [
        r"C:\d",
        r"C:\data\exports",
        r"C:\Users\phoebe\Finance\2026\Q3\exports",
        r"C:\Users\phoebe\OneDrive - Contoso Ltd\Finance\2026\Q3\exports",
        r"C:\Users\phoebe\OneDrive - Contoso Ltd\Shared Documents\Finance"
        r"\Reporting\2026\Q3\regional\emea\exports",
        r"C:\Users\phoebe\OneDrive - Contoso Ltd\Shared Documents\Finance"
        r"\Reporting\2026\Q3\regional\emea\exports\final\approved\circulated",
    ]
    depth, over, deliv = [], [], []
    for d in dests:
        r = S.audit(NAMES, S.WINDOWS, d, "pathvalidate")
        f = [x for x in r.findings if x.code == "PATH_LENGTH_EXCEEDED"]
        depth.append(len(d))
        over.append(len(f[0].names) if f else 0)
        deliv.append(r.delivered)

    ax.plot(depth, deliv, "-o", color=DELIVERED, lw=1.8, ms=4.5,
            label="delivered", zorder=4)
    ax.plot(depth, over, "-s", color=ACCENT, lw=1.8, ms=4.5,
            label="path too long", zorder=4)
    for xx, yy in zip(depth, over):
        ax.annotate(str(yy), (xx, yy + 1.1), fontsize=7, color=ACCENT, ha="center")

    ax.set_xlabel("characters in the destination path")
    ax.set_ylabel("names")
    ax.set_ylim(-1.5, len(NAMES) - 8)
    ax.set_title("5  Validity depends on where you write it", loc="left")
    ax.legend(loc="center left", frameon=False, fontsize=7)
    # How many names move is a property of this corpus, not of MAX_PATH: only
    # names whose length falls in the 259-minus-depth band can change verdict.
    # Saying "2 of 42" without saying that would overstate the effect.
    ax.text(
        0.98, 0.60,
        "only names in the\n259-minus-depth band\ncan change verdict",
        transform=ax.transAxes, ha="right", va="center", fontsize=6.8,
        color="#5c5c66", linespacing=1.5,
    )
    tidy(ax, "y")


def panel6_round_trip(ax: plt.Axes) -> None:
    """Files lost building an archive on one volume and opening it on another."""
    profs = [S.LINUX_EXT4, S.OBJECT_STORE, S.WINDOWS, S.MACOS_APFS]
    short = {"linux-ext4": "ext4", "object-store": "objstore",
             "windows-ntfs": "NTFS", "macos-apfs": "APFS"}

    lost = [[S.round_trip(NAMES, a, b)["lost"] for b in profs] for a in profs]
    vmax = max(max(r) for r in lost) or 1

    for i, row in enumerate(lost):
        for j, v in enumerate(row):
            shade = v / vmax
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=(plt.matplotlib.colors.to_rgba(ACCENT, 0.12 + 0.78 * shade)
                               if v else PAPER),
                    edgecolor="#b9b9c2", lw=0.8, zorder=3,
                )
            )
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                    color="white" if shade > 0.45 else "#5c5c66",
                    fontweight="bold" if v else "normal", zorder=4)

    ax.set_xlim(-0.5, len(profs) - 0.5)
    ax.set_ylim(len(profs) - 0.5, -0.5)
    ax.set_xticks(range(len(profs)))
    ax.set_xticklabels([short[p.name] for p in profs], fontsize=7.5)
    ax.set_yticks(range(len(profs)))
    ax.set_yticklabels([short[p.name] for p in profs], fontsize=7.5)
    ax.set_xlabel("extracted on")
    ax.set_ylabel("archived on")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("6  Files lost crossing volumes (nothing sanitised)", loc="left")


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.4))
    panel1_frontier(axes[0][0])
    panel2_target_dependence(axes[0][1])
    panel3_length_units(axes[0][2])
    panel4_case_tables(axes[1][0])
    panel5_destination_depth(axes[1][1])
    panel6_round_trip(axes[1][2])

    fig.suptitle(
        "filename-sanitiser  ·  what a str -> str function cannot tell you",
        fontsize=13, fontweight="bold", x=0.008, ha="left", y=0.985,
    )
    fig.text(
        0.008, 0.945,
        f"{len(NAMES)} source names from one export dump, six sanitisers, five "
        f"target profiles. Every value computed by evidence.py; no randomness.",
        fontsize=8.5, color="#5c5c66", ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.928))
    fig.savefig("sanitise_audit.png", dpi=170, facecolor=PAPER)
    print("wrote sanitise_audit.png")


if __name__ == "__main__":
    main()
