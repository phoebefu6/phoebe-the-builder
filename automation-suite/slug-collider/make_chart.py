"""The six-panel audit figure. Every value is computed at draw time.

Run: python3 make_chart.py [outfile.png]
"""

from __future__ import annotations

import sys
import unicodedata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import slug as S
from matplotlib.patches import Rectangle

INK = "#1b1b1f"
MUTED = "#8a8a94"
GRID = "#e3e3e8"
BAD = "#c0392b"
WARN = "#d98324"
OK = "#2d7d5a"
COOL = "#2f6f9f"
PAPER = "#fbfbfd"

PROFILE_ORDER = [
    "django_ascii",
    "casefold_ascii",
    "rails_parameterize",
    "naive_regex",
    "wordpress",
    "django_unicode",
    "github_anchor",
]


def _style(ax, title: str, subtitle: str = "") -> None:
    ax.set_facecolor(PAPER)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left", pad=26)
    if subtitle:
        ax.text(
            0, 1.012, subtitle, transform=ax.transAxes, color=MUTED,
            fontsize=8.5, va="bottom", ha="left",
        )


# ---------------------------------------------------------------------------


def panel_distinct_urls(ax) -> None:
    n = len(S.CORPUS)
    vals, labels = [], []
    for name in PROFILE_ORDER:
        r = S.audit(S.CORPUS, name)
        vals.append(len({s for s in r.slugs.values() if s}))
        labels.append(name)
    y = range(len(vals))
    colors = [BAD if v < n * 0.75 else (WARN if v < n * 0.85 else COOL) for v in vals]
    ax.barh(list(y), vals, color=colors, height=0.62)
    ax.axvline(n, color=INK, lw=1.2, ls="--")
    ax.text(n, len(vals) - 0.35, f" {n} titles", color=INK, fontsize=8, va="center")
    for i, v in enumerate(vals):
        ax.text(v - 0.6, i, str(v), color="white", fontsize=8.5,
                fontweight="bold", va="center", ha="right")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, n + 6)
    ax.invert_yaxis()
    ax.set_xlabel("distinct usable URLs", color=MUTED, fontsize=8)
    _style(ax, "Every algorithm loses URLs",
           f"{n} ordinary titles in, fewer than {n} addresses out")


def panel_findings(ax) -> None:
    kinds = [
        (S.Kind.COLLISION, BAD, "collision"),
        (S.Kind.EMPTY_SLUG, "#7b1e1e", "empty slug"),
        (S.Kind.ROUTE_SHADOW, WARN, "route shadow"),
        (S.Kind.CONFUSABLE_SPLIT, COOL, "confusable split"),
    ]
    lefts = [0.0] * len(PROFILE_ORDER)
    y = list(range(len(PROFILE_ORDER)))
    for kind, colour, label in kinds:
        vals = [len(S.audit(S.CORPUS, n).of_kind(kind)) for n in PROFILE_ORDER]
        ax.barh(y, vals, left=lefts, color=colour, height=0.62, label=label)
        for i, (v, left_edge) in enumerate(zip(vals, lefts)):
            if v:
                ax.text(left_edge + v / 2, i, str(v), color="white", fontsize=7.5,
                        fontweight="bold", ha="center", va="center")
        lefts = [a + b for a, b in zip(lefts, vals)]
    ax.set_yticks(y)
    ax.set_yticklabels(PROFILE_ORDER, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("findings", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.09))
    _style(ax, "Different algorithms, different failures",
           "no profile trades its way out - it picks which loss to take")


def panel_the_fold(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rows = ["café", "Ångström", "Straße", "Łódź", "Søren", "Encyclopædia"]
    ax.text(0.0, 0.94, "title", color=MUTED, fontsize=8)
    ax.text(0.28, 0.94, "decomposes?", color=MUTED, fontsize=8)
    ax.text(0.55, 0.94, "django slugify()", color=MUTED, fontsize=8)
    top = 0.83
    step = 0.125
    for i, t in enumerate(rows):
        yy = top - i * step
        nfkd = unicodedata.normalize("NFKD", t)
        lost = "".join(
            ch for ch in nfkd if ord(ch) > 127 and not unicodedata.combining(ch)
        )
        out = S.django_ascii(t)
        good = not lost
        ax.text(0.0, yy, t, color=INK, fontsize=10)
        ax.text(0.28, yy, "yes" if good else f"no  ({lost})",
                color=OK if good else BAD, fontsize=9)
        ax.text(0.55, yy, f"/{out}", color=INK if good else BAD,
                fontsize=9.5, family="monospace")
        if not good:
            ax.add_patch(
                Rectangle((0.54, yy - 0.033), 0.45, 0.075, facecolor=BAD,
                          alpha=0.07, edgecolor="none")
            )
    ax.text(0.0, 0.06,
            "NFKD splits a composed letter into base + mark, so the base survives\n"
            "the ASCII filter. A letter with no decomposition is deleted outright.",
            color=MUTED, fontsize=8)
    _style(ax, "The fold is not accent-stripping",
           "it keeps whatever happens to decompose")
    for s in ax.spines.values():
        s.set_visible(False)


def panel_truncation(ax) -> None:
    caps = [255, 200, 160, 120, 100, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15]
    curve = S.truncation_curve(S.CORPUS, "django_ascii", caps)
    x = [c for c, _, _, _ in curve]
    hit = [h for _, _, h, _ in curve]
    groups = [g for _, g, _, _ in curve]
    ax.plot(x, hit, color=BAD, lw=2, marker="o", ms=3, label="titles sharing a URL")
    ax.plot(x, groups, color=MUTED, lw=1.4, ls="--", marker="s", ms=2.6,
            label="collision groups")
    ax.invert_xaxis()
    for cap in (255, 50):
        ax.axvline(cap, color=GRID, lw=1)
    ax.annotate("VARCHAR(255)", xy=(255, min(groups)), fontsize=7.5, color=MUTED,
                ha="left", va="bottom", xytext=(3, 2), textcoords="offset points")
    ax.annotate("VARCHAR(50)", xy=(50, min(groups)), fontsize=7.5, color=MUTED,
                ha="left", va="bottom", xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("slug column length cap (characters, shrinking ->)",
                  color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=7.5, loc="center left")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _style(ax, "Shortening the column adds collisions",
           "group count wobbles; the number of titles that lose a URL only rises")


def panel_order(ax) -> None:
    c = list(S.CORPUS)
    orders = {
        "as listed": c,
        "reversed": list(reversed(c)),
        "A-Z": sorted(c),
        "Z-A": sorted(c, reverse=True),
    }
    n, unstable = S.order_sensitivity(c, list(orders.values()))
    demo = ["Hello, World!", "Hello --- World", "Hello World"]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xs = [0.34, 0.56, 0.78]
    for j, t in enumerate(demo):
        ax.text(xs[j], 0.9, t.replace("Hello", "H"), color=MUTED, fontsize=7.5,
                ha="center", rotation=0)
    demo_orders = [
        ("as listed", demo),
        ("reversed", list(reversed(demo))),
        ("A-Z", sorted(demo)),
    ]
    for i, (label, order) in enumerate(demo_orders):
        yy = 0.74 - i * 0.16
        got = S.assign(order)
        ax.text(0.0, yy, label, color=INK, fontsize=8.5)
        for j, t in enumerate(demo):
            u = got[t]
            bare = not u.endswith(("-2", "-3"))
            ax.text(xs[j], yy, "/hello-world" + ("" if bare else u[-2:]),
                    color=OK if bare else BAD, fontsize=8.5, ha="center",
                    family="monospace")
    ax.text(0.0, 0.2,
            f"{n} of {len(c)} titles received more than one URL across four\n"
            f"plausible import orders. Re-importing from a backup that iterates\n"
            f"differently does not preserve a single one of them.",
            color=INK, fontsize=8.5)
    _style(ax, "The URL is a function of import order",
           "not of the post")
    for s in ax.spines.values():
        s.set_visible(False)


def panel_two_failures(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    r = S.audit(S.CORPUS, "django_ascii")
    n_coll = sum(len(f.titles) for f in r.of_kind(S.Kind.COLLISION))
    n_split = sum(len(f.titles) for f in r.of_kind(S.Kind.CONFUSABLE_SPLIT))

    boxes = [
        (0.0, 0.52, BAD, "COLLISION", f"{n_coll} titles",
         "two titles, one URL", "caught by a UNIQUE index"),
        (0.52, 0.52, COOL, "CONFUSABLE SPLIT", f"{n_split} titles",
         "one headline, two URLs", "caught by nothing"),
    ]
    for x, y, colour, kind, count, what, caught in boxes:
        ax.add_patch(Rectangle((x, y - 0.02), 0.46, 0.44, facecolor=colour,
                               alpha=0.08, edgecolor=colour, lw=1.1))
        ax.text(x + 0.03, y + 0.34, kind, color=colour, fontsize=9,
                fontweight="bold")
        ax.text(x + 0.03, y + 0.24, count, color=INK, fontsize=15,
                fontweight="bold")
        ax.text(x + 0.03, y + 0.15, what, color=INK, fontsize=8.5)
        ax.text(x + 0.03, y + 0.07, caught, color=MUTED, fontsize=8)

    pair = ["Аpple silicon benchmarks", "Apple silicon benchmarks"]
    ax.text(0.0, 0.4, "the pair that no constraint sees:", color=MUTED, fontsize=8)
    for i, t in enumerate(pair):
        yy = 0.3 - i * 0.13
        ax.text(0.0, yy, t, color=INK, fontsize=9.5)
        ax.text(0.0, yy - 0.06,
                f"U+{ord(t[0]):04X}   ->  /{S.django_ascii(t)}",
                color=MUTED, fontsize=8, family="monospace")
    ax.text(0.0, 0.0,
            "identical on screen, distinct in the database, two live pages.",
            color=BAD, fontsize=8.5)
    _style(ax, "Two failures, one of them invisible",
           "a uniqueness constraint only sees the left-hand one")
    for s in ax.spines.values():
        s.set_visible(False)


def main(out: str = "slug_audit.png") -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 15.5))
    fig.patch.set_facecolor("white")
    panel_distinct_urls(axes[0][0])
    panel_findings(axes[0][1])
    panel_the_fold(axes[1][0])
    panel_truncation(axes[1][1])
    panel_order(axes[2][0])
    panel_two_failures(axes[2][1])

    fig.suptitle(
        "Slug collider - what a slugifier cannot tell you about the corpus it just slugified",
        color=INK, fontsize=14, fontweight="bold", x=0.02, ha="left", y=0.985,
    )
    fig.text(
        0.02, 0.965,
        f"{len(S.CORPUS)} ordinary blog titles - none adversarial, none invalid - "
        f"through {len(S.PROFILES)} published slug algorithms. All values computed at draw time.",
        color=MUTED, fontsize=9.5, ha="left",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.945), h_pad=4.0, w_pad=3.0)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "slug_audit.png")
