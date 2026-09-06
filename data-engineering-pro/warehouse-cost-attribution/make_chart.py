"""Figures for the cost-attribution audit. Every value comes from the engine.

``cost_audit.png`` / ``.svg`` - six panels, the README hero.

The notebook draws its own smaller figure inline rather than importing this module, so
each figure is defined in exactly one place.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import costs as C
import matplotlib.pyplot as plt

INK = "#141414"
MUTED = "#8a8a8a"
GRID = "#e4e2dd"
PAPER = "#faf8f4"
RED = "#c0392b"
ORANGE = "#d98324"
BLUE = "#4a7c8c"
GREEN = "#4b7f52"
PURPLE = "#7a5a8c"

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

SHORT = {"analytics": "analytics", "growth": "growth", "finance": "finance",
         "ml_platform": "ml platform", "exec_reporting": "exec reporting",
         "scheduled_unowned": "orphaned jobs"}
METHOD_COLOR = {"direct_bytes": RED, "query_count": ORANGE, "equal_split": MUTED,
                "standalone": PURPLE, "marginal": GREEN, "first_toucher": "#b07d2b",
                "shapley": BLUE}


def _title(ax, text, sub=""):
    ax.set_title(text, loc="left", fontsize=9.5, fontweight="bold", color=INK, pad=26)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=7.6, color=MUTED, va="bottom")


# ---------------------------------------------------------------------------- panel 1
def panel_composition(ax):
    reads = C._reads_by_table(C.TEAM_NAMES)
    scan = storage = 0.0
    for table, r in reads.items():
        t = C.TABLES[table]
        cold = min(r, C.WORKDAYS * C.COLD_SLOTS_PER_DAY)
        scan += cold * t.scan_gb * C.SCAN_RATE + (r - cold) * t.scan_gb * C.SCAN_RATE * C.CACHE_RATE
        storage += t.storage_gb * C.STORAGE_RATE
    models = sum(C.WORKDAYS * m.build_gb * C.SCAN_RATE for m in C.MODELS)

    parts = [("query scans", scan, BLUE), ("storage", storage, GREEN),
             ("upstream models", models, PURPLE), ("reservation", C.RESERVED_FLOOR, ORANGE)]
    left = 0.0
    for label, amt, col in parts:
        ax.barh(1, amt, left=left, color=col, height=0.5,
                label=f"{label}  {amt/C.INVOICE:.0%}")
        if amt / C.INVOICE > 0.30:
            ax.text(left + amt / 2, 1, f"{amt/C.INVOICE:.0%}", ha="center", va="center",
                    fontsize=8.4, color="white", fontweight="bold")
        left += amt
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower right", handlelength=1.1)

    m_tot = sum(C.raw_marginal().values())
    s_tot = sum(C.raw_standalone().values())
    ax.barh(0.15, m_tot, color=RED, height=0.28)
    ax.text(m_tot + 900, 0.15, f"sum of MARGINAL costs  ${m_tot:,.0f}  ({m_tot/C.INVOICE:.0%})",
            va="center", fontsize=7.6, color=RED, fontweight="bold")
    ax.barh(1.75, s_tot, color=MUTED, height=0.28, alpha=0.55)
    ax.text(C.INVOICE * 1.02, 1.75, f"sum of STANDALONE costs  ${s_tot:,.0f}  ({s_tot/C.INVOICE:.0%})",
            va="center", fontsize=7.6, color=INK)
    ax.axvline(C.INVOICE, color=INK, lw=1.6)
    ax.text(C.INVOICE, 2.15, f"  the invoice  ${C.INVOICE:,.0f}", fontsize=8, fontweight="bold", color=INK)

    ax.set_yticks([0.15, 1, 1.75])
    ax.set_yticklabels(["if everyone pays\nwhat they cost you", "the invoice", "if everyone pays\nas if alone"],
                       fontsize=7.4)
    ax.set_xlim(0, s_tot * 1.02)
    ax.set_ylim(-0.25, 2.35)
    ax.set_xlabel("$ per month")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "The bill does not decompose",
           "charge marginal cost and 90% is unfunded; charge standalone and you collect 3.25x")


# ---------------------------------------------------------------------------- panel 2
def panel_methods(ax):
    allocs = {n: f() for n, f in C.METHODS.items()}
    order = sorted(C.TEAM_NAMES, key=lambda t: -max(allocs[m][t] for m in allocs))
    for i, team in enumerate(order):
        vals = [allocs[m][team] for m in allocs]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        for m in allocs:
            ax.scatter(allocs[m][team], i, s=42, color=METHOD_COLOR[m], zorder=3,
                       edgecolor=PAPER, linewidth=0.7, label=m if i == 0 else None)
        ax.text(max(vals) + 500, i, f"{max(vals)/max(min(vals),1):.0f}x", va="center",
                fontsize=7.4, color=INK, fontweight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[t] for t in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("$ billed to the team, for the same month")
    ax.set_xlim(-800, max(max(a.values()) for a in allocs.values()) * 1.16)
    ax.legend(frameon=False, fontsize=6.9, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "Seven defensible rules, one invoice",
           "exec reporting pays \\$32 or \\$5,092 depending only on which sentence was said")


# ---------------------------------------------------------------------------- panel 3
def panel_core(ax):
    sh = C.shapley()
    order = sorted(C.TEAM_NAMES, key=lambda t: -(C.core_range(t)[1] - C.core_range(t)[0]))
    for i, team in enumerate(order):
        lo, hi = C.core_range(team)
        ax.barh(i, hi - lo, left=lo, height=0.5, color=BLUE, alpha=0.22, zorder=1)
        ax.plot([lo, hi], [i, i], color=BLUE, lw=1.2, zorder=2)
        ax.scatter(sh[team], i, s=64, color=BLUE, zorder=4, marker="D",
                   edgecolor=PAPER, linewidth=0.8, label="shapley" if i == 0 else None)
        ax.scatter(lo, i, s=28, color=BLUE, zorder=4, marker="|", linewidth=1.6,
                   label="its marginal cost" if i == 0 else None)
        ax.text(hi + 400, i, f"{(hi-lo)/C.INVOICE:.0%} of the invoice", va="center",
                fontsize=7.2, color=MUTED)
    viol = C.core_violations(C.method_equal_split())
    S, ex = viol[0]
    ax.text(0.985, 0.30,
            "the core still rejects 2 of 7 rules -\nbut as COALITIONS, not individuals:\n"
            f"{{{', '.join(S)}}}\nobject to equal_split by \\${ex:,.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.2, color=RED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[t] for t in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("$ this team could be billed and nobody could object")
    ax.set_xlim(-700, 30_000)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "'Fair' is a polytope, not a point",
           "every shaded value satisfies the strongest test cooperative game theory offers")


# ---------------------------------------------------------------------------- panel 4
def panel_blame(ax):
    sh = C.shapley()
    saving = C.unowned_cost()
    labels = ["what Shapley says the\norphaned jobs consume", "what switching them\noff would save"]
    vals = [sh["scheduled_unowned"], saving]
    bars = ax.bar([0, 1], vals, color=[PURPLE, RED], width=0.5)
    for b, v_ in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v_ + 180, f"${v_:,.0f}\n{v_/C.INVOICE:.1%} of invoice",
                ha="center", fontsize=8.4, fontweight="bold",
                color=PURPLE if v_ > 1000 else RED)
    ax.annotate("", xy=(1, saving + 120), xytext=(0, vals[0]),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2,
                                connectionstyle="arc3,rad=-0.32"))
    ax.text(0.5, vals[0] * 0.62, f"{vals[0]/saving:.0f}x apart", ha="center", fontsize=9,
            fontweight="bold", color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("$ per month")
    ax.set_ylim(0, vals[0] * 1.35)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "Attribution is not the same question as saving",
           "both numbers are correct, and they support opposite decisions")


# ---------------------------------------------------------------------------- panel 5
def panel_unattributable(ax):
    res = C.RESERVED_FLOOR
    attributable = C.INVOICE - res
    ax.pie([attributable, res],
           labels=[f"usage\n${attributable:,.0f}", f"reservation\n${res:,.0f}\n({res/C.INVOICE:.0%})"],
           colors=[BLUE, ORANGE], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor=PAPER, linewidth=2),
           textprops=dict(fontsize=8.2, color=INK))
    ax.text(0, 0, "marginal cost\nof the reservation\nto EVERY team:\n$0", ha="center", va="center",
            fontsize=8.4, color=RED, fontweight="bold")
    _title(ax, "A sixth of the bill has no owner",
           "the floor is owed the moment anybody uses the warehouse, and survives losing any team")


# ---------------------------------------------------------------------------- panel 6
def panel_sampling(ax):
    exact = C.shapley()
    draws = [25, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]
    errs = []
    for m in draws:
        s = C.sampled_shapley(m, seed=1)
        errs.append(max(abs(s[n] - exact[n]) for n in C.TEAM_NAMES) / C.INVOICE)
    ax.plot(draws, errs, color=BLUE, lw=2.2, marker="o", ms=4, label="sampled Shapley error")
    ax.axhline(0.01, color=GREEN, lw=1.2, ls=":")
    ax.text(28, 0.0112, "1% of the invoice", color=GREEN, fontsize=7.4)

    for name, fn in C.METHODS.items():
        if name == "shapley":
            continue
        a = fn()
        dev = max(abs(a[n] - exact[n]) for n in C.TEAM_NAMES) / C.INVOICE
        ax.axhline(dev, color=METHOD_COLOR[name], lw=1.0, ls="--", alpha=0.75)
        ax.text(11_500, dev, f" {name}", fontsize=6.8, color=METHOD_COLOR[name], va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sampled orderings")
    ax.set_ylabel("max error, as a share of the invoice")
    ax.set_xlim(22, 60_000)
    ax.legend(frameon=False, fontsize=7.2, loc="lower left")
    ax.grid(color=GRID, lw=0.7, which="both")
    ax.set_axisbelow(True)
    _title(ax, "Sample Shapley; do not build the exact one",
           "dashed lines are how far each cheap rule sits from it - none of them approximate it")


def build(path: str = "cost_audit") -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13.8, 15.4))
    panel_composition(axes[0][0])
    panel_methods(axes[0][1])
    panel_core(axes[1][0])
    panel_sampling(axes[1][1])
    panel_blame(axes[2][0])
    panel_unattributable(axes[2][1])
    fig.suptitle("Who spent the forty thousand?",
                 x=0.008, y=0.995, ha="left", fontsize=14.5, fontweight="bold", color=INK)
    fig.text(0.008, 0.977,
             "A warehouse invoice is a joint cost, and a joint cost has no unique owner. Six teams, "
             "one month, 64 coalitions and 720 orderings computed exactly.",
             ha="left", fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.968])
    fig.subplots_adjust(hspace=0.46, wspace=0.28)
    fig.savefig(f"{path}.png", dpi=300)
    fig.savefig(f"{path}.svg")
    plt.close(fig)
    print(f"wrote {path}.png and {path}.svg")


if __name__ == "__main__":
    build()
