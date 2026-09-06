"""Six panels. Every number is computed here, not typed."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import codelay as C
import matplotlib.pyplot as plt
import numpy as np

INK = "#1d2433"
MUTED = "#8b93a7"
GRID = "#e4e7ee"
BAD = "#c0392b"
OK = "#1f7a5a"
MID = "#c98a1e"
BLUE = "#2f5f9e"
FACE = "#fbfbfd"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "figure.facecolor": FACE,
    "axes.facecolor": FACE,
})


def style(ax, title, sub=""):
    ax.set_title(title, loc="left", pad=21 if sub else 6)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=7.6,
                color=MUTED, va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def main() -> None:
    items = C.backlog()
    lin = C.linearised(items)
    sr = C.sweep(items)
    sl = C.sweep(lin)
    costs = C.all_costs(items)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    fig.subplots_adjust(left=0.072, right=0.985, top=0.845, bottom=0.075,
                        wspace=0.26, hspace=0.50)

    # ---------------------------------------------------------------- panel 1
    ax = axes[0][0]
    weeks = np.linspace(0, 40, 400)
    for k, col in (("A", BLUE), ("B", BAD), ("D", OK), ("H", MID)):
        it = items[k]
        ax.plot(weeks, [it.cod.cum(w) for w in weeks], color=col, lw=1.9,
                label=f"{it.name}  (quoted {it.cod.rate(0):.0f}/wk, {it.cod.kind})")
    ax.axvline(26, color=MUTED, lw=0.9, ls=":")
    ax.text(26.4, 2400, "fixed date", fontsize=7, color=MUTED)
    ax.set_xlabel("week the item actually ships")
    ax.set_ylabel("total cost of delay ($k)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    style(ax, "1  Cost of delay is not a scalar",
          "the 0/week item ends up the most expensive one in the backlog")

    # ---------------------------------------------------------------- panel 2
    ax = axes[0][1]
    ax.hist(costs, bins=90, color="#ccd3e2", edgecolor="none")
    marks = [("optimum", sr["best"], OK), ("cd3 mean-rate", 3904.6, OK),
             ("random mean", sr["mean"], MUTED),
             ("rice", C.cost_of(C.order_rice(items), items), BAD),
             ("cd3 as elicited", C.cost_of(C.order_cd3_initial(items), items), BAD),
             ("hippo", C.cost_of(C.order_hippo(items), items), BAD)]
    top = ax.get_ylim()[1]
    for i, (lbl, v, col) in enumerate(marks):
        ax.axvline(v, color=col, lw=1.5)
        ax.text(v, top * (0.97 - 0.115 * i), f" {lbl} {v:.0f}", fontsize=7,
                color=col, va="top", rotation=0,
                ha="left" if v < 5200 else "right")
    ax.set_xlabel("total delay cost of the ordering ($k)")
    ax.set_ylabel(f"orderings (all {len(costs):,})")
    style(ax, "2  Four of nine methods lose to a hat",
          "every ordering enumerated - this is the population, not a sample")

    # ---------------------------------------------------------------- panel 3
    ax = axes[0][2]
    labels = ["linear backlog\n(Smith's rule holds)", "real shapes\n(nothing else changed)"]
    opt = [sl["best"], sr["best"]]
    cd3 = [C.cost_of(C.order_cd3_mean(lin), lin), C.cost_of(C.order_cd3_mean(items), items)]
    x = np.arange(2)
    ax.bar(x - 0.19, opt, 0.36, color=OK, label="exhaustive optimum")
    ax.bar(x + 0.19, cd3, 0.36, color=BLUE, label="CD3 / WSJF")
    for xi, (o, c) in enumerate(zip(opt, cd3)):
        g = c - o
        ax.text(xi + 0.19, c + 70, f"gap {g:.1f}" if g > 1 else "gap 0.0000",
                ha="center", fontsize=7.6, color=BAD if g > 1 else OK,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("total delay cost ($k)")
    ax.set_ylim(0, 5700)
    ax.legend(frameon=False, fontsize=7.4, loc="upper right")
    style(ax, "3  WSJF is optimal, and needs four conditions",
          "identical ordering and identical cost, until a shape is non-linear")

    # ---------------------------------------------------------------- panel 4
    ax = axes[1][0]
    names, fins = [], []
    for n, f in C.ORDERINGS.items():
        names.append(n)
        fins.append(C.completions(f(items), items)["B"])
    names.append("optimum")
    fins.append(C.completions(sr["best_order"], items)["B"])
    order = np.argsort(fins)
    names = [names[i] for i in order]
    fins = [fins[i] for i in order]
    cols = [BAD if v > 26 else (OK if v > 20 else MID) for v in fins]
    ax.barh(range(len(names)), fins, color=cols, height=0.62)
    ax.axvline(26, color=INK, lw=1.4)
    ax.text(26.6, len(names) - 0.35, "the date", fontsize=7.4, color=INK,
            fontweight="bold")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7.6)
    ax.invert_yaxis()
    ax.set_xlabel("week soc2-evidence ships")
    ax.set_xlim(0, 44)
    for i, v in enumerate(fins):
        pen = items["B"].cod.cum(v)
        off = -3.4 if names[i] == "optimum" else 0.6
        ax.text(v + off, i, f"{v:.0f}" + (f"   pays {pen:.0f}" if pen else ""),
                va="center", fontsize=7, color=INK if pen else MUTED,
                ha="right" if off < 0 else "left")
    style(ax, "4  Nobody schedules to the date",
          "miss by 14 weeks, or hit it 22 weeks early - only the optimum lands near it")

    # ---------------------------------------------------------------- panel 5
    ax = axes[1][1]
    ex2 = C.optimal_two_team_assignment(lin)["best"]
    gr2 = C.parallel_cost(C.order_cd3_mean(lin), lin, 2)
    vals = [sl["best"], ex2, gr2]
    labs = ["1 team\noptimum", "2 teams\nexact optimum", "2 teams\nCD3 list-schedule"]
    ax.bar(labs, vals, color=[MUTED, OK, BLUE], width=0.56)
    for i, v in enumerate(vals):
        ax.text(i, v + 55, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
    ax.annotate("", xy=(1, ex2 + 320), xytext=(0, sl["best"] + 320),
                arrowprops=dict(arrowstyle="->", color=BAD, lw=1.2))
    ax.text(0.5, sl["best"] + 420, f"-{100 * (1 - ex2 / sl['best']):.1f}%, not -50%",
            ha="center", fontsize=7.6, color=BAD, fontweight="bold")
    ax.text(2, gr2 + 480, f"list-scheduling gap\n{100 * (gr2 / ex2 - 1):.2f}%",
            ha="center", fontsize=7.4, color=OK)
    ax.set_ylabel("total delay cost ($k)")
    ax.set_ylim(0, 5400)
    style(ax, "5  The condition you can safely ignore",
          "parallel capacity is where people look first, and it costs 0.13% here")

    # ---------------------------------------------------------------- panel 6
    ax = axes[1][2]
    sigmas = [0.2, 0.35, 0.5, 0.7]
    res = [C.noise_sweep(items, s, 2000) for s in sigmas]
    truth = res[0]["truth_cost"]
    ax.plot(sigmas, [r["mean"] for r in res], "o-", color=BLUE, lw=1.8,
            label="mean realised cost")
    ax.fill_between(sigmas, [truth] * 4, [r["p90"] for r in res],
                    color=BLUE, alpha=0.13, label="up to p90")
    ax.axhline(truth, color=OK, lw=1.4)
    ax.axhline(sr["best"], color=MUTED, lw=1.1, ls="--")
    ax.axhline(sr["mean"], color=BAD, lw=1.1, ls=":")
    for y, txt, col in ((truth, f"CD3 on true durations  {truth:.0f}", OK),
                        (sr["best"], f"optimum  {sr['best']:.0f}", MUTED),
                        (sr["mean"], f"random mean  {sr['mean']:.0f}", BAD)):
        ax.text(0.783, y + 40, txt, fontsize=7, color=col, va="bottom",
                ha="right")
    for s, r in zip(sigmas, res):
        ax.text(s, r["mean"] - 130, f"{100 * r['reorder_rate']:.1f}%", ha="center",
                fontsize=7, color=INK)
    ax.set_xlim(0.17, 0.79)
    ax.set_xlabel("lognormal sigma on every duration estimate")
    ax.set_ylabel("total delay cost ($k)")
    ax.set_ylim(3150, 5450)
    ax.legend(frameon=False, fontsize=7.1, loc="upper left")
    style(ax, "6  The rank is noise, the cost is not",
          "% labels = trials where the ranking changed; it never survives")

    fig.text(0.055, 0.962, "An ordering is not a schedule",
             fontsize=17, fontweight="bold", color=INK)
    fig.text(0.055, 0.921,
             "One 9-item backlog, 40 weeks, every one of the 362,880 orderings "
             "priced against the schedule it produces. Cost of delay in $k.",
             fontsize=9, color=MUTED)
    fig.text(0.055, 0.895,
             "Day 158 - cost-of-delay - phoebe-the-builder", fontsize=7.6,
             color=MUTED)

    fig.savefig("cod_audit.png", dpi=180, facecolor=FACE)
    fig.savefig("cod_audit.svg", facecolor=FACE)
    print("wrote cod_audit.png, cod_audit.svg")


if __name__ == "__main__":
    main()
