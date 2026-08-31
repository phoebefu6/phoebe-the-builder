"""Figures for the Goodhart audit. Every value is computed here, not typed.

``goodhart_audit.png`` / ``.svg`` - six panels, the README hero.

The notebook draws its own smaller figure inline rather than importing this
module, so each figure is defined in exactly one place.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import goodhart as G  # noqa: E402

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

W = G.World()
ALPHA = 0.05


def _title(ax, n, text, sub):
    ax.text(0, 1.115, f"{n}  {text}", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color=INK, va="bottom")
    ax.text(0, 1.038, sub, transform=ax.transAxes,
            fontsize=7.6, color=MUTED, va="bottom")


# ---------------------------------------------------------------- panel 1
def panel_mechanism(ax):
    panel = G.simulate(W, regime="continuous")
    p = panel.proxy.mean(axis=1)
    y = panel.outcome.mean(axis=1)
    t = np.arange(panel.n_periods)
    ax.axvspan(panel.t_target - 0.5, panel.n_periods - 0.5, color=RED, alpha=0.05, lw=0)
    ax.axvline(panel.t_target - 0.5, color=RED, lw=1.1, ls=(0, (4, 2)))
    ax.plot(t, p, color=BLUE, lw=2.0, marker="o", ms=3.2, label="proxy (the KPI)")
    ax.plot(t, y, color=RED, lw=2.0, marker="s", ms=3.2, label="outcome (what it stood for)")
    ax.text(panel.t_target - 0.35, ax.get_ylim()[1], " proxy becomes a target",
            color=RED, fontsize=7.4, va="top")
    d = G.decompose(W, panel)
    ax.annotate(f"{d['exchange_rate']:.2f} outcome points\nper proxy point",
                xy=(panel.n_periods - 1.2, y[-1]), xytext=(0.40, 0.16),
                textcoords="axes fraction", fontsize=7.4, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    ax.set_xlabel("period")
    ax.set_ylabel("mean level")
    ax.legend(frameon=False, fontsize=7.4, loc="upper left", bbox_to_anchor=(0, 0.86))
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "1", "The KPI went up", "and the thing it was chosen to stand for went down")


# ---------------------------------------------------------------- panel 2
def panel_correlation_vs_damage(ax):
    us, drhos, dmgs = [], [], []
    for med in (3.0, 2.2, 1.9, 1.8, 1.7, 1.5, 1.2):
        w = replace(W, scruple_median=med)
        panel = G.simulate(w, regime="continuous")
        r1 = np.corrcoef(panel.post(panel.proxy), panel.post(panel.outcome))[0, 1]
        d = G.decompose(w, panel)
        us.append(d["diverted_share"])
        drhos.append(r1)
        dmgs.append(100 * -d["outcome_delta_true"] / panel.pre(panel.outcome).mean())
    ax.plot(us, dmgs, color=RED, lw=2.0, marker="s", ms=3.4)
    ax.set_ylabel("% of the outcome destroyed", color=RED)
    ax.tick_params(axis="y", colors=RED)
    ax.set_xlabel("share of effort diverted to moving the number")
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(us, drhos, color=BLUE, lw=2.0, marker="o", ms=3.4)
    ax2.set_ylabel("corr(proxy, outcome)", color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE)
    ax2.set_ylim(0.0, 1.0)
    ax2.axhline(W.rho_clean, color=BLUE, lw=0.9, ls=":")
    ax2.text(us[0], W.rho_clean + 0.03, f"chosen at {W.rho_clean:.2f}", fontsize=7.0, color=BLUE)
    ax2.annotate(f"still {drhos[-1]:.2f}", xy=(us[-1], drhos[-1]),
                 xytext=(us[-1] - 0.28, drhos[-1] - 0.22), fontsize=7.4, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "2", "The correlation barely moves",
           f"{dmgs[-1]:.0f}% of the outcome gone; the correlation slides {W.rho_clean - drhos[-1]:.2f}")


# ---------------------------------------------------------------- panel 3
def panel_winners_curse(ax):
    k, reps = 12, 150
    sizes = (15, 30, 60, 120, 300, 900, 3600)
    win_m, rnd_m = [], []
    for n_select in sizes:
        win, rnd = [], []
        for rep in range(reps):
            rng = np.random.default_rng(9000 + rep)
            betas, gammas = rng.uniform(0.55, 1.15, k), rng.uniform(0.45, 1.35, k)
            y, P = G.simulate_candidates(W, betas, gammas, seed=int(rng.integers(1, 2**31)))
            fy, fP = y.ravel(), P.reshape(k, -1)
            idx = rng.choice(fy.size, n_select, replace=False)
            sel = np.array([np.corrcoef(fP[j][idx], fy[idx])[0, 1] for j in range(k)])
            a, b = int(np.argmax(sel)), int(rng.integers(0, k))
            win.append(np.corrcoef(fP[a], fy)[0, 1] - sel[a])
            rnd.append(np.corrcoef(fP[b], fy)[0, 1] - sel[b])
        win_m.append(np.mean(win))
        rnd_m.append(np.mean(rnd))
    ax.axhspan(-0.0978, -0.0550, color=RED, alpha=0.11, lw=0)
    ax.text(sizes[-1], -0.074, " what real\n gaming did", fontsize=7.2, color=RED, va="center")
    ax.plot(sizes, win_m, color=PURPLE, lw=2.0, marker="o", ms=3.6,
            label="the proxy that was chosen")
    ax.plot(sizes, rnd_m, color=MUTED, lw=1.6, marker="^", ms=3.4, ls="--",
            label="a proxy picked at random")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("observations the metric was chosen on")
    ax.set_ylabel("change in correlation")
    ax.legend(frameon=False, fontsize=7.4, loc="lower right")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "3", "Nobody is gaming anything here",
           "the chosen proxy decays because it was chosen")


# ---------------------------------------------------------------- panel 4
def panel_detector_auc(ax):
    reps, n_agents = 120, 120
    names, aucs, free = [], [], []
    store = {n: ([], []) for n in G.DETECTOR_NAMES}
    for rep in range(reps):
        w = replace(W, seed=4000 + rep, n_agents=n_agents)
        for gaming, slot in ((False, 0), (True, 1)):
            panel = G.simulate(w, regime="threshold", gaming=gaming)
            for n, v in G.run_all(panel).items():
                store[n][slot].append(v.pvalue)
    for n in G.DETECTOR_NAMES:
        null, alt = np.array(store[n][0]), np.array(store[n][1])
        u = stats.mannwhitneyu(-alt, -null, alternative="greater").statistic
        names.append(n)
        aucs.append(u / (alt.size * null.size))
        probe = getattr(G, n)(G.simulate(replace(W, n_agents=n_agents), regime="threshold"))
        free.append(not probe.needs_outcome)
    order = np.argsort(aucs)
    names = [names[i] for i in order]
    aucs = [aucs[i] for i in order]
    free = [free[i] for i in order]
    colors = [GREEN if f else BLUE for f in free]
    ax.barh(range(len(names)), aucs, color=colors, height=0.62)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.6)
    ax.axvline(0.5, color=MUTED, lw=0.9, ls=":")
    ax.text(0.505, -0.75, "coin flip", fontsize=7.0, color=MUTED)
    for i, a in enumerate(aucs):
        ax.text(a + 0.008, i, f"{a:.3f}", va="center", fontsize=7.2, color=INK)
    ax.set_xlim(0.42, 1.06)
    ax.set_xlabel("AUC against a threshold target")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GREEN),
               plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    ax.legend(handles, ["needs no outcome", "needs the outcome"],
              frameon=False, fontsize=7.2, loc="lower right")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "4", "The best detector never looks at the outcome",
           f"{n_agents} agents, {reps} paired worlds")


# ---------------------------------------------------------------- panel 5
def panel_damage_at_detection(ax):
    lag, reps = 4, 60
    dmg, fires = [], {n: [] for n in G.DETECTOR_NAMES}
    for rep in range(reps):
        w = replace(W, seed=5000 + rep)
        panel = G.simulate(w, regime="threshold", n_post=14)
        dmg.append(w.outcome_cost * panel.diverted[panel.t_target:].mean(axis=1))
        for n in G.DETECTOR_NAMES:
            fn = getattr(G, n)
            needs = fn(panel).needs_outcome
            hit = None
            for t in range(panel.t_target + 2, panel.n_periods + 1):
                win = t - lag if needs else t
                if win < panel.t_target + 2:
                    continue
                if fn(panel, upto=win).pvalue < ALPHA:
                    hit = t - panel.t_target
                    break
            fires[n].append(hit)
    cum = np.cumsum(np.mean(dmg, axis=0))
    x = np.arange(1, len(cum) + 1)
    ax.fill_between(x, 0, cum, color=RED, alpha=0.13, lw=0)
    ax.plot(x, cum, color=RED, lw=2.0)
    ax.set_xlabel("periods since the target was set")
    ax.set_ylabel("cumulative outcome damage")
    shown = ["bunching", "holdout_divergence", "ratio_shift", "corr_drop"]
    offs = {"bunching": (12, -4), "holdout_divergence": (10, 20),
            "ratio_shift": (-80, 22), "corr_drop": (14, -26)}
    for n in shown:
        hits = [f for f in fires[n] if f is not None]
        if not hits:
            continue
        m = int(np.median(hits))
        col = GREEN if not getattr(G, n)(G.simulate(W, regime="threshold")).needs_outcome else BLUE
        ax.plot([m], [cum[m - 1]], marker="o", ms=6, color=col, zorder=5)
        ax.annotate(f"{n}\n{100*cum[m-1]/cum[-1]:.0f}% of the damage",
                    xy=(m, cum[m - 1]), xytext=offs[n], textcoords="offset points",
                    fontsize=7.0, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7))
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "5", "The outcome arrives too late to help",
           f"outcome lagged {lag} periods; markers at the median first firing")


# ---------------------------------------------------------------- panel 6
def panel_holdout_leak(ax):
    reps = 90
    leaks = (0.0, 0.25, 0.5, 0.6, 0.65, 0.7, 0.75, 0.9, 1.0)
    power = []
    for leak in leaks:
        hits = 0
        for rep in range(reps):
            w = replace(W, seed=6000 + rep)
            panel = G.simulate(w, regime="continuous")
            leaked = G.Panel(panel.proxy, panel.outcome,
                             panel.holdout + leak * w.gamma * panel.diverted,
                             panel.diverted, panel.t_target, panel.threshold)
            hits += G.holdout_divergence(leaked).pvalue < ALPHA
        power.append(hits / reps)
    ax.plot(leaks, power, color=GREEN, lw=2.2, marker="o", ms=4)
    ax.fill_between(leaks, 0, power, color=GREEN, alpha=0.10, lw=0)
    ax.axhline(0.8, color=MUTED, lw=0.9, ls=":")
    ax.text(0.02, 0.82, "80% power", fontsize=7.0, color=MUTED)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("share of the exploit that also lands on the holdout")
    ax.set_ylabel("power to detect gaming")
    ax.annotate("the holdout became\na managed metric",
                xy=(1.0, power[-1]), xytext=(0.55, 0.42), fontsize=7.4, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(ax, "6", "A holdout is a commitment, not a statistic",
           "it stops working before anyone notices it stopped")


def main() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 13.6))
    fig.suptitle("A proxy metric is a bet that a correlation survives being optimised",
                 x=0.055, y=0.985, ha="left", fontsize=14.5, fontweight="bold", color=INK)
    fig.text(0.055, 0.958,
             "Day 162  ·  goodhart-detector  ·  one latent driver, one outcome, one proxy, "
             "one exploit — and seven ways of noticing",
             ha="left", fontsize=9, color=MUTED)
    for fn, ax in zip(
        [panel_mechanism, panel_correlation_vs_damage, panel_winners_curse,
         panel_detector_auc, panel_damage_at_detection, panel_holdout_leak],
        axes.ravel(),
    ):
        fn(ax)
    fig.tight_layout(rect=(0.012, 0.008, 0.988, 0.945))
    fig.subplots_adjust(hspace=0.52, wspace=0.28)
    fig.savefig("goodhart_audit.png", dpi=300)
    fig.savefig("goodhart_audit.svg")
    print("wrote goodhart_audit.png / .svg")


if __name__ == "__main__":
    main()
