"""Six panels, all of them recomputed from sequential.py rather than typed in."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sequential import (  # noqa: E402
    ALPHA,
    K_DAILY,
    N_MAX,
    P0,
    P1,
    Trial,
    bonferroni_bounds,
    equal_looks,
    first_crossing,
    msprt_crossing,
    naive_bounds,
    obf_bounds,
    pocock_bounds,
    score,
    simulate,
)

INK = "#16222e"
MUTE = "#8b9aa7"
GOOD = "#1f7a5c"
BAD = "#b3402f"
WARN = "#c98a1a"
COOL = "#2b6ca3"
PLUM = "#6b4d8f"
GRID = "#dfe5ea"
M = 40_000
SEED = 4242


def _style(ax, title: str, sub: str = "") -> None:
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=30)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=8.6, color=MUTE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTE, labelsize=8.4, length=0)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


BOUNDS = {
    "naive": naive_bounds(K_DAILY, ALPHA),
    "bonferroni": bonferroni_bounds(K_DAILY, ALPHA),
    "pocock": pocock_bounds(K_DAILY, ALPHA, step=0.005),
    "obf": obf_bounds(K_DAILY, ALPHA, step=0.005),
}
LABELS = {
    "fixed": "fixed horizon",
    "naive": "naive peek",
    "bonferroni": "Bonferroni",
    "pocock": "Pocock",
    "obf": "O'Brien-Fleming",
    "msprt": "mSPRT",
}
COLORS = {
    "fixed": MUTE,
    "naive": BAD,
    "bonferroni": WARN,
    "pocock": COOL,
    "obf": GOOD,
    "msprt": PLUM,
}


def compute():
    looks = equal_looks(K_DAILY)
    t_null = simulate(looks, P0, P0, M, SEED)
    t_alt = simulate(looks, P0, P1, M, SEED + 1)
    fx_null = Trial(looks[-1:], t_null.z[:, -1:], t_null.diff[:, -1:], t_null.se[:, -1:], P0, P0)
    fx_alt = Trial(looks[-1:], t_alt.z[:, -1:], t_alt.diff[:, -1:], t_alt.se[:, -1:], P0, P1)
    tau = P1 - P0

    res = {}
    res["fixed"] = (
        score(fx_null, first_crossing(fx_null.z, naive_bounds(1, ALPHA)), "fixed"),
        score(fx_alt, first_crossing(fx_alt.z, naive_bounds(1, ALPHA)), "fixed"),
    )
    for name, b in BOUNDS.items():
        res[name] = (
            score(t_null, first_crossing(t_null.z, b), name),
            score(t_alt, first_crossing(t_alt.z, b), name),
        )
    res["msprt"] = (
        score(t_null, msprt_crossing(t_null, tau, ALPHA), "msprt"),
        score(t_alt, msprt_crossing(t_alt, tau, ALPHA), "msprt"),
    )

    ks = [1, 2, 3, 5, 10, 20, 50, 100]
    naive_curve = []
    for k in ks:
        t = simulate(equal_looks(k), P0, P0, M, SEED + 10 + k)
        naive_curve.append(float((first_crossing(t.z, naive_bounds(k, ALPHA)) >= 0).mean()))
        del t

    cont_looks = np.arange(500, 200_001, 500, dtype=np.int64)
    tc = simulate(cont_looks, P0, P0, 10_000, SEED + 3)
    ever = np.maximum.accumulate(np.abs(tc.z) >= 1.959964, axis=1).mean(axis=0)
    del tc

    sweep = []
    for rel in (0.20, 0.10, 0.05, 0.03, 0.02):
        p1 = P0 * (1 + rel)
        t = simulate(looks, P0, p1, M, SEED + 100 + int(rel * 100))
        o = score(t, first_crossing(t.z, BOUNDS["pocock"]), "pocock")
        tf = Trial(looks[-1:], t.z[:, -1:], t.diff[:, -1:], t.se[:, -1:], P0, p1)
        of = score(tf, first_crossing(tf.z, naive_bounds(1, ALPHA)), "fixed")
        sweep.append((rel, o.est_bias, o.reject_rate, of.reject_rate))
        del t, tf
    return res, ks, naive_curve, cont_looks, ever, sweep


def panel_looks(ax, ks, curve):
    ax.plot(ks, curve, "-o", color=BAD, lw=2.0, ms=5, zorder=3)
    ax.axhline(ALPHA, color=INK, lw=1.2, ls="--", zorder=2)
    ax.text(100, ALPHA + 0.014, "what the p-value claims: 0.05", fontsize=8.2, color=INK, ha="right")
    for k, v in zip(ks, curve):
        if k in (1, 5, 20, 100):
            ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.4, color=BAD, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("looks before the planned end", fontsize=8.6, color=MUTE)
    ax.set_ylabel("false-positive rate", fontsize=8.6, color=MUTE)
    ax.set_ylim(0, 0.44)
    _style(ax, "1. Nothing but the stopping rule changes",
           f"two arms converting identically at {P0:.0%}; stop the first time p < 0.05")


def panel_unbounded(ax, cont_looks, ever):
    ax.plot(cont_looks, ever, color=BAD, lw=2.0)
    ax.axhline(ALPHA, color=INK, lw=1.2, ls="--")
    ax.axvline(N_MAX, color=MUTE, lw=1.0, ls=":")
    ax.text(N_MAX * 1.10, 0.135, "planned end\nof the experiment", fontsize=8.0, color=MUTE)
    ax.text(198_000, ALPHA + 0.014, "what the p-value claims: 0.05", fontsize=8.2,
            color=INK, ha="right")
    for n in (20_000, 200_000):
        j = int(n // 500) - 1
        ax.annotate(f"{ever[j]:.2f}", (n, ever[j]), textcoords="offset points",
                    xytext=(-4, 8), ha="right", fontsize=8.4, color=BAD, fontweight="bold")
    ax.set_xlabel("visitors per arm, peeking every 500", fontsize=8.6, color=MUTE)
    ax.set_ylabel("chance of a 'win' so far", fontsize=8.6, color=MUTE)
    ax.set_ylim(0, 0.55)
    _style(ax, "2. An experiment never called is eventually significant",
           "the same empty world, monitored continuously and left running")


def panel_boundaries(ax):
    x = np.arange(1, K_DAILY + 1)
    for name in ("naive", "bonferroni", "pocock", "obf"):
        ax.plot(x, BOUNDS[name], "-o", ms=3.4, lw=1.8, color=COLORS[name], label=LABELS[name])
    ax.set_ylim(1.6, 5.2)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("look number (of 20)", fontsize=8.6, color=MUTE)
    ax.set_ylabel("|z| needed to stop", fontsize=8.6, color=MUTE)
    ax.legend(frameon=False, fontsize=8.0, loc="upper right", ncol=2)
    ax.text(1.4, 4.85, "O'Brien-Fleming starts at 9.5,\noff the top of this axis",
            fontsize=7.8, color=GOOD)
    _style(ax, "3. Four bars, one alpha", "all but the red one spend 0.05 across all twenty looks")


def panel_tradeoff(ax, res):
    for name, (o0, o1) in res.items():
        ax.scatter(o1.expected_n, o1.reject_rate, s=90, color=COLORS[name], zorder=3,
                   edgecolor="white", lw=1.2)
        dx, dy, ha = 0.0, 0.016, "center"
        if name == "bonferroni":
            dx, dy, ha = -350, -0.004, "right"
        elif name == "msprt":
            dx, dy, ha = 350, -0.004, "left"
        ax.annotate(f"{LABELS[name]}  FPR {o0.reject_rate:.3f}" if dx else
                    f"{LABELS[name]}\nFPR {o0.reject_rate:.3f}",
                    (o1.expected_n + dx, o1.reject_rate + dy), ha=ha, va="center" if dx else "bottom",
                    fontsize=8.0, color=COLORS[name], fontweight="bold")
    ax.set_xlabel("visitors per arm actually consumed (true lift present)", fontsize=8.6, color=MUTE)
    ax.set_ylabel("power", fontsize=8.6, color=MUTE)
    ax.set_ylim(0.60, 1.02)
    ax.set_xlim(5_000, 23_500)
    _style(ax, "4. The trade is real, and one point on it is cheating",
           "up and to the left is better; the red point bought its position with alpha")


def panel_bias(ax, res):
    names = ["fixed", "obf", "msprt", "bonferroni", "pocock", "naive"]
    vals = [res[n][1].est_at_stop for n in names]
    ax.barh(range(len(names)), vals, color=[COLORS[n] for n in names], height=0.62)
    ax.axvline(P1 - P0, color=INK, lw=1.4, ls="--")
    ax.text(P1 - P0, -0.78, f"true lift {P1 - P0:.3f}", fontsize=8.4,
            color=INK, ha="center", fontweight="bold")
    for i, (n, v) in enumerate(zip(names, vals)):
        ax.text(v + 0.0004, i, f"{v:.4f}  ({res[n][1].est_bias:+.0%})", va="center",
                fontsize=8.2, color=INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABELS[n] for n in names], fontsize=8.6)
    ax.set_ylim(-1.05, 5.55)
    ax.set_xlim(0, 0.022)
    ax.set_xlabel("lift reported at the stopping look", fontsize=8.6, color=MUTE)
    ax.grid(False, axis="y")
    _style(ax, "5. Every valid test here overstates the effect",
           "averaged over the runs that rejected; a boundary controls the sign, not the size")


def panel_curse(ax, sweep):
    rel = [s[0] for s in sweep]
    bias = [s[1] for s in sweep]
    pw = [s[2] for s in sweep]
    ax.plot(rel, bias, "-o", color=BAD, lw=2.0, ms=5, zorder=3)
    for r, b in zip(rel, bias):
        ax.annotate(f"{b:+.0%}", (r, b), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.4, color=BAD, fontweight="bold")
    ax.set_xlabel("true relative lift", fontsize=8.6, color=MUTE)
    ax.set_ylabel("overstatement of the reported lift", fontsize=8.6, color=BAD)
    ax.set_xlim(0.008, 0.215)
    ax.set_ylim(-0.3, 6.2)
    ax.set_xscale("log")
    ax.set_xticks(rel)
    ax.set_xticklabels([f"{r:.0%}" for r in rel])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    ax2 = ax.twinx()
    ax2.plot(rel, pw, "-s", color=COOL, lw=1.6, ms=4)
    ax2.set_ylabel("power of the same design", fontsize=8.6, color=COOL)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(colors=COOL, labelsize=8.2, length=0)
    for s in ("top", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    _style(ax, "6. The weaker the effect, the bigger the lie",
           "20-look Pocock: only the lucky runs cross, so the survivor is mostly luck")


def main() -> None:
    res, ks, curve, cont_looks, ever, sweep = compute()
    fig, axes = plt.subplots(3, 2, figsize=(15.2, 16.4))
    fig.patch.set_facecolor("white")
    panel_looks(axes[0][0], ks, curve)
    panel_unbounded(axes[0][1], cont_looks, ever)
    panel_boundaries(axes[1][0])
    panel_tradeoff(axes[1][1], res)
    panel_bias(axes[2][0], res)
    panel_curse(axes[2][1], sweep)
    fig.suptitle("A stopping rule is part of the test", x=0.008, y=0.998, ha="left",
                 fontsize=16.5, fontweight="bold", color=INK)
    fig.text(0.008, 0.9785,
             "Day 164 - peeking-cost - group-sequential boundaries solved from the "
             "Armitage-McPherson recursion, every rate measured on simulated Bernoulli traffic",
             ha="left", va="top", fontsize=9.6, color=MUTE)
    fig.tight_layout(rect=(0, 0, 1, 0.9725))
    fig.savefig("peeking_cost_audit.png", dpi=300, facecolor="white")
    fig.savefig("peeking_cost_audit.svg", facecolor="white")
    print("wrote peeking_cost_audit.png and .svg")


if __name__ == "__main__":
    main()
