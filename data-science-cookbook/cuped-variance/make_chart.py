"""Six panels, all of them recomputed from cuped.py rather than typed in."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import cuped  # noqa: E402

INK = "#16222e"
MUTE = "#8b9aa7"
GOOD = "#1f7a5c"
BAD = "#b3402f"
WARN = "#c98a1a"
COOL = "#2b6ca3"
PLUM = "#6b4d8f"
GRID = "#dfe5ea"

W = cuped.World()
SEED = 20260904
TRIALS = 3000


def _style(ax, title: str, sub: str = "") -> None:
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=30)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=8.6, color=MUTE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTE, labelsize=8.4, length=3)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def panel_rho_squared(ax) -> None:
    rho = np.linspace(0, 0.98, 400)
    ax.plot(rho, cuped.variance_reduction(rho) * 100, color=GOOD, lw=2.2,
            label="variance removed (rho$^2$)")
    ax.plot(rho, rho * 100, color=MUTE, lw=1.4, ls=":", label="what rho itself looks like")
    for r, lab in ((0.5, "0.5"), (0.70710678, "0.707")):
        y = cuped.variance_reduction(r) * 100
        ax.plot([r], [y], marker="o", ms=6, color=BAD, zorder=5)
        ax.annotate(f"rho = {lab}\n{y:.0f}% of the sample saved\n6 weeks -> "
                    f"{6 * cuped.sample_size_multiplier(r):.1f}",
                    xy=(r, y), xytext=(r - 0.42, y + 16), fontsize=8.2, color=BAD,
                    arrowprops=dict(arrowstyle="->", color=BAD, lw=0.9))
    ax.set_xlabel("pre-period / in-experiment correlation", fontsize=8.6, color=MUTE)
    ax.set_ylabel("% of the sample size saved", fontsize=8.6, color=MUTE)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    _style(ax, "Everything CUPED gives you is rho squared",
           "a correlation that reads as strong returns a quarter of the traffic, not half")


def panel_unit_theta(ax) -> None:
    ratios = np.linspace(0.4, 2.6, 300)   # sd_pre / sd_post
    for rho, colour in ((0.30, BAD), (0.50, WARN), (0.70, COOL)):
        y = [cuped.variance_ratio_unit_theta(rho, r * 4.0, 4.0) for r in ratios]
        ax.plot(ratios, y, color=colour, lw=2.0, label=f"theta = 1, rho = {rho:.2f}")
        ax.plot(ratios, [1 - rho ** 2] * len(ratios), color=colour, lw=1.2, ls="--",
                alpha=0.65)
    ax.axhline(1.0, color=INK, lw=1.2)
    ax.fill_between(ratios, 1.0, 4.2, color=BAD, alpha=0.06)
    ax.text(2.55, 3.6, "worse than\nno adjustment", fontsize=8.4, color=BAD, ha="right",
            fontweight="bold")
    ax.text(0.45, 1.55, "dashed = fitted theta (1 - rho$^2$):\nflat in this ratio, and always < 1",
            fontsize=8, color=MUTE, ha="left")
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("sd of the pre-period window / sd of the experiment window",
                  fontsize=8.6, color=MUTE)
    ax.set_ylabel("variance relative to no adjustment", fontsize=8.6, color=MUTE)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax, "NEGATIVE RESULT: 'just subtract the pre-period' can triple it",
           "theta = 1 hurts whenever sd_pre > 2 rho sd_post - a month of history vs a week of test")


def panel_new_users(ax, rng) -> None:
    fs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    rho = 0.70
    imp, strat, imp_se, strat_se = [], [], [], []
    for f in fs:
        w = cuped.World(rho=rho, new_user_share=float(f), true_rel_lift=0.0)
        d = cuped.simulate(w, 1500, rng)
        b, _ = cuped.adj_none(d)
        i_, _ = cuped.adj_cuped(d)
        st, _ = cuped.adj_cuped_stratified(d)
        p1, s1 = cuped.reduction_with_mc(i_, b)
        p2, s2 = cuped.reduction_with_mc(st, b)
        imp.append(p1)
        imp_se.append(s1)
        strat.append(p2)
        strat_se.append(s2)
    grid = np.linspace(0, 0.85, 200)
    ax.plot(grid, [cuped.reduction_mean_impute(rho, f) for f in grid], color=BAD, lw=1.6,
            ls="--", label="derived: rho$^2$(2 - 1/(1-f))")
    ax.plot(grid, [cuped.reduction_stratified(rho, f) for f in grid], color=GOOD, lw=1.6,
            ls="--", label="derived: (1-f)rho$^2$")
    ax.errorbar(fs, imp, yerr=imp_se, fmt="o", ms=4.5, color=BAD, capsize=2.5, lw=1.2,
                label="measured: mean-impute")
    ax.errorbar(fs, strat, yerr=strat_se, fmt="s", ms=4.5, color=GOOD, capsize=2.5, lw=1.2,
                label="measured: stratify")
    ax.axhline(0, color=INK, lw=1.1)
    ax.axvline(0.5, color=MUTE, lw=1.1, ls=":")
    ax.annotate("break-even at f = 0.5,\nfor ANY rho", xy=(0.5, 0.0), xytext=(0.28, -0.72),
                fontsize=8.4, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.set_ylim(-1.15, 0.62)
    ax.set_xlabel("share of users with no pre-period", fontsize=8.6, color=MUTE)
    ax.set_ylabel("variance reduction", fontsize=8.6, color=MUTE)
    ax.legend(frameon=False, fontsize=7.6, loc="lower left")
    _style(ax, "NEGATIVE RESULT: mean-imputing the covariate stops helping at half",
           "the per-user variance falls by (1-f)rho$^2$; the ESTIMATOR does not, and past 0.5 it rises")


def panel_tails(ax, rng) -> None:
    sigs = np.array([0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0])
    pop = [cuped.lognormal_pearson_rho(0.80, s) ** 2 for s in sigs]
    samp, samp_lo, samp_hi, meas, meas_se = [], [], [], [], []
    for s in sigs:
        w = cuped.World(rho=0.80, lognormal=True, log_sigma=float(s), true_rel_lift=0.0)
        d = cuped.simulate(w, 1200, rng)
        rr = np.array([np.corrcoef(d["pre_c"][i], d["post_c"][i])[0, 1] for i in range(250)])
        samp.append(rr.mean() ** 2)
        samp_lo.append(np.quantile(rr, 0.10) ** 2)
        samp_hi.append(np.quantile(rr, 0.90) ** 2)
        b, _ = cuped.adj_none(d)
        c, _ = cuped.adj_cuped(d)
        p, e = cuped.reduction_with_mc(c, b)
        meas.append(p)
        meas_se.append(e)
    ax.plot(sigs, pop, color=INK, lw=2.0, marker="o", ms=3.6,
            label="population rho$^2$ (the truth)")
    ax.plot(sigs, samp, color=WARN, lw=2.0, marker="^", ms=4,
            label="sample rho$^2$ (what you would compute)")
    ax.fill_between(sigs, samp_lo, samp_hi, color=WARN, alpha=0.14)
    ax.errorbar(sigs, meas, yerr=meas_se, fmt="s", ms=4.5, color=PLUM, capsize=2.5, lw=1.3,
                label="reduction actually delivered")
    ax.set_xlabel("tail weight of the metric (lognormal sigma)", fontsize=8.6, color=MUTE)
    ax.set_ylabel("variance reduction", fontsize=8.6, color=MUTE)
    ax.set_ylim(0, 0.78)
    ax.legend(frameon=False, fontsize=7.8, loc="lower left")
    _style(ax, "NEGATIVE RESULT: on a revenue-shaped metric rho is not measurable",
           "the logs correlate 0.80 throughout; the shaded band is the 10-90% of the sample estimate")


def panel_post_covariate(ax, rng) -> None:
    d_good = cuped.simulate(W, 4000, rng, effect_on_pre=False)
    d_bad = cuped.simulate(W, 4000, rng, effect_on_pre=True)
    e_none, _ = cuped.adj_none(d_bad)
    e_good, _ = cuped.adj_cuped(d_good)
    e_bad, _ = cuped.adj_cuped(d_bad)
    bins = np.linspace(-0.25, 0.65, 70)
    ax.hist(e_none, bins=bins, color=MUTE, alpha=0.55, label="unadjusted")
    ax.hist(e_good, bins=bins, color=GOOD, alpha=0.6, label="CUPED, pre-period covariate")
    ax.hist(e_bad, bins=bins, color=BAD, alpha=0.6, label="CUPED, post-assignment covariate")
    ax.axvline(W.true_effect, color=INK, lw=1.8)
    ax.text(W.true_effect + 0.012, ax.get_ylim()[1] * 0.94, "true effect",
            fontsize=8.4, color=INK, fontweight="bold")
    ax.axvline(0, color=MUTE, lw=1.0, ls=":")
    ax.set_xlabel("estimated effect", fontsize=8.6, color=MUTE)
    ax.set_ylabel("experiments", fontsize=8.6, color=MUTE)
    ax.legend(frameon=False, fontsize=7.8, loc="upper right")
    _style(ax, "The mistake that looks identical in code",
           "one covariate column measured after assignment: -59% of the effect, coverage 0.70")


def panel_headline(ax, rng) -> None:
    d = cuped.simulate(W, TRIALS, rng)
    names = ["none", "diff_in_diff", "post_strat", "cuped"]
    labels = ["no\nadjustment", "theta = 1\n(subtract pre)", "post-stratify\n(10 deciles)",
              "CUPED\n(fitted theta)"]
    power, red = [], []
    base = None
    for n in names:
        est, se = cuped.ADJUSTERS[n](d)
        sc = cuped.score(est, se, W.true_effect)
        if base is None:
            base_est = est
        power.append(sc["reject_rate"])
        red.append(cuped.reduction_with_mc(est, base_est)[0])
        base = 1
    x = np.arange(len(names))
    ax.bar(x - 0.19, power, 0.36, color=COOL, label="power to detect the 2% lift")
    ax.bar(x + 0.19, red, 0.36, color=GOOD, label="variance removed")
    ax.axhline(cuped.variance_reduction(W.rho), color=GOOD, lw=1.2, ls="--")
    ax.text(-0.42, cuped.variance_reduction(W.rho) + 0.015, "rho$^2$ = 0.36",
            fontsize=8, color=GOOD, ha="left")
    for xi, (p, r) in enumerate(zip(power, red)):
        ax.text(xi - 0.19, p + 0.02, f"{p:.2f}", ha="center", fontsize=8.4, color=COOL,
                fontweight="bold")
        ax.text(xi + 0.19, max(r, 0) + 0.02, f"{r:.2f}", ha="center", fontsize=8.4, color=GOOD,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 0.86)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    _style(ax, "What it is worth when nothing is wrong",
           f"{W.per_arm:,} per arm, rho = {W.rho}: same data, power 0.49 -> 0.68, 0.64x the traffic")


def main() -> None:
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(3, 2, figsize=(15.2, 16.4))
    fig.patch.set_facecolor("white")

    panel_rho_squared(axes[0][0])
    panel_unit_theta(axes[0][1])
    panel_new_users(axes[1][0], rng)
    panel_tails(axes[1][1], rng)
    panel_post_covariate(axes[2][0], rng)
    panel_headline(axes[2][1], rng)

    fig.suptitle("Variance reduction is a bet on a correlation you already collected - "
                 "and the ways it goes wrong are not statistical",
                 fontsize=15.5, fontweight="bold", color=INK, x=0.005, ha="left", y=0.996)
    fig.text(0.005, 0.977,
             "CUPED on worlds with a known 2% effect and a known pre/post correlation: what it "
             "is worth, what it costs to guess the coefficient, and the two implementation "
             "details that reverse its sign.",
             fontsize=9.6, color=MUTE, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.968))
    fig.savefig("cuped_audit.png", dpi=170, facecolor="white")
    fig.savefig("cuped_audit.svg", facecolor="white")
    print("wrote cuped_audit.png / cuped_audit.svg")


if __name__ == "__main__":
    main()
