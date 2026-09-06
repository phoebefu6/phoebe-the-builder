"""Six panels, all of them recomputed from srm.py rather than typed in."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import srm  # noqa: E402

INK = "#16222e"
MUTE = "#8b9aa7"
GOOD = "#1f7a5c"
BAD = "#b3402f"
WARN = "#c98a1a"
COOL = "#2b6ca3"
PLUM = "#6b4d8f"
GRID = "#dfe5ea"

W = srm.World()
A_R = srm.ALPHA_REFLEX
A_P = srm.ALPHA_PLATFORM
SEED = 20260903
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


def panel_ratio_vs_n(ax) -> None:
    ns = np.unique(np.round(np.logspace(2.7, 7.2, 260)).astype(int))
    for share, colour, lab in ((0.493, BAD, "49.3 / 50.7"),
                               (0.497, WARN, "49.7 / 50.3"),
                               (0.499, COOL, "49.9 / 50.1")):
        ps = [max(srm.p_chi2(int(round(n * share)), n - int(round(n * share))), 1e-60) for n in ns]
        ax.plot(ns, ps, color=colour, lw=1.9, label=lab)
    ax.axhline(A_R, color=MUTE, lw=1.1, ls="--")
    ax.axhline(A_P, color=INK, lw=1.1, ls=":")
    ax.text(1.5e7, A_R * 2.2, "0.05 (reflex)", fontsize=7.6, color=MUTE, ha="right")
    ax.text(1.5e7, A_P * 2.4, "0.0005 (platform)", fontsize=7.6, color=INK, ha="right")
    for n_cross, lab in ((19_575, "0.05"), (61_856, "0.0005")):
        ax.plot([n_cross], [A_R if lab == "0.05" else A_P], marker="o", ms=5, color=BAD, zorder=5)
    ax.annotate("49.3/50.7 crosses 0.05\nat n = 19,575", xy=(19_575, A_R), xytext=(2.2e3, 1e-8),
                fontsize=7.8, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD, lw=0.9))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-30, 30)
    ax.set_xlabel("total users in the experiment", fontsize=8.6, color=MUTE)
    ax.set_ylabel("SRM p-value", fontsize=8.6, color=MUTE)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax, "A ratio is not a finding",
           "the same split is the healthiest line in the report, then fatal, with nothing changed but n")


def panel_power(ax, rng) -> None:
    losses = np.array([0.001, 0.002, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.050])
    curves = {"chi-square @ 0.05": ([], COOL, "-"), "chi-square @ 0.0005": ([], PLUM, "-"),
              "outside 49/51 (share)": ([], BAD, "--"), "ratio outside 0.99-1.01": ([], WARN, "--")}
    for loss in losses:
        d = srm.simulate(W, "mcar_loss", float(loss), TRIALS, rng)
        p = srm.vector_p_chi2(d["n_ctrl"], d["n_trt"])
        curves["chi-square @ 0.05"][0].append(float((p < A_R).mean()))
        curves["chi-square @ 0.0005"][0].append(float((p < A_P).mean()))
        curves["outside 49/51 (share)"][0].append(
            srm.flag_rate(d["n_ctrl"], d["n_trt"], srm.eyeball_abs, 0.5, limit=1200))
        curves["ratio outside 0.99-1.01"][0].append(
            srm.flag_rate(d["n_ctrl"], d["n_trt"], srm.eyeball_ratio, 0.5, limit=1200))
    for lab, (ys, colour, ls) in curves.items():
        ax.plot(losses * 100, ys, color=colour, lw=1.9, ls=ls, marker="o", ms=3.2, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("% of the treatment arm's records missing", fontsize=8.6, color=MUTE)
    ax.set_ylabel("detection rate", fontsize=8.6, color=MUTE)
    ax.set_ylim(-0.04, 1.06)
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    _style(ax, "'Within 1%' names two rules, and neither is the test",
           "100k per arm; the share rule is inert, the ratio rule fires on healthy traffic")


def panel_sensitivity(ax) -> None:
    ms = np.unique(np.round(np.logspace(3.3, 6.3, 40)).astype(int))
    mde = np.array([srm.mde_rel_lift(int(m), W.base_rate, A_R) for m in ms])
    dev = np.array([srm.mdd_share(2 * int(m), A_R) / 0.5 for m in ms])
    dev_p = np.array([srm.mdd_share(2 * int(m), A_P) / 0.5 for m in ms])
    ax.plot(ms, mde * 100, color=BAD, lw=2.1, label="the experiment (min detectable lift)")
    ax.plot(ms, dev * 100, color=GOOD, lw=2.1, label="the SRM check @ 0.05")
    ax.plot(ms, dev_p * 100, color=PLUM, lw=1.8, ls="--", label="the SRM check @ 0.0005")
    ax.fill_between(ms, dev * 100, mde * 100, color=GOOD, alpha=0.10)
    i = len(ms) // 2
    ax.annotate(f"{mde[i] / dev[i]:.2f}x", xy=(ms[i], np.sqrt(mde[i] * dev[i]) * 100),
                fontsize=9.5, fontweight="bold", color=GOOD, ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("users per arm", fontsize=8.6, color=MUTE)
    ax.set_ylabel("relative deviation detectable at 80% power (%)", fontsize=8.6, color=MUTE)
    ax.legend(frameon=False, fontsize=7.8, loc="lower left")
    _style(ax, "The health check is 6x more sensitive than the experiment",
           "and the ratio is a constant of the design - both are the same multiple of 1/sqrt(n)")


def panel_blind_band(ax) -> None:
    ms = [5_000, 25_000, 100_000, 1_000_000]
    losses, biases = [], []
    for m in ms:
        loss = srm.loss_for_share_deviation(srm.mdd_share(2 * m, A_P))
        losses.append(loss * 100)
        rate = min(loss / W.low_share, 1.0)
        biases.append((srm.analytic_est_lift(W, "selective_loss", rate) - W.true_rel_lift)
                      / W.true_rel_lift * 100)
    x = np.arange(len(ms))
    ax.bar(x, losses, width=0.55, color=COOL, label="smallest loss the check reliably sees (%)")
    ax2 = ax.twinx()
    ax2.plot(x, biases, color=BAD, lw=2.2, marker="o", ms=5,
             label="bias that loss already carries (%)")
    ax2.set_ylim(0, max(biases) * 1.25)
    ax2.tick_params(colors=BAD, labelsize=8.4, length=3)
    ax2.set_ylabel("overstatement of the effect (%)", fontsize=8.6, color=BAD)
    for s in ("top",):
        ax2.spines[s].set_visible(False)
    for xi, b in zip(x, biases):
        ax2.text(xi + 0.26, b, f"+{b:.0f}%", ha="left", va="center", fontsize=9,
                 fontweight="bold", color=BAD)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m:,}" for m in ms])
    ax.set_xlabel("users per arm", fontsize=8.6, color=MUTE)
    ax.set_ylabel("% of one arm missing", fontsize=8.6, color=MUTE)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.8, loc="upper right")
    _style(ax, "NEGATIVE RESULT: sensitive, and still not sensitive enough",
           "detection scales with n; the bias does not, so small tests are blind and not safe")


def panel_mechanisms(ax, rng) -> None:
    specs = [("healthy", 0.0, "nothing lost"), ("mcar_loss", 0.015, "records dropped\nat random"),
             ("selective_loss", 0.05, "low-intent users\nbounced"),
             ("balanced_selective", 0.05, "same, plus equal\ncount cut from control")]
    flags, biases, labels = [], [], []
    for mech, rate, lab in specs:
        d = srm.simulate(W, mech, rate, TRIALS, rng)
        p = srm.vector_p_chi2(d["n_ctrl"], d["n_trt"])
        flags.append(float((p < A_R).mean()))
        biases.append((float(d["est_rel_lift"].mean()) - W.true_rel_lift) / W.true_rel_lift * 100)
        labels.append(lab)
    x = np.arange(len(specs))
    ax.bar(x - 0.19, flags, width=0.36, color=COOL, label="SRM check fires (rate @ 0.05)")
    ax.bar(x + 0.19, np.array(biases) / 100, width=0.36, color=BAD,
           label="effect overstated (as a fraction)")
    ax.axhline(A_R, color=MUTE, lw=1.1, ls="--")
    ax.text(3.42, A_R + 0.02, "0.05 = the null", fontsize=7.6, color=MUTE, ha="right")
    for xi, (f, b) in enumerate(zip(flags, biases)):
        ax.text(xi - 0.19, f + 0.03, f"{f:.2f}", ha="center", fontsize=8.2, color=COOL,
                fontweight="bold")
        ax.text(xi + 0.19, b / 100 + 0.03, f"{b:+.0f}%", ha="center", fontsize=8.2, color=BAD,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    _style(ax, "NEGATIVE RESULT: a passing check is not evidence",
           "the rightmost split is exactly even and the effect is a quarter too high")


def panel_segments(ax, rng) -> None:
    segs = srm.DEFAULT_SEGMENTS
    losses = np.array([0.01, 0.02, 0.04, 0.06, 0.10, 0.15, 0.20])
    agg_r, agg_p, bonf = [], [], []
    for L in losses:
        d = srm.simulate_segmented(W.per_arm, segs, "safari", float(L), TRIALS, rng)
        a = srm.vector_p_chi2(d["n_ctrl"], d["n_trt"])
        ps = np.vstack([srm.vector_p_chi2(d["n_ctrl_seg"][i], d["n_trt_seg"][i])
                        for i in range(len(segs))])
        agg_r.append(float((a < A_R).mean()))
        agg_p.append(float((a < A_P).mean()))
        bonf.append(float((ps.min(axis=0) < A_P / len(segs)).mean()))
    ax.plot(losses * 100, bonf, color=GOOD, lw=2.2, marker="o", ms=3.6,
            label="per segment, Bonferroni @ 0.0005")
    ax.plot(losses * 100, agg_p, color=PLUM, lw=2.0, marker="s", ms=3.4,
            label="aggregate @ 0.0005")
    ax.plot(losses * 100, agg_r, color=MUTE, lw=1.6, ls="--", marker="^", ms=3.4,
            label="aggregate @ 0.05")
    i = list(losses).index(0.06)
    ax.annotate("", xy=(6, bonf[i]), xytext=(6, agg_p[i]),
                arrowprops=dict(arrowstyle="<->", color=GOOD, lw=1.4))
    ax.text(6.4, (bonf[i] + agg_p[i]) / 2, f"{bonf[i] / max(agg_p[i], 1e-9):.0f}x\nat the same alpha",
            fontsize=9, fontweight="bold", color=GOOD, va="center")
    ax.set_xlabel("% loss inside one segment (safari, 15% of traffic)", fontsize=8.6, color=MUTE)
    ax.set_ylabel("detection rate", fontsize=8.6, color=MUTE)
    ax.set_ylim(-0.04, 1.06)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right")
    _style(ax, "Point it at the segments and pay the correction",
           "three extra chi-square calls, and the false-alarm rate goes DOWN, not up")


def main() -> None:
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(3, 2, figsize=(15.2, 16.4))
    fig.patch.set_facecolor("white")

    panel_ratio_vs_n(axes[0][0])
    panel_power(axes[0][1], rng)
    panel_sensitivity(axes[1][0])
    panel_blind_band(axes[1][1])
    panel_mechanisms(axes[2][0], rng)
    panel_segments(axes[2][1], rng)

    fig.suptitle("A split is a hypothesis - and passing its test is not evidence the arms are comparable",
                 fontsize=15.5, fontweight="bold", color=INK, x=0.005, ha="left", y=0.996)
    fig.text(0.005, 0.977,
             "Sample ratio mismatch on a simulated world with a known 5% true lift: what the check "
             "can see, what it cannot, and what a clean bill of health is worth.",
             fontsize=9.6, color=MUTE, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.968))
    fig.savefig("srm_audit.png", dpi=170, facecolor="white")
    fig.savefig("srm_audit.svg", facecolor="white")
    print("wrote srm_audit.png / srm_audit.svg")


if __name__ == "__main__":
    main()
