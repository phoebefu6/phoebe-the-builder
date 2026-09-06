"""Render the six-panel audit figure used in the README and the notebook.

Each panel answers one of the questions the tool exists to answer, and the panels are
ordered so that the two on the left are the ones a normal distribution-fitting workflow
already produces, while the four on the right are the ones it leaves out.

Run: python3 make_chart.py [output.png]
"""

from __future__ import annotations

import sys
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evidence import ks_calibration, mixture_sample, rounding_vs_n
from fitting import (
    bootstrap_ks,
    family,
    fit_distributions,
    fit_params,
    qq_points,
    sample_book,
)

INK = "#1b1b1f"
GRID = "#dcdce4"
PASS = "#2a7f62"
FAIL = "#b4451f"
ACCENT = "#4460a0"
MUTED = "#9a9aa8"


def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK)
    ax.tick_params(labelsize=7.5, colors=INK, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def panel_qq(ax: plt.Axes, x: np.ndarray, fam_name: str, label: str, passed: bool) -> None:
    fam = family(fam_name)
    params = fit_params(fam, x)
    theo, emp = qq_points(fam, x, params)
    colour = PASS if passed else FAIL
    # Clip to the observed range. A Student-t fit with low df puts its extreme quantiles
    # thousands of units out, and letting the axis follow them hides the departure that
    # matters by compressing the entire body of the data into one pixel.
    lo = float(np.min(emp))
    hi = float(np.max(emp))
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, linewidth=1.0,
            linestyle="--", zorder=1)
    ax.scatter(theo, emp, s=5, color=colour, alpha=0.55, linewidths=0, zorder=2)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    off = int(np.sum((theo < lo - pad) | (theo > hi + pad)))
    if off:
        noun = "quantile" if off == 1 else "quantiles"
        ax.text(0.03, 0.93, f"{off} fitted {noun} off-scale", transform=ax.transAxes,
                fontsize=7.5, color=FAIL)
    _style(ax, label, f"{fam_name} quantile", "observed quantile")


def panel_pdf_overlay(ax: plt.Axes, x: np.ndarray, names: Sequence[str]) -> None:
    bins = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 70)
    ax.hist(
        x, bins=bins, density=True, color="#e6e6ee", edgecolor="#c9c9d6", linewidth=0.3, zorder=1
    )
    grid = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 600)
    colours = [FAIL, ACCENT, MUTED]
    for name, colour in zip(names, colours):
        fam = family(name)
        params = fit_params(fam, x)
        ax.plot(grid, fam.dist.pdf(grid, *params), color=colour, linewidth=1.6, label=name)
    ax.set_xscale("log")
    ax.legend(fontsize=7.5, frameon=False)
    _style(
        ax,
        "Two modes, and no candidate has two",
        "latency (ms, log scale)",
        "density",
    )


def panel_pvalue_histograms(ax: plt.Axes, cal) -> None:  # noqa: ANN001 - CalibrationResult
    ax.hist(
        cal.p_naive_all,
        bins=np.linspace(0, 1, 21),
        color=FAIL,
        alpha=0.65,
        label=f"naive (rejects {cal.reject_naive:.0%})",
    )
    ax.hist(
        cal.p_bootstrap_all,
        bins=np.linspace(0, 1, 21),
        color=PASS,
        alpha=0.65,
        label=f"bootstrap (rejects {cal.reject_bootstrap:.0%})",
    )
    ax.axhline(
        len(cal.p_naive_all) / 20.0, color=INK, linewidth=1.0, linestyle="--",
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _style(
        ax,
        "p-values when the null is TRUE (should be flat)",
        "p-value",
        "datasets",
    )


def panel_weight_vs_stability(ax: plt.Axes, report, subtitle: str) -> None:  # noqa: ANN001
    ranked = [r for r in report.ranked if r.win_share > 0.005 or r.aic_weight > 0.005][:5]
    names = [r.name for r in ranked]
    idx = np.arange(len(names))
    ax.barh(idx + 0.19, [r.aic_weight for r in ranked], height=0.36, color=ACCENT,
            label="Akaike weight")
    ax.barh(idx - 0.19, [r.win_share for r in ranked], height=0.36, color=PASS,
            label="bootstrap win share")
    for i, r in enumerate(ranked):
        ax.text(r.aic_weight + 0.015, i + 0.19, f"{r.aic_weight:.2f}", fontsize=7,
                va="center", color=ACCENT)
        ax.text(r.win_share + 0.015, i - 0.19, f"{r.win_share:.2f}", fontsize=7,
                va="center", color=PASS)
    ax.set_yticks(idx)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    _style(ax, subtitle, "share", "")


def _mean_rounding_curve(sizes, decimals, n_boot, seeds):  # noqa: ANN001, ANN202
    """Average the bootstrap p-value over several independent datasets per n.

    A single dataset per point produces a curve that jumps between 0.1 and 0.9 for reasons
    that have nothing to do with n, because a p-value under a true null IS uniform noise.
    Averaging over replicates plots the thing being claimed - how the typical p-value moves
    with n - instead of one draw from it.
    """
    out = []
    for n in sizes:
        vals = [
            rounding_vs_n(sizes=(n,), decimals=decimals, n_boot=n_boot, seed=s)[0].p_bootstrap
            for s in seeds
        ]
        out.append(float(np.mean(vals)))
    return out


def panel_rounding(ax: plt.Axes, sizes, rounded, raw) -> None:  # noqa: ANN001
    ax.plot(sizes, rounded, marker="o", markersize=4, color=FAIL, linewidth=1.4,
            label="rounded to 1 dp")
    ax.plot(sizes, raw, marker="o", markersize=4, color=PASS, linewidth=1.4,
            label="unrounded")
    ax.axhline(0.05, color=INK, linewidth=1.0, linestyle="--")
    ax.text(sizes[0], 0.075, "alpha = 0.05", fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7.5, frameon=False)
    _style(
        ax,
        "The data is genuinely normal in both lines",
        "n (log scale)",
        "mean bootstrap KS p-value",
    )


def panel_null_distribution(ax: plt.Axes, x: np.ndarray, fam_name: str) -> None:
    fam = family(fam_name)
    params = fit_params(fam, x)
    rng = np.random.default_rng(3)
    res = bootstrap_ks(fam, x, params, n_boot=400, rng=rng)
    ax.hist(res.null_distribution, bins=40, color="#dfe4f1", edgecolor="#c2c9df",
            linewidth=0.3, label="refit KS null")
    ax.axvline(res.d_observed, color=FAIL, linewidth=1.8,
               label=f"observed D = {res.d_observed:.4f}")
    ax.legend(fontsize=7.5, frameon=False)
    _style(
        ax,
        f"Why the naive p-value is wrong ({fam_name} fit)",
        "KS distance",
        "bootstrap replicates",
    )


def build(output: str = "fit_audit.png") -> str:
    book = sample_book()
    mixture = mixture_sample(1500, seed=17)

    print("  running calibration simulation ...")
    cal = ks_calibration("normal", n=180, n_datasets=250, n_boot=150, seed=11)
    print("  running rounding sweep ...")
    sizes = (100, 400, 1600, 6400, 20000)
    seeds = (29, 131, 233, 337, 439)
    rounded = _mean_rounding_curve(sizes, 1, 150, seeds)
    raw = _mean_rounding_curve(sizes, None, 150, seeds)
    print("  fitting a small gamma sample for the stability panel ...")
    # n=150 gamma: close enough to lognormal that the AIC weight is confident and the
    # bootstrap is not. That gap is the panel.
    small_gamma = np.random.default_rng(41).gamma(2.4, 18.0, 150)
    rep_small = fit_distributions(
        small_gamma,
        families=[family(n) for n in ("gamma", "lognormal", "weibull", "exponential")],
        n_boot=0,
        stability_reps=300,
        seed=41,
        probe_location=False,
    )

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0), dpi=200)
    fig.patch.set_facecolor("white")

    panel_qq(axes[0][0], book["session_seconds"], "lognormal",
             "Passes: lognormal on lognormal data", passed=True)
    panel_qq(axes[0][1], mixture, "student_t",
             "Fails: the AIC winner on mixture data", passed=False)
    panel_pdf_overlay(axes[0][2], mixture, ["student_t", "lognormal", "weibull"])

    panel_pvalue_histograms(axes[1][0], cal)
    panel_weight_vs_stability(
        axes[1][1], rep_small, "Weight says certain; resampling disagrees (n=150 gamma)"
    )
    panel_rounding(axes[1][2], sizes, rounded, raw)

    fig.suptitle(
        "Distribution Fitter - the four questions an AIC table does not answer",
        fontsize=13.5,
        color=INK,
        fontweight="bold",
        x=0.008,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.008,
        0.952,
        "Left column: what a normal workflow already shows.  "
        "Everything else: whether the winner is adequate, stable, and not an artefact of rounding.",
        fontsize=9.0,
        color="#55555f",
        ha="left",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.935))
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "fit_audit.png"
    print(f"building {out}")
    print(f"wrote {build(out)}")
