"""The experiments that turn the claims in `fitting.py` into numbers.

Each function here is a small, seeded simulation. They are slow relative to a single fit
because every one of them refits distributions thousands of times - that is the cost of
answering a calibration question instead of asserting one.

Every experiment takes explicit `n_*` arguments so the notebook can run a fast version and
the README can quote a bigger one, with the sizes stated next to the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from fitting import (
    Family,
    FitReport,
    _information_criteria,
    bootstrap_ks,
    family,
    fit_distributions,
    fit_params,
    loglikelihood,
    probe_free_location,
    selection_stability,
    support_violation,
)

# --------------------------------------------------------------------------------------
# E1 - the naive KS p-value has no power once you fit the parameters
# --------------------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    label: str
    n: int
    n_datasets: int
    n_boot: int
    alpha: float
    reject_naive: float
    reject_bootstrap: float
    mean_p_naive: float
    mean_p_bootstrap: float
    p_naive_all: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    p_bootstrap_all: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def line(self) -> str:
        return (
            f"{self.label:<26}{self.n:>7}{self.reject_naive:>14.3f}"
            f"{self.reject_bootstrap:>14.3f}{self.mean_p_naive:>12.3f}"
            f"{self.mean_p_bootstrap:>12.3f}"
        )


def ks_calibration(
    fam_name: str = "normal",
    n: int = 200,
    n_datasets: int = 200,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 11,
) -> CalibrationResult:
    """Type-I error rate of both KS tests when the null is TRUE.

    Data is drawn from the family being tested, so a correctly calibrated test rejects
    exactly `alpha` of the time and a uniform p-value distribution has mean 0.5. The naive
    test does neither: fitting the parameters shrinks D, the reference distribution does not
    know that, and the p-values pile up near 1. A test that never rejects when the null is
    true is not conservative in any useful sense - it has thrown away the power it would
    need when the null is false.
    """
    fam = family(fam_name)
    rng = np.random.default_rng(seed)
    truth = _canonical_params(fam)

    rej_naive = rej_boot = 0
    ps_naive: List[float] = []
    ps_boot: List[float] = []
    for _ in range(n_datasets):
        x = fam.dist.rvs(*truth, size=n, random_state=rng)
        params = fit_params(fam, x)
        res = bootstrap_ks(fam, x, params, n_boot=n_boot, rng=rng)
        ps_naive.append(res.p_naive)
        ps_boot.append(res.p_bootstrap)
        rej_naive += int(res.p_naive < alpha)
        rej_boot += int(res.p_bootstrap < alpha)

    return CalibrationResult(
        label=f"{fam_name} (null is true)",
        n=n,
        n_datasets=n_datasets,
        n_boot=n_boot,
        alpha=alpha,
        reject_naive=rej_naive / n_datasets,
        reject_bootstrap=rej_boot / n_datasets,
        mean_p_naive=float(np.mean(ps_naive)),
        mean_p_bootstrap=float(np.mean(ps_boot)),
        p_naive_all=np.asarray(ps_naive),
        p_bootstrap_all=np.asarray(ps_boot),
    )


def ks_power(
    fitted_family: str = "normal",
    true_family: str = "logistic",
    n: int = 200,
    n_datasets: int = 200,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 13,
) -> CalibrationResult:
    """Rejection rate of both tests when the null is FALSE - i.e. actual power.

    Same machinery, data drawn from a different family. Now rejection is the correct answer
    and the naive test's inflated p-values become a straightforward failure to detect.
    """
    fit_fam = family(fitted_family)
    true_fam = family(true_family)
    rng = np.random.default_rng(seed)
    truth = _canonical_params(true_fam)

    rej_naive = rej_boot = 0
    ps_naive: List[float] = []
    ps_boot: List[float] = []
    for _ in range(n_datasets):
        x = true_fam.dist.rvs(*truth, size=n, random_state=rng)
        params = fit_params(fit_fam, x)
        res = bootstrap_ks(fit_fam, x, params, n_boot=n_boot, rng=rng)
        ps_naive.append(res.p_naive)
        ps_boot.append(res.p_bootstrap)
        rej_naive += int(res.p_naive < alpha)
        rej_boot += int(res.p_bootstrap < alpha)

    return CalibrationResult(
        label=f"{fitted_family} on {true_family} data",
        n=n,
        n_datasets=n_datasets,
        n_boot=n_boot,
        alpha=alpha,
        reject_naive=rej_naive / n_datasets,
        reject_bootstrap=rej_boot / n_datasets,
        mean_p_naive=float(np.mean(ps_naive)),
        mean_p_bootstrap=float(np.mean(ps_boot)),
        p_naive_all=np.asarray(ps_naive),
        p_bootstrap_all=np.asarray(ps_boot),
    )


def _canonical_params(fam: Family) -> Tuple[float, ...]:
    """A fixed, reasonable parameter vector per family, for simulation."""
    table: Dict[str, Tuple[float, ...]] = {
        "normal": (0.0, 1.0),
        "logistic": (0.0, 0.55),
        "student_t": (4.0, 0.0, 1.0),
        "uniform": (0.0, 1.0),
        "lognormal": (0.85, 0.0, 60.0),
        "gamma": (2.4, 0.0, 18.0),
        "weibull": (1.6, 0.0, 40.0),
        "exponential": (0.0, 25.0),
        "pareto": (2.5, 0.0, 1.0),
        "beta": (2.0, 5.0, 0.0, 1.0),
    }
    if fam.name not in table:
        raise KeyError(f"no canonical parameters for {fam.name}")
    return table[fam.name]


def calibration_table(results: Sequence[CalibrationResult]) -> str:
    header = (
        f"{'setting':<26}{'n':>7}{'reject naive':>14}{'reject boot':>14}"
        f"{'mean p nv':>12}{'mean p bt':>12}"
    )
    lines = [header, "-" * len(header)]
    lines.extend(r.line() for r in results)
    lines.append("-" * len(header))
    lines.append(
        "rows 1-2: null TRUE, target = alpha. rows 3+: null FALSE, higher is better."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# E2 - AIC always produces a winner, including when nothing fits
# --------------------------------------------------------------------------------------


@dataclass
class NoFitResult:
    label: str
    n: int
    winner: str
    winner_weight: float
    winner_p_boot: float
    winner_win_share: float
    n_adequate: int
    n_candidates: int
    runner_up: Optional[str] = None
    runner_up_weight: float = float("nan")

    def line(self) -> str:
        return (
            f"{self.label:<28}{self.n:>7}{self.winner:<13}{self.winner_weight:>9.3f}"
            f"{self.winner_p_boot:>10.3f}{self.winner_win_share:>9.0%}"
            f"{self.n_adequate:>7}/{self.n_candidates}"
        )


def mixture_sample(n: int, seed: int = 17) -> np.ndarray:
    """A two-component lognormal mixture: outside every candidate family, by construction.

    This is what real latency looks like - a fast path and a slow path - and it is exactly
    the shape that a single-family fit cannot represent no matter which family wins.
    """
    rng = np.random.default_rng(seed)
    n_slow = int(round(0.12 * n))
    fast = rng.lognormal(mean=3.0, sigma=0.35, size=n - n_slow)
    slow = rng.lognormal(mean=5.4, sigma=0.55, size=n_slow)
    out = np.concatenate([fast, slow])
    rng.shuffle(out)
    return out


def _summarise(label: str, report: FitReport) -> NoFitResult:
    ranked = report.ranked
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    return NoFitResult(
        label=label,
        n=report.diagnostics.n,
        winner=best.name,
        winner_weight=best.aic_weight,
        winner_p_boot=best.ks.p_bootstrap if best.ks else float("nan"),
        winner_win_share=best.win_share,
        n_adequate=len(report.adequate),
        n_candidates=len(ranked),
        runner_up=runner.name if runner else None,
        runner_up_weight=runner.aic_weight if runner else float("nan"),
    )


def confident_winner_on_unfittable_data(
    n: int = 1200,
    n_boot: int = 150,
    stability_reps: int = 60,
    seed: int = 17,
) -> Tuple[NoFitResult, NoFitResult, FitReport, FitReport]:
    """Side by side: a column that IS one of the candidates, and a column that is not.

    The AIC table looks the same in both cases - a winner, a delta, a weight near 1. Only the
    bootstrap KS column distinguishes "this is the distribution" from "this is the closest
    thing in a list that does not contain the distribution".
    """
    rng = np.random.default_rng(seed)
    true_lognormal = rng.lognormal(mean=4.2, sigma=0.85, size=n)
    mixture = mixture_sample(n, seed=seed + 1)

    rep_true = fit_distributions(
        true_lognormal, n_boot=n_boot, stability_reps=stability_reps, seed=seed, probe_location=False
    )
    rep_mix = fit_distributions(
        mixture, n_boot=n_boot, stability_reps=stability_reps, seed=seed, probe_location=False
    )
    return (
        _summarise("truly lognormal", rep_true),
        _summarise("lognormal mixture", rep_mix),
        rep_true,
        rep_mix,
    )


def nofit_table(results: Sequence[NoFitResult]) -> str:
    header = (
        f"{'data':<28}{'n':>7}{'AIC winner':<13}{'weight':>9}{'p boot':>10}"
        f"{'win%':>9}{'adequate':>10}"
    )
    lines = [header, "-" * len(header)]
    lines.extend(r.line() for r in results)
    lines.append("-" * len(header))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# E3 - selection stability vs the delta-AIC rule of thumb
# --------------------------------------------------------------------------------------


@dataclass
class StabilityRow:
    n: int
    true_family: str
    winner: str
    delta_to_true: float
    weight_true: float
    win_share_true: float
    win_share_winner: float
    reps: int

    def line(self) -> str:
        return (
            f"{self.n:>7}  {self.winner:<12}{self.delta_to_true:>11.2f}"
            f"{self.weight_true:>11.3f}{self.win_share_true:>13.0%}"
            f"{self.win_share_winner:>13.0%}"
        )


def stability_vs_n(
    true_family: str = "gamma",
    rivals: Sequence[str] = ("gamma", "lognormal", "weibull", "normal"),
    sizes: Sequence[int] = (100, 400, 1600, 6400),
    reps: int = 80,
    seed: int = 23,
) -> List[StabilityRow]:
    """How often does the true family actually win, as n grows?

    Gamma and lognormal are close over most of their parameter space; at n=100 the AIC
    winner is close to a coin flip and the Akaike weight does not say so, because a weight
    is computed from the one delta you happened to observe. The bootstrap win share is the
    same quantity the weight is trying to approximate, measured instead of assumed.
    """
    fams = [family(name) for name in rivals]
    true_fam = family(true_family)
    truth = _canonical_params(true_fam)

    rows: List[StabilityRow] = []
    for n in sizes:
        rng = np.random.default_rng(seed + n)
        x = true_fam.dist.rvs(*truth, size=n, random_state=rng)
        fits: Dict[str, float] = {}
        for fam in fams:
            if support_violation(fam, x) is not None:
                continue
            try:
                params = fit_params(fam, x)
                ll = loglikelihood(fam, x, params)
            except Exception:  # noqa: BLE001
                continue
            if math.isfinite(ll):
                fits[fam.name] = _information_criteria(ll, fam.n_free, n)[0]
        if not fits:
            continue
        winner = min(fits, key=lambda k: fits[k])
        best_aic = fits[winner]
        weights = {k: math.exp(-0.5 * (v - best_aic)) for k, v in fits.items()}
        total = sum(weights.values())
        shares = selection_stability(x, fams, n_boot=reps, rng=rng)
        rows.append(
            StabilityRow(
                n=n,
                true_family=true_family,
                winner=winner,
                delta_to_true=fits.get(true_family, float("nan")) - best_aic,
                weight_true=weights.get(true_family, float("nan")) / total,
                win_share_true=shares.get(true_family, float("nan")),
                win_share_winner=shares.get(winner, float("nan")),
                reps=reps,
            )
        )
    return rows


def stability_table(rows: Sequence[StabilityRow]) -> str:
    header = (
        f"{'n':>7}  {'AIC winner':<12}{'dAIC(true)':>11}{'w(true)':>11}"
        f"{'win%(true)':>13}{'win%(winner)':>13}"
    )
    lines = [header, "-" * len(header)]
    lines.extend(r.line() for r in rows)
    lines.append("-" * len(header))
    if rows:
        lines.append(
            f"true family = {rows[0].true_family}; win shares over {rows[0].reps} "
            "nonparametric bootstrap resamples"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# E4 - rounding, n, and the rejection of the true family
# --------------------------------------------------------------------------------------


@dataclass
class RoundingRow:
    n: int
    decimals: Optional[int]
    tie_fraction: float
    d_observed: float
    p_bootstrap: float
    rejected: bool

    def line(self) -> str:
        dec = "raw" if self.decimals is None else f"{self.decimals}dp"
        return (
            f"{self.n:>8}{dec:>7}{self.tie_fraction:>10.3f}{self.d_observed:>10.4f}"
            f"{self.p_bootstrap:>11.3f}{'REJECT' if self.rejected else 'keep':>9}"
        )


def rounding_vs_n(
    sizes: Sequence[int] = (100, 500, 2000, 8000, 20000),
    decimals: Optional[int] = 1,
    n_boot: int = 150,
    alpha: float = 0.05,
    seed: int = 29,
) -> List[RoundingRow]:
    """Fit a normal to normal data that has been rounded, and sweep n.

    The data IS normal. The only defect is that it was recorded to one decimal place, which
    every real measurement is. The KS distance barely moves with n; the reference
    distribution shrinks like 1/sqrt(n), so past some sample size the true family is
    rejected on the rounding alone. This is not a bug in the test - it is the test correctly
    answering a question nobody wanted to ask.
    """
    fam = family("normal")
    rows: List[RoundingRow] = []
    for n in sizes:
        rng = np.random.default_rng(seed + n)
        x = rng.normal(0.0, 1.0, size=n)
        if decimals is not None:
            x = np.round(x, decimals)
        params = fit_params(fam, x)
        res = bootstrap_ks(fam, x, params, n_boot=n_boot, rng=rng)
        n_unique = int(np.unique(x).size)
        rows.append(
            RoundingRow(
                n=n,
                decimals=decimals,
                tie_fraction=1.0 - n_unique / n,
                d_observed=res.d_observed,
                p_bootstrap=res.p_bootstrap,
                rejected=res.p_bootstrap < alpha,
            )
        )
    return rows


def rounding_table(rows: Sequence[RoundingRow]) -> str:
    header = f"{'n':>8}{'round':>7}{'ties':>10}{'KS D':>10}{'p boot':>11}{'verdict':>9}"
    lines = [header, "-" * len(header)]
    lines.extend(r.line() for r in rows)
    lines.append("-" * len(header))
    lines.append("data is genuinely normal in every row; only n and the rounding differ")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# E5 - what a free location parameter buys
# --------------------------------------------------------------------------------------


@dataclass
class LocationRow:
    family_name: str
    is_true_family: bool
    loglik_gain: float
    aic_pinned: float
    aic_free: float
    loc_free: float
    data_min: float
    invalid: bool

    @property
    def free_wins(self) -> bool:
        return self.aic_free < self.aic_pinned

    def line(self) -> str:
        gain = "-inf" if not math.isfinite(self.loglik_gain) else f"{self.loglik_gain:.2f}"
        aic_free = "inf" if not math.isfinite(self.aic_free) else f"{self.aic_free:.1f}"
        pick = "INVALID" if self.invalid else ("free" if self.free_wins else "pinned")
        truth = "true family" if self.is_true_family else "wrong family"
        return (
            f"{self.family_name:<12}{truth:<14}{gain:>10}"
            f"{self.aic_pinned:>12.1f}{aic_free:>12}"
            f"{self.loc_free:>10.3f}{self.data_min:>10.3f}{pick:>9}"
        )


def free_location_cost(
    n: int = 1200,
    seed: int = 31,
    families: Sequence[str] = ("gamma", "lognormal", "weibull"),
) -> List[LocationRow]:
    """Refit positive-support families with loc free, on gamma data, and count the damage.

    The data is gamma with loc=0, so the third parameter has nothing to estimate. Three
    things happen instead, and none of them is "it recovers zero":

    - gamma (the TRUE family) gains essentially no log-likelihood. The parameter is
      genuinely redundant, so AIC's +2 penalty correctly rejects it.
    - lognormal (a WRONG family) gains a large amount, because the shift lets it imitate a
      gamma. The free parameter's value is proportional to how wrong the family is, which is
      precisely backwards for a contest meant to identify the right one.
    - weibull's optimiser walks into the loc -> min(x) singularity and returns a loc ABOVE
      the smallest observation. The resulting model assigns zero density to real data
      points; its log-likelihood is -inf.

    A single AIC table that mixes 2-parameter and 3-parameter fits of the same families is
    scoring all three of these against each other as if they were comparable.
    """
    rng = np.random.default_rng(seed)
    x = rng.gamma(shape=2.4, scale=18.0, size=n)
    rows: List[LocationRow] = []
    for name in families:
        probe = probe_free_location(family(name), x)
        if probe is None:
            continue
        rows.append(
            LocationRow(
                family_name=name,
                is_true_family=(name == "gamma"),
                loglik_gain=probe.loglik_gain,
                aic_pinned=probe.aic_pinned,
                aic_free=probe.aic_free,
                loc_free=probe.loc_free,
                data_min=probe.data_min,
                invalid=probe.free_fit_invalid,
            )
        )
    return rows


def location_table(rows: Sequence[LocationRow]) -> str:
    header = (
        f"{'family':<12}{'role':<14}{'dlogLik':>10}{'AIC pin':>12}{'AIC free':>12}"
        f"{'loc hat':>10}{'min(x)':>10}{'AIC pick':>9}"
    )
    lines = [header, "-" * len(header)]
    lines.extend(r.line() for r in rows)
    lines.append("-" * len(header))
    lines.append(
        "data is gamma(2.4, 18) with loc = 0; the third parameter has nothing to find. "
        "INVALID = fitted loc exceeds min(x), so observed points have zero density."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# E6 - what the wrong distribution costs downstream
# --------------------------------------------------------------------------------------


@dataclass
class TailRow:
    family_name: str
    is_winner: bool
    q99: float
    q999: float
    empirical_q99: float
    empirical_q999: float

    @property
    def err99(self) -> float:
        return 100.0 * (self.q99 - self.empirical_q99) / self.empirical_q99

    @property
    def err999(self) -> float:
        return 100.0 * (self.q999 - self.empirical_q999) / self.empirical_q999

    def line(self) -> str:
        tag = "AIC winner" if self.is_winner else ""
        return (
            f"{self.family_name:<12}{tag:<12}{self.q99:>10.1f}{self.err99:>+9.0f}%"
            f"{self.q999:>12.1f}{self.err999:>+9.0f}%"
        )


def tail_error(
    x: Optional[np.ndarray] = None,
    families: Sequence[str] = ("student_t", "lognormal", "weibull", "gamma"),
) -> List[TailRow]:
    """What the fitted quantiles say versus what the data says, at p99 and p99.9.

    This is where a distribution fit is actually spent: capacity planning, SLA targets, VaR,
    simulation inputs. All of those read a tail quantile off the fitted model, and the tail
    is the part of the fit that the body of the data constrains least. When no candidate is
    adequate, the tail errors are not small - and crucially they do not all point the same
    way, so there is no safe direction to round.
    """
    from fitting import fit_params, sample_book  # local import keeps module import cheap

    if x is None:
        x = sample_book()["latency_ms"]
    emp99 = float(np.percentile(x, 99))
    emp999 = float(np.percentile(x, 99.9))

    rep = fit_distributions(x, n_boot=0, stability_reps=0, probe_location=False)
    winner = rep.best.name if rep.best else ""

    rows: List[TailRow] = []
    for name in families:
        fam = family(name)
        if support_violation(fam, x) is not None:
            continue
        params = fit_params(fam, x)
        rows.append(
            TailRow(
                family_name=name,
                is_winner=(name == winner),
                q99=float(fam.dist.ppf(0.99, *params)),
                q999=float(fam.dist.ppf(0.999, *params)),
                empirical_q99=emp99,
                empirical_q999=emp999,
            )
        )
    return rows


def tail_table(rows: Sequence[TailRow]) -> str:
    if not rows:
        return "no families in support"
    header = (
        f"{'family':<12}{'':<12}{'p99':>10}{'err':>10}{'p99.9':>12}{'err':>10}"
    )
    lines = [header, "-" * len(header)]
    lines.extend(r.line() for r in rows)
    lines.append("-" * len(header))
    lines.append(
        f"empirical p99 = {rows[0].empirical_q99:.1f}, "
        f"empirical p99.9 = {rows[0].empirical_q999:.1f}"
    )
    return "\n".join(lines)
