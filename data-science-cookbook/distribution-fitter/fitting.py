"""Distribution fitting that reports whether anything actually fits.

`scipy.stats.<dist>.fit(x)` is one line, and AIC ranking is three more. The reason
distribution-fitting ships wrong answers is not the arithmetic - it is that every step of
the standard workflow answers a *relative* question and gets read as an *absolute* one.

Five mechanisms:

1. AIC ALWAYS PICKS A WINNER. Ranking by AIC tells you which candidate is least bad. It
   cannot tell you that all of them are bad. Data from a mixture, or a Gumbel, or anything
   outside the candidate set still produces a confident-looking table with a winner and an
   Akaike weight of 0.98. The absolute question needs its own test.
2. THE NAIVE KS P-VALUE IS WRONG WHEN YOU FIT THE PARAMETERS FIRST. `ks_1samp(x, dist.cdf,
   args=fitted)` assumes the parameters were known before seeing the data. Estimating them
   pulls the fitted CDF toward the sample, shrinking D, so the p-value is far too large and
   the test has almost no power. The fix is a parametric bootstrap that REFITS on every
   simulated sample (the Lilliefors construction, generalised).
3. SUPPORT IS A CONSTRAINT, NOT A NUISANCE. A lognormal cannot describe a column containing
   zero. Silently dropping the offending rows changes the data being compared, so the AIC is
   no longer on the same footing as the other candidates. Excluded means excluded, with a
   reason printed.
4. A FREE LOCATION PARAMETER IS NOT FREE. `lognorm.fit(x)` estimates a third parameter that
   drifts to just under min(x), buying log-likelihood from the boundary rather than from
   shape. It wins AIC contests against honestly-parameterised rivals and generalises worse.
   Positive-support families are fit with loc pinned at 0 by default, and the difference is
   reported rather than hidden.
5. GOODNESS OF FIT IS A QUESTION ABOUT n. No real column is exactly parametric. At n=50
   nothing is distinguishable; at n=50000 the true family is rejected too, usually on a
   rounding artefact. Ties are measured and reported, because heavy rounding invalidates a
   continuous fit before any p-value is computed.

Plus bootstrap selection stability, because "delta-AIC > 2" is a rule of thumb about the
statistic, not about how often that winner would win again on resampled data.

No database, no network. numpy + scipy only.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

REAL = "real"
POSITIVE = "positive"
UNIT = "unit"


# --------------------------------------------------------------------------------------
# Candidate families
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    """One candidate distribution.

    `n_free` counts the parameters actually estimated *after* the fixed ones are pinned.
    Getting this count wrong is the quiet way to corrupt an AIC table: scipy always returns
    a loc and a scale, whether or not they were free.
    """

    name: str
    dist_name: str
    support: str
    fix_loc: bool = False
    fix_scale: bool = False
    n_free: int = 2
    note: str = ""

    @property
    def dist(self):  # noqa: ANN201 - scipy frozen-dist factory, no useful annotation
        return getattr(stats, self.dist_name)


DEFAULT_FAMILIES: Tuple[Family, ...] = (
    Family("normal", "norm", REAL, n_free=2, note="loc, scale"),
    Family("logistic", "logistic", REAL, n_free=2, note="heavier tails than normal"),
    Family("student_t", "t", REAL, n_free=3, note="df, loc, scale - df is a real parameter"),
    Family("uniform", "uniform", REAL, n_free=2, note="boundary MLE: loc=min, scale=range"),
    Family("lognormal", "lognorm", POSITIVE, fix_loc=True, n_free=2, note="loc pinned at 0"),
    Family("gamma", "gamma", POSITIVE, fix_loc=True, n_free=2, note="loc pinned at 0"),
    Family("weibull", "weibull_min", POSITIVE, fix_loc=True, n_free=2, note="loc pinned at 0"),
    Family("exponential", "expon", POSITIVE, fix_loc=True, n_free=1, note="one free parameter"),
    Family("pareto", "pareto", POSITIVE, fix_loc=True, n_free=2, note="boundary MLE: scale=min"),
    Family("beta", "beta", UNIT, fix_loc=True, fix_scale=True, n_free=2, note="needs 0<x<1"),
)

FAMILIES_BY_NAME: Dict[str, Family] = {f.name: f for f in DEFAULT_FAMILIES}


def family(name: str) -> Family:
    if name not in FAMILIES_BY_NAME:
        raise KeyError(f"unknown family {name!r}; known: {sorted(FAMILIES_BY_NAME)}")
    return FAMILIES_BY_NAME[name]


# --------------------------------------------------------------------------------------
# Mechanism 3 - support is a constraint
# --------------------------------------------------------------------------------------


def support_violation(fam: Family, x: np.ndarray) -> Optional[str]:
    """Return a human-readable reason this family cannot describe `x`, or None.

    Deliberately not a filter. A family whose support excludes some of the data is dropped
    from the comparison entirely, because fitting it to the surviving subset would put its
    AIC on a different dataset than everyone else's.
    """
    if fam.support == POSITIVE:
        bad = int(np.sum(x <= 0))
        if bad:
            return f"{bad} value(s) <= 0; support is x > 0"
    if fam.support == UNIT:
        bad = int(np.sum((x <= 0) | (x >= 1)))
        if bad:
            return f"{bad} value(s) outside (0, 1)"
    return None


# --------------------------------------------------------------------------------------
# Data diagnostics - mechanism 5's precondition
# --------------------------------------------------------------------------------------


@dataclass
class DataDiagnostics:
    n: int
    n_unique: int
    tie_fraction: float
    minimum: float
    maximum: float
    n_nonpositive: int
    decimals: Optional[int]
    skew: float
    excess_kurtosis: float

    @property
    def heavily_tied(self) -> bool:
        """True when ties are common enough to inflate the KS statistic on their own.

        The threshold is a judgement call, not a theorem. Below it the continuous
        approximation is usually survivable; above it a rejection tells you the data is
        rounded, which you already knew.
        """
        return self.tie_fraction >= 0.05

    def describe(self) -> str:
        lines = [
            f"n = {self.n},  unique = {self.n_unique},  tie fraction = {self.tie_fraction:.3f}",
            f"range = [{self.minimum:.4g}, {self.maximum:.4g}],  values <= 0: {self.n_nonpositive}",
            f"skew = {self.skew:+.3f},  excess kurtosis = {self.excess_kurtosis:+.3f}",
        ]
        if self.decimals is not None:
            lines.append(
                f"values are rounded to {self.decimals} decimal place(s) - the data is "
                "discrete, every continuous fit is an approximation"
            )
        if self.heavily_tied:
            lines.append(
                "WARNING: heavy ties. The KS statistic is inflated by the ECDF's jumps "
                "regardless of shape, so a rejection here is evidence about rounding, "
                "not about the family."
            )
        return "\n".join(lines)


def _infer_decimals(x: np.ndarray, max_places: int = 6) -> Optional[int]:
    """Smallest d such that every value is exactly its own d-decimal rounding, else None.

    Exact equality against `np.round(x, d)` rather than a tolerance: a relative tolerance
    scales with the data and will declare any large-magnitude float "rounded", which is how
    you end up warning about discreteness on a perfectly continuous column.
    """
    for d in range(max_places + 1):
        if np.array_equal(x, np.round(x, d)):
            return d
    return None


def diagnose(x: Sequence[float]) -> DataDiagnostics:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise ValueError("need at least 2 finite observations")
    n = int(arr.size)
    n_unique = int(np.unique(arr).size)
    return DataDiagnostics(
        n=n,
        n_unique=n_unique,
        tie_fraction=1.0 - n_unique / n,
        minimum=float(arr.min()),
        maximum=float(arr.max()),
        n_nonpositive=int(np.sum(arr <= 0)),
        decimals=_infer_decimals(arr),
        skew=float(stats.skew(arr)),
        excess_kurtosis=float(stats.kurtosis(arr)),
    )


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------


def _fit_kwargs(fam: Family) -> Dict[str, float]:
    kw: Dict[str, float] = {}
    if fam.fix_loc:
        kw["floc"] = 0.0
    if fam.fix_scale:
        kw["fscale"] = 1.0
    return kw


def fit_params(fam: Family, x: np.ndarray) -> Tuple[float, ...]:
    """MLE fit with the family's fixed parameters pinned. Raises on failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = fam.dist.fit(x, **_fit_kwargs(fam))
    params = tuple(float(p) for p in params)
    if not all(math.isfinite(p) for p in params):
        raise RuntimeError(f"{fam.name}: non-finite parameters {params}")
    return params


def loglikelihood(fam: Family, x: np.ndarray, params: Sequence[float]) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = fam.dist.logpdf(x, *params)
    if not np.all(np.isfinite(lp)):
        return -np.inf
    return float(np.sum(lp))


def ks_statistic(fam: Family, x: np.ndarray, params: Sequence[float]) -> float:
    """One-sample KS distance between the ECDF of `x` and the fitted CDF.

    Computed directly rather than via `ks_1samp` so that the same code path is used for the
    observed sample and for every bootstrap replicate - a mismatch there is the kind of bug
    that produces a beautifully calibrated wrong answer.
    """
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.size
    cdf = fam.dist.cdf(xs, *params)
    d_plus = np.max(np.arange(1, n + 1) / n - cdf)
    d_minus = np.max(cdf - np.arange(0, n) / n)
    return float(max(d_plus, d_minus))


# --------------------------------------------------------------------------------------
# Mechanism 2 - the parametric bootstrap KS test
# --------------------------------------------------------------------------------------


@dataclass
class BootstrapKS:
    d_observed: float
    p_naive: float
    p_bootstrap: float
    n_replicates: int
    n_failed: int
    null_distribution: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def bootstrap_ks(
    fam: Family,
    x: np.ndarray,
    params: Sequence[float],
    n_boot: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> BootstrapKS:
    """Parametric-bootstrap p-value for the KS statistic with estimated parameters.

    The whole content of this function is the `fit_params` call inside the loop. Simulating
    from the fitted distribution and comparing against the *same* fitted CDF reproduces the
    textbook KS null and is just as wrong as the naive test. Refitting each replicate
    reproduces the estimation-shrinkage that the observed D also enjoyed, which is what
    makes the two comparable.

    The p-value uses the add-one estimator (1 + #{D_b >= D_obs}) / (1 + B), so it is never
    exactly zero - with B replicates you cannot resolve a p-value below 1/(B+1) and the
    number should not pretend otherwise.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n = int(np.asarray(x).size)
    d_obs = ks_statistic(fam, x, params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_naive = float(stats.ks_1samp(x, fam.dist.cdf, args=params).pvalue)

    sims: List[float] = []
    failed = 0
    for _ in range(n_boot):
        xs = fam.dist.rvs(*params, size=n, random_state=rng)
        try:
            params_b = fit_params(fam, xs)
            sims.append(ks_statistic(fam, xs, params_b))
        except Exception:  # noqa: BLE001 - a failed replicate is data, not a crash
            failed += 1
    null = np.asarray(sims, dtype=float)
    if null.size == 0:
        return BootstrapKS(d_obs, p_naive, float("nan"), 0, failed, null)
    p_boot = (1.0 + float(np.sum(null >= d_obs))) / (1.0 + null.size)
    return BootstrapKS(d_obs, p_naive, p_boot, null.size, failed, null)


# --------------------------------------------------------------------------------------
# Fit results
# --------------------------------------------------------------------------------------


@dataclass
class FitResult:
    family: Family
    n: int
    params: Optional[Tuple[float, ...]] = None
    loglik: float = float("nan")
    aic: float = float("nan")
    aicc: float = float("nan")
    bic: float = float("nan")
    ks: Optional[BootstrapKS] = None
    excluded_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    aic_weight: float = float("nan")
    delta_aic: float = float("nan")
    win_share: float = float("nan")

    @property
    def name(self) -> str:
        return self.family.name

    @property
    def ok(self) -> bool:
        return self.excluded_reason is None and self.params is not None

    def adequate(self, alpha: float = 0.05) -> Optional[bool]:
        """None when no bootstrap test was run; otherwise 'not rejected at alpha'."""
        if self.ks is None or not math.isfinite(self.ks.p_bootstrap):
            return None
        return self.ks.p_bootstrap >= alpha


def _information_criteria(loglik: float, k: int, n: int) -> Tuple[float, float, float]:
    aic = 2.0 * k - 2.0 * loglik
    denom = n - k - 1
    aicc = aic + (2.0 * k * (k + 1) / denom) if denom > 0 else float("inf")
    bic = k * math.log(n) - 2.0 * loglik
    return aic, aicc, bic


# --------------------------------------------------------------------------------------
# Mechanism 4 - the price of a free location parameter
# --------------------------------------------------------------------------------------


@dataclass
class LocationProbe:
    family_name: str
    loglik_pinned: float
    loglik_free: float
    loc_free: float
    data_min: float
    aic_pinned: float
    aic_free: float

    @property
    def loglik_gain(self) -> float:
        return self.loglik_free - self.loglik_pinned

    @property
    def free_fit_invalid(self) -> bool:
        """True when the estimated loc sits above min(x).

        That is not a near miss. Every observation below `loc` has zero density under the
        fitted model, so the log-likelihood is -inf: scipy has returned a "fit" that assigns
        impossible to data it was given. It happens because the 3-parameter MLE for Weibull
        and gamma is unbounded as loc approaches min(x) from below, so the optimiser walks
        toward a singularity and steps over it.
        """
        return self.loc_free > self.data_min

    @property
    def free_wins_aic(self) -> bool:
        return self.aic_free < self.aic_pinned


def probe_free_location(fam: Family, x: np.ndarray) -> Optional[LocationProbe]:
    """Refit a positive-support family with loc free and report what that bought.

    Three outcomes, all of them arguments against mixing pinned and unpinned fits in one
    table: the parameter buys nothing (true family), it buys a lot (wrong family, which is
    exactly backwards for a selection contest), or the optimiser returns a loc above min(x)
    and the fit is not a probability model for this data at all.
    """
    if not fam.fix_loc or fam.support != POSITIVE or fam.fix_scale:
        return None
    try:
        pinned = fit_params(fam, x)
        ll_pinned = loglikelihood(fam, x, pinned)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            free = tuple(float(p) for p in fam.dist.fit(x))
        ll_free = loglikelihood(fam, x, free)  # -inf when loc > min(x)
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(ll_pinned):
        return None
    loc = float(free[-2])
    aic_p, _, _ = _information_criteria(ll_pinned, fam.n_free, x.size)
    aic_f, _, _ = _information_criteria(ll_free, fam.n_free + 1, x.size)
    return LocationProbe(fam.name, ll_pinned, ll_free, loc, float(np.min(x)), aic_p, aic_f)


# --------------------------------------------------------------------------------------
# Mechanism 1 + selection stability
# --------------------------------------------------------------------------------------


def selection_stability(
    x: np.ndarray,
    families: Sequence[Family],
    n_boot: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Share of nonparametric bootstrap resamples in which each family wins on AIC.

    Akaike weights are a function of one dataset's delta-AIC. They answer "given these
    numbers, how much better is the winner" - not "would the winner win again". Resampling
    the rows and rerunning the whole comparison answers the second question, and the two
    answers routinely disagree by 40 percentage points when the candidates are close.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    arr = np.asarray(x, dtype=float)
    n = arr.size
    wins: Dict[str, int] = {f.name: 0 for f in families}
    counted = 0
    for _ in range(n_boot):
        xb = arr[rng.integers(0, n, size=n)]
        best_name, best_aic = None, float("inf")
        for fam in families:
            if support_violation(fam, xb) is not None:
                continue
            try:
                params = fit_params(fam, xb)
                ll = loglikelihood(fam, xb, params)
            except Exception:  # noqa: BLE001
                continue
            if not math.isfinite(ll):
                continue
            aic, _, _ = _information_criteria(ll, fam.n_free, n)
            if aic < best_aic:
                best_name, best_aic = fam.name, aic
        if best_name is not None:
            wins[best_name] += 1
            counted += 1
    if counted == 0:
        return {name: float("nan") for name in wins}
    return {name: c / counted for name, c in wins.items()}


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


@dataclass
class FitReport:
    diagnostics: DataDiagnostics
    results: List[FitResult]
    alpha: float
    stability_reps: int
    location_probes: List[LocationProbe] = field(default_factory=list)

    @property
    def ranked(self) -> List[FitResult]:
        usable = [r for r in self.results if r.ok and math.isfinite(r.aic)]
        return sorted(usable, key=lambda r: r.aic)

    @property
    def excluded(self) -> List[FitResult]:
        return [r for r in self.results if not r.ok]

    @property
    def best(self) -> Optional[FitResult]:
        ranked = self.ranked
        return ranked[0] if ranked else None

    @property
    def adequate(self) -> List[FitResult]:
        return [r for r in self.ranked if r.adequate(self.alpha) is True]

    def verdict(self) -> str:
        """One paragraph that says what the table is entitled to claim."""
        best = self.best
        if best is None:
            return "No candidate could be fitted. Check the data's support and finiteness."
        adequate = self.adequate
        lines: List[str] = []
        if not adequate:
            lines.append(
                f"NO ADEQUATE FIT. Every candidate is rejected by the bootstrap KS test at "
                f"alpha={self.alpha:g}. `{best.name}` is the AIC winner, which makes it the "
                f"least-bad of a set that does not contain the answer - a ranking, not a fit."
            )
        elif best.adequate(self.alpha) is True:
            lines.append(
                f"`{best.name}` wins on AIC and survives the bootstrap KS test "
                f"(p={best.ks.p_bootstrap:.3f}). That is the strongest claim this tool makes: "
                f"not rejected, on this sample size."
            )
        else:
            names = ", ".join(f"`{r.name}`" for r in adequate)
            lines.append(
                f"SPLIT VERDICT. `{best.name}` wins on AIC but is rejected by the bootstrap KS "
                f"test (p={best.ks.p_bootstrap:.3f}); {names} survive(s) the absolute test. "
                f"Prefer a candidate that is not rejected over one that merely ranks first."
            )
        if math.isfinite(best.win_share):
            if best.win_share < 0.6:
                lines.append(
                    f"The winner is unstable: it takes the top rank in only "
                    f"{best.win_share:.0%} of {self.stability_reps} bootstrap resamples. "
                    f"Akaike weight {best.aic_weight:.2f} overstates the case."
                )
            else:
                lines.append(
                    f"The winner is stable: top rank in {best.win_share:.0%} of "
                    f"{self.stability_reps} bootstrap resamples."
                )
        if self.diagnostics.heavily_tied:
            lines.append(
                "Ties are heavy enough to inflate every KS statistic here; treat the "
                "absolute tests as advisory and the ranking as the usable output."
            )
        if self.diagnostics.n > 5000:
            lines.append(
                f"At n={self.diagnostics.n} the KS test can resolve departures too small to "
                "matter for any downstream decision. Rejection is not the same as unusable."
            )
        return " ".join(lines)

    def table(self) -> str:
        header = (
            f"{'family':<12}{'k':>3}{'logLik':>12}{'AIC':>12}{'dAIC':>9}"
            f"{'weight':>8}{'KS D':>8}{'p naive':>9}{'p boot':>8}{'win%':>7}"
        )
        sep = "-" * len(header)
        rows = [header, sep]
        for r in self.ranked:
            p_naive = r.ks.p_naive if r.ks else float("nan")
            p_boot = r.ks.p_bootstrap if r.ks else float("nan")
            d = r.ks.d_observed if r.ks else float("nan")
            flag = ""
            if r.adequate(self.alpha) is False:
                flag = " *"
            rows.append(
                f"{r.name:<12}{r.family.n_free:>3}{r.loglik:>12.2f}{r.aic:>12.2f}"
                f"{r.delta_aic:>9.2f}{r.aic_weight:>8.3f}{d:>8.4f}"
                f"{p_naive:>9.3f}{p_boot:>8.3f}{r.win_share * 100:>6.0f}%{flag}"
            )
        rows.append(sep)
        rows.append("* rejected by the bootstrap KS test at alpha=%g" % self.alpha)
        for r in self.excluded:
            rows.append(f"excluded: {r.name:<12} {r.excluded_reason}")
        return "\n".join(rows)


def fit_distributions(
    x: Sequence[float],
    families: Optional[Sequence[Family]] = None,
    n_boot: int = 200,
    stability_reps: int = 100,
    alpha: float = 0.05,
    seed: int = 0,
    probe_location: bool = True,
) -> FitReport:
    """Fit every candidate, rank them, and test whether any of them is adequate.

    `n_boot` sets the resolution of the bootstrap p-values (the smallest reportable value is
    1/(n_boot+1)); `stability_reps` sets the resolution of the win shares. Both cost one
    full refit per replicate, which is where all the runtime goes.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    diagnostics = diagnose(arr)
    fams = list(DEFAULT_FAMILIES if families is None else families)
    rng = np.random.default_rng(seed)

    results: List[FitResult] = []
    for fam in fams:
        reason = support_violation(fam, arr)
        if reason is not None:
            results.append(FitResult(fam, arr.size, excluded_reason=reason))
            continue
        try:
            params = fit_params(fam, arr)
            ll = loglikelihood(fam, arr, params)
        except Exception as exc:  # noqa: BLE001
            results.append(FitResult(fam, arr.size, excluded_reason=f"fit failed: {exc}"))
            continue
        if not math.isfinite(ll):
            results.append(
                FitResult(fam, arr.size, excluded_reason="log-likelihood is -inf under the fit")
            )
            continue
        aic, aicc, bic = _information_criteria(ll, fam.n_free, arr.size)
        res = FitResult(fam, arr.size, params=params, loglik=ll, aic=aic, aicc=aicc, bic=bic)
        if n_boot > 0:
            res.ks = bootstrap_ks(fam, arr, params, n_boot=n_boot, rng=rng)
        results.append(res)

    usable = [r for r in results if r.ok and math.isfinite(r.aic)]
    if usable:
        best_aic = min(r.aic for r in usable)
        weights = {r.name: math.exp(-0.5 * (r.aic - best_aic)) for r in usable}
        total = sum(weights.values())
        for r in usable:
            r.delta_aic = r.aic - best_aic
            r.aic_weight = weights[r.name] / total if total > 0 else float("nan")

    if stability_reps > 0 and usable:
        shares = selection_stability(
            arr, [r.family for r in usable], n_boot=stability_reps, rng=rng
        )
        for r in usable:
            r.win_share = shares.get(r.name, float("nan"))

    probes: List[LocationProbe] = []
    if probe_location:
        for r in usable:
            probe = probe_free_location(r.family, arr)
            if probe is not None:
                probes.append(probe)

    return FitReport(
        diagnostics=diagnostics,
        results=results,
        alpha=alpha,
        stability_reps=stability_reps,
        location_probes=probes,
    )


# --------------------------------------------------------------------------------------
# QQ support
# --------------------------------------------------------------------------------------


def qq_points(
    fam: Family, x: Sequence[float], params: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Theoretical vs empirical quantiles using the (i - 0.5)/n plotting positions.

    The 0.5 offset matters at the tails: with i/n the largest point maps to the infinite
    quantile of any unbounded family, and every QQ plot ends in a spike that is an artefact
    of the plotting position rather than a property of the fit.
    """
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = fam.dist.ppf(probs, *params)
    return np.asarray(theoretical, dtype=float), xs


# --------------------------------------------------------------------------------------
# Sample data - the "Halcyon" telemetry book
# --------------------------------------------------------------------------------------


def sample_book(seed: int = 5) -> Dict[str, np.ndarray]:
    """Four columns from a fictional product's telemetry, each with a different lesson.

    - session_seconds : genuinely lognormal, clean. The case where the tool should say yes.
    - basket_value    : gamma, rounded to cents. Real data is discrete; watch the ties.
    - latency_ms      : a two-component mixture. No candidate is right, and AIC still
                        produces a confident winner.
    - daily_return    : Student-t with 4 df. Normal will not be rejected at small n and
                        will be badly wrong in the tail, which is the whole point.
    """
    rng = np.random.default_rng(seed)
    session_seconds = rng.lognormal(mean=4.2, sigma=0.85, size=1200)

    basket_value = np.round(rng.gamma(shape=2.4, scale=18.0, size=1200), 2)

    fast = rng.lognormal(mean=3.0, sigma=0.35, size=1050)
    slow = rng.lognormal(mean=5.4, sigma=0.55, size=150)
    latency_ms = np.round(np.concatenate([fast, slow]), 1)
    rng.shuffle(latency_ms)

    daily_return = rng.standard_t(df=4, size=900) * 0.011

    return {
        "session_seconds": session_seconds,
        "basket_value": basket_value,
        "latency_ms": latency_ms,
        "daily_return": daily_return,
    }
