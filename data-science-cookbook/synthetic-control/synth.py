"""Synthetic control: fit a donor pool to one treated unit, and say what the fit cannot know.

The engine is deliberately small.  Everything interesting in this build is a property of
the ESTIMATOR - the simplex constraint, the placebo permutation, the convex hull - and not
of any particular dataset, so the code that produces the numbers has to be short enough to
read in one sitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import nnls

# ----------------------------------------------------------------- the solver


def solve_simplex_ls(X: np.ndarray, y: np.ndarray, penalty: float = 1e6) -> np.ndarray:
    """argmin_w ||Xw - y||^2  subject to  w >= 0, sum(w) = 1.

    Solved as a non-negative least squares problem with the sum constraint appended as a
    heavily weighted row.  nnls is an exact active-set method, so the only approximation
    here is the finite penalty: the returned weights sum to 1 to about 1/penalty.
    """
    scale = float(np.abs(X).max()) or 1.0
    m = penalty * scale
    x_aug = np.vstack([X, np.full((1, X.shape[1]), m)])
    y_aug = np.concatenate([y, [m]])
    w, _ = nnls(x_aug, y_aug)
    return w


def solve_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """argmin_w ||Xw - y||^2 with no constraint at all - the comparison in section 6."""
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return w


SOLVERS = {"simplex": solve_simplex_ls, "ols": solve_ols}


# ----------------------------------------------------------------- the fit


@dataclass
class SynthFit:
    weights: np.ndarray
    gaps: np.ndarray  # treated minus synthetic, every period
    t_pre: int

    @property
    def pre_gaps(self) -> np.ndarray:
        return self.gaps[: self.t_pre]

    @property
    def post_gaps(self) -> np.ndarray:
        return self.gaps[self.t_pre :]

    @property
    def att(self) -> float:
        """The estimate: mean gap over the post period."""
        return float(self.post_gaps.mean())

    @property
    def pre_mspe(self) -> float:
        return float((self.pre_gaps**2).mean())

    @property
    def post_mspe(self) -> float:
        return float((self.post_gaps**2).mean())

    @property
    def pre_rmspe(self) -> float:
        return float(np.sqrt(self.pre_mspe))

    @property
    def ratio(self) -> float:
        """Abadie's test statistic: post-period fit error relative to pre-period fit error."""
        return self.post_mspe / self.pre_mspe if self.pre_mspe > 0 else np.inf


def fit_synth(Y: np.ndarray, treated: int, t_pre: int, solver: str = "simplex") -> SynthFit:
    """Y is (units, periods).  Row `treated` is the treated unit; every other row is a donor."""
    donors = [i for i in range(Y.shape[0]) if i != treated]
    x_pre = Y[donors, :t_pre].T  # (t_pre, n_donors)
    y_pre = Y[treated, :t_pre]
    w = SOLVERS[solver](x_pre, y_pre)
    synthetic = w @ Y[donors, :]
    return SynthFit(weights=w, gaps=Y[treated] - synthetic, t_pre=t_pre)


# ----------------------------------------------------------------- inference


def placebo_pvalue(
    Y: np.ndarray, treated: int, t_pre: int, statistic: str = "ratio", solver: str = "simplex"
) -> Tuple[float, np.ndarray]:
    """Refit the estimator with every unit in turn playing the treated unit.

    The p-value is the treated unit's rank among all N units on the chosen statistic.  Its
    smallest attainable value is 1/N = 1/(donors + 1) whatever the data says - that floor is
    a property of the design, measured in section 2.
    """
    n_units = Y.shape[0]
    stats = np.empty(n_units)
    for i in range(n_units):
        f = fit_synth(Y, i, t_pre, solver=solver)
        stats[i] = f.ratio if statistic == "ratio" else abs(f.att)
    p = float((stats >= stats[treated]).sum() / n_units)
    return p, stats


def pvalue_floor(n_donors: int) -> float:
    """The smallest p-value the permutation can return: no data enters this."""
    return 1.0 / (n_donors + 1)


# ----------------------------------------------------------------- the data


def make_panel(
    rng: np.random.Generator,
    n_donors: int = 20,
    t_pre: int = 30,
    t_post: int = 12,
    n_factors: int = 3,
    noise: float = 1.0,
    mix_alpha: float = 0.3,
    effect: float = 0.0,
    in_hull: bool = True,
    hull_shift: float = 0.0,
    contaminated: Optional[Dict[int, float]] = None,
    anticipation: int = 0,
    anticipation_share: float = 1.0,
) -> np.ndarray:
    """An interactive fixed-effects panel: Y_it = lambda_i . F_t + eps_it, plus the effect.

    `in_hull=True` draws the treated unit's factor loadings as a convex combination of the
    donors', so some convex combination of donors reproduces its factor structure exactly and
    the estimator is consistent.  A small `mix_alpha` makes that combination lopsided - the
    treated unit is then an unusual member of its own pool, which is the case synthetic control
    exists for.  `hull_shift` adds a constant no convex combination can reach.
    """
    t_total = t_pre + t_post
    factors = np.cumsum(rng.normal(0, 1, size=(n_factors, t_total)), axis=1)
    factors += rng.normal(0, 1, size=(n_factors, t_total))
    donor_load = rng.uniform(0.2, 1.8, size=(n_donors, n_factors))

    if in_hull:
        mix = rng.dirichlet(np.ones(n_donors) * mix_alpha)
        treated_load = mix @ donor_load
    else:
        treated_load = donor_load.max(axis=0) + 0.0

    loadings = np.vstack([treated_load, donor_load])  # row 0 is treated
    Y = loadings @ factors + rng.normal(0, noise, size=(n_donors + 1, t_total))
    if hull_shift:
        Y[0] += hull_shift

    if anticipation:
        ramp = np.linspace(0, effect * anticipation_share, anticipation + 1)[1:]
        Y[0, t_pre - anticipation : t_pre] += ramp
    Y[0, t_pre:] += effect
    if contaminated:
        for donor_idx, gamma in contaminated.items():
            Y[donor_idx + 1, t_pre:] += gamma * effect
    return Y


# ----------------------------------------------------------------- comparisons


def naive_estimates(Y: np.ndarray, treated: int, t_pre: int) -> Dict[str, float]:
    """The three things people reach for when there is no control group."""
    donors = [i for i in range(Y.shape[0]) if i != treated]
    y_t, d = Y[treated], Y[donors]

    # 1. the single donor with the closest pre-period path
    errs = ((d[:, :t_pre] - y_t[:t_pre]) ** 2).mean(axis=1)
    best = d[int(errs.argmin())]
    # 2. the unweighted donor average
    avg = d.mean(axis=0)
    # 3. difference-in-differences against that average
    did = (y_t[t_pre:].mean() - y_t[:t_pre].mean()) - (avg[t_pre:].mean() - avg[:t_pre].mean())
    return {
        "best_donor": float((y_t[t_pre:] - best[t_pre:]).mean() - (y_t[:t_pre] - best[:t_pre]).mean()),
        "donor_mean": float((y_t[t_pre:] - avg[t_pre:]).mean()),
        "did": float(did),
    }


def attenuation_prediction(weights: np.ndarray, contaminated: Dict[int, float]) -> float:
    """Section 7's closed form: a contaminated donor pushes the estimate to tau*(1 - sum w_j g_j)."""
    return 1.0 - sum(weights[j] * g for j, g in contaminated.items())


def hull_excess(Y: np.ndarray, treated: int, t_pre: int) -> Dict[str, float]:
    """How far outside the donor range the treated unit sits, per period.

    The synthetic unit is a convex combination, so it is trapped between the lowest and the
    highest donor at every period.  Anything the treated unit does outside that band cannot be
    matched, which makes this an OBSERVABLE lower bound on the gap - it needs no counterfactual
    and no knowledge of the true effect.
    """
    donors = [i for i in range(Y.shape[0]) if i != treated]
    lo, hi = Y[donors].min(axis=0), Y[donors].max(axis=0)
    excess = np.maximum(Y[treated] - hi, 0.0) + np.minimum(Y[treated] - lo, 0.0)
    return {"pre": float(excess[:t_pre].mean()), "post": float(excess[t_pre:].mean())}


def in_donor_band(Y: np.ndarray, treated: int, fit: "SynthFit") -> bool:
    """The exact structural guarantee: no convex combination leaves the donor range."""
    donors = [i for i in range(Y.shape[0]) if i != treated]
    synthetic = Y[treated] - fit.gaps
    lo, hi = Y[donors].min(axis=0), Y[donors].max(axis=0)
    tol = 1e-9 * max(1.0, float(np.abs(Y).max()))
    return bool(np.all(synthetic >= lo - tol) and np.all(synthetic <= hi + tol))


def rmse(values: np.ndarray, truth: float) -> float:
    return float(np.sqrt(((np.asarray(values) - truth) ** 2).mean()))
