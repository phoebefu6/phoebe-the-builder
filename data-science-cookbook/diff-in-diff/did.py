"""Difference-in-differences: the estimator, its assumption, and the tests that
are supposed to protect it.

Everything here is pure numpy/scipy on a panel held as two ``(N, T)`` arrays -
``Y`` (outcome) and ``D`` (treated indicator).  Balanced panels only, which is
what makes the two-way within transform an exact annihilator of the unit and
time fixed effects, and therefore makes every closed form below checkable to
machine precision rather than approximately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# panel algebra
# --------------------------------------------------------------------------


def two_way_demean(X: np.ndarray) -> np.ndarray:
    """Residual of X on unit dummies + time dummies, for a balanced panel.

    For a balanced panel the projection onto span{unit dummies, time dummies}
    is exactly ``xbar_i + xbar_t - xbar``, so this is the exact FWL annihilator
    and not an approximation.  Every estimator below leans on that.
    """
    return X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True) + X.mean()


def did_2x2(Y: np.ndarray, treated: np.ndarray, pre: Sequence[int], post: Sequence[int]) -> float:
    """The four-means estimator.  No regression, no fixed effects, no options."""
    t = np.asarray(treated, dtype=bool)
    pre = list(pre)
    post = list(post)
    return float(
        (Y[t][:, post].mean() - Y[t][:, pre].mean())
        - (Y[~t][:, post].mean() - Y[~t][:, pre].mean())
    )


def twfe(Y: np.ndarray, D: np.ndarray) -> float:
    """Two-way fixed effects coefficient on the treatment dummy."""
    Dt = two_way_demean(D)
    return float((Dt * Y).sum() / (Dt * Dt).sum())


def twfe_cell_weights(D: np.ndarray) -> np.ndarray:
    """Weight TWFE places on each *treated cell's own* treatment effect.

    Under ``y_it = a_i + g_t + tau_it D_it + e_it`` and FWL,

        E[beta_twfe] = sum_{it: D=1} w_it tau_it,   w_it = Dtilde_it / sum Dtilde^2

    and ``sum w_it == 1`` exactly, because ``sum_{D=1} Dtilde == sum Dtilde^2``.
    Nothing forces an individual ``w_it`` to be positive.  That is the whole
    staggered-adoption problem, in one line of algebra.
    """
    Dt = two_way_demean(D)
    denom = (Dt * Dt).sum()
    return np.where(D > 0, Dt / denom, 0.0)


def heterogeneity_bound(W: np.ndarray, D: np.ndarray, att: float) -> float:
    """Smallest sd of treatment effects that makes E[beta_twfe] zero.

    Minimise Var(tau) over treated cells subject to ``sum w tau = 0`` and
    ``mean(tau) = att``.  The Lagrangian gives ``tau = a + c*w``, hence

        sd_min = |att| * sd(w) / |mean(w) - sum(w^2)|

    (de Chaisemartin & D'Haultfoeuille's robustness measure, rederived).
    """
    w = W[D > 0]
    wbar = float(w.mean())
    sw2 = float((w * w).sum())
    denom = wbar - sw2
    if abs(denom) < 1e-300:
        return float("inf")
    return float(abs(att) * w.std(ddof=0) / abs(denom))


# --------------------------------------------------------------------------
# fixed-effects OLS with a choice of variance estimator
# --------------------------------------------------------------------------


@dataclass
class FitResult:
    beta: np.ndarray
    vcov: np.ndarray
    dof: int
    labels: List[str]

    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.vcov), 0.0, None))

    def tstat(self) -> np.ndarray:
        se = self.se()
        return np.where(se > 0, self.beta / np.where(se > 0, se, 1.0), 0.0)

    def pvalue(self) -> np.ndarray:
        return 2.0 * stats.t.sf(np.abs(self.tstat()), self.dof)

    def wald(self, idx: Sequence[int]) -> Tuple[float, float]:
        """Joint chi-square test that the named coefficients are all zero."""
        idx = list(idx)
        b = self.beta[idx]
        V = self.vcov[np.ix_(idx, idx)]
        stat = float(b @ np.linalg.solve(V, b))
        return stat, float(stats.chi2.sf(stat, len(idx)))


def fe_ols(
    Y: np.ndarray,
    regressors: Sequence[np.ndarray],
    labels: Optional[Sequence[str]] = None,
    vcov: str = "cluster",
    cluster_id: Optional[np.ndarray] = None,
) -> FitResult:
    """OLS with unit and time fixed effects absorbed by the within transform.

    ``vcov='iid'`` is the textbook homoskedastic-independent formula - the one
    a regression command prints by default.  ``vcov='cluster'`` clusters, with
    the usual finite-sample correction and a t(G-1) reference.  ``cluster_id``
    is one label per *unit*; the default clusters on the unit itself, which is
    the finest level available and - as section 4 measures - not always the
    right one.
    """
    N, T = Y.shape
    K = len(regressors)
    yt = two_way_demean(Y).reshape(-1)
    Xt = np.column_stack([two_way_demean(np.asarray(x, float)).reshape(-1) for x in regressors])
    XtX = Xt.T @ Xt
    XtXinv = np.linalg.inv(XtX)
    b = XtXinv @ (Xt.T @ yt)
    resid = yt - Xt @ b
    n = N * T
    k_total = K + N + T - 1  # absorbed FE cost degrees of freedom too

    if vcov == "cluster":
        per_unit = np.einsum("ntk,nt->nk", Xt.reshape(N, T, K), resid.reshape(N, T))
        if cluster_id is None:
            scores = per_unit
        else:
            cid = np.asarray(cluster_id)
            uniq = np.unique(cid)
            scores = np.stack([per_unit[cid == c].sum(axis=0) for c in uniq])
        G = scores.shape[0]
        meat = scores.T @ scores
        corr = (G / (G - 1)) * ((n - 1) / (n - k_total))
        V = XtXinv @ meat @ XtXinv * corr
        dof = G - 1
    elif vcov == "iid":
        s2 = float(resid @ resid) / (n - k_total)
        V = XtXinv * s2
        dof = n - k_total
    else:
        raise ValueError(f"unknown vcov {vcov!r}")

    return FitResult(beta=b, vcov=V, dof=dof, labels=list(labels or [f"x{i}" for i in range(K)]))


# --------------------------------------------------------------------------
# event study
# --------------------------------------------------------------------------


def event_dummies(adopt: np.ndarray, T: int, event_times: Sequence[int]) -> List[np.ndarray]:
    """One (N, T) indicator per event time, relative to each unit's adoption.

    Never-treated units (``adopt = inf``) get zeros everywhere, which is what
    makes them the comparison group rather than a coefficient.
    """
    rel = np.arange(T)[None, :] - adopt[:, None]  # -inf for never-treated, so never equal to e
    return [(rel == float(e)).astype(float) for e in event_times]


def event_study(
    Y: np.ndarray,
    adopt: np.ndarray,
    event_times: Sequence[int],
    vcov: str = "cluster",
) -> FitResult:
    """Event-study coefficients relative to the omitted base period e = -1.

    ``event_times`` MUST cover every period each treated unit is observed in.
    A period with no dummy falls into the omitted base category, which
    contaminates the reference point and biases every coefficient - by enough
    to flip signs, silently (see ``test_a_short_event_window...``).  There is
    no warning for this in any implementation, including this one; the caller
    owns the window.
    """
    ev = [int(e) for e in event_times if int(e) != -1]
    cols = event_dummies(adopt, Y.shape[1], ev)
    return fe_ols(Y, cols, labels=[f"e{e:+d}" for e in ev], vcov=vcov)


def pretrend_test(fit: FitResult) -> Tuple[float, float, int]:
    """Joint test that every pre-period (lead) coefficient is zero."""
    idx = [i for i, lab in enumerate(fit.labels) if lab.startswith("e-")]
    if not idx:
        return 0.0, 1.0, 0
    stat, p = fit.wald(idx)
    return stat, p, len(idx)


# --------------------------------------------------------------------------
# not-yet-treated group-time ATT (Callaway & Sant'Anna in its simplest form)
# --------------------------------------------------------------------------


def group_time_att(Y: np.ndarray, adopt: np.ndarray) -> Dict[Tuple[int, int], float]:
    """ATT(g, t): each cohort against units not yet treated at t, base g-1.

    Never uses an already-treated unit as a control.  That single restriction
    is the difference between this and TWFE.
    """
    N, T = Y.shape
    out: Dict[Tuple[int, int], float] = {}
    groups = sorted({int(g) for g in adopt if np.isfinite(g)})
    for g in groups:
        base = g - 1
        if base < 0:
            continue
        gi = adopt == g
        for t in range(g, T):
            ctrl = adopt > t  # inf > t is True, so never-treated are included
            if ctrl.sum() == 0 or gi.sum() == 0:
                continue
            out[(g, t)] = float(
                (Y[gi, t].mean() - Y[gi, base].mean()) - (Y[ctrl, t].mean() - Y[ctrl, base].mean())
            )
    return out


def aggregate_att(atts: Dict[Tuple[int, int], float], adopt: np.ndarray, T: int) -> float:
    """Average ATT(g, t) over treated cells, weighted by cohort size.

    Weights match the estimand TWFE is usually *claimed* to target: the mean
    treatment effect over all treated unit-periods.
    """
    num = 0.0
    den = 0.0
    for (g, t), a in atts.items():
        n_g = float((adopt == g).sum())
        num += n_g * a
        den += n_g
    return float(num / den) if den else float("nan")


# --------------------------------------------------------------------------
# data-generating process
# --------------------------------------------------------------------------


def ar1_errors(rng: np.random.Generator, N: int, T: int, rho: float, sigma: float = 1.0) -> np.ndarray:
    """Stationary AR(1) idiosyncratic errors, unit by unit."""
    e = np.empty((N, T))
    sd0 = sigma / np.sqrt(1.0 - rho ** 2) if abs(rho) < 1 else sigma
    e[:, 0] = rng.normal(0.0, sd0, N)
    u = rng.normal(0.0, sigma, (N, T))
    for t in range(1, T):
        e[:, t] = rho * e[:, t - 1] + u[:, t]
    return e


def make_panel(
    rng: np.random.Generator,
    n_treated: int = 100,
    n_control: int = 100,
    T: int = 12,
    t0: int = 6,
    effect: float = 1.0,
    diff_trend: float = 0.0,
    rho: float = 0.0,
    sigma: float = 1.0,
    unit_sd: float = 1.0,
    time_shock_sd: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Common-timing panel.  Returns ``(Y, D, adopt, tau)``.

    ``diff_trend`` is the per-period slope the treated group has *in addition*
    to the common time effects.  It is the parallel-trends violation, in the
    units the estimator reports, and setting it to zero is the only thing that
    makes DiD unbiased here.
    """
    N = n_treated + n_control
    treated = np.zeros(N, dtype=bool)
    treated[:n_treated] = True
    t = np.arange(T)

    alpha = rng.normal(0.0, unit_sd, (N, 1))
    gamma = rng.normal(0.0, time_shock_sd, (1, T)) + 0.30 * t[None, :]
    trend = diff_trend * treated[:, None] * t[None, :]

    D = (treated[:, None] & (t[None, :] >= t0)).astype(float)
    tau = effect * D
    e = ar1_errors(rng, N, T, rho, sigma)

    Y = alpha + gamma + trend + tau + e
    adopt = np.where(treated, float(t0), np.inf)
    return Y, D, adopt, tau


def make_staggered(
    rng: np.random.Generator,
    cohorts: Sequence[Tuple[int, int]],
    T: int = 20,
    n_never: int = 0,
    growth: float = 0.0,
    level: float = 1.0,
    sigma: float = 1.0,
    unit_sd: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Staggered adoption with *dynamic* effects: tau grows with exposure.

    ``cohorts`` is a list of ``(adoption_period, n_units)``.  The effect on a
    unit treated for ``k`` periods is ``level + growth * k`` - strictly
    positive and strictly increasing for any ``growth >= 0``.  There is no
    heterogeneity here that a practitioner would call unreasonable.
    """
    adopt_list: List[float] = []
    for g, n in cohorts:
        adopt_list += [float(g)] * n
    adopt_list += [np.inf] * n_never
    adopt = np.array(adopt_list)
    N = adopt.size
    t = np.arange(T)

    D = np.zeros((N, T))
    tau = np.zeros((N, T))
    for i in range(N):
        g = adopt[i]
        if np.isfinite(g):
            for j in range(int(g), T):
                D[i, j] = 1.0
                tau[i, j] = level + growth * (j - g)

    alpha = rng.normal(0.0, unit_sd, (N, 1))
    gamma = 0.20 * t[None, :]
    Y = alpha + gamma + tau + rng.normal(0.0, sigma, (N, T))
    return Y, D, adopt, tau
