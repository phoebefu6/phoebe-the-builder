"""Interference (SUTVA violation) engine.

A randomised A/B test estimates the difference between a treated unit and a control
unit *in a world where half of everybody is treated*.  The decision it is used for is
the difference between everybody treated and everybody control.  Those two quantities
are equal only if one unit's assignment cannot touch another unit's outcome - the
"no interference" half of SUTVA.  This module builds two worlds where it is false,
in opposite directions, and the three designs usually proposed as the fix.

Nothing here is estimated from data the caller does not have: every ground truth is a
second simulation of the SAME world under global treatment and global control, which
is exactly the quantity an experiment can never observe.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# 1.  A rationed marketplace: interference through a shared finite supply.
# --------------------------------------------------------------------------- #


def rationed_outcomes(
    z: np.ndarray,
    p_control: float,
    p_treat: float,
    supply: int,
    rng: np.random.Generator,
    group: Optional[np.ndarray] = None,
    supply_per_group: Optional[int] = None,
) -> np.ndarray:
    """One period of a marketplace with a finite number of things to sell.

    Each user attempts with probability ``p_treat`` if assigned, ``p_control`` if not.
    Attempts are served while supply lasts; when attempts exceed supply, the served
    set is a uniform random subset (proportional rationing, no queue priority).

    ``group`` + ``supply_per_group`` makes supply LOCAL to each group, which is what
    makes cluster randomisation able to contain the interference.  Omit them and the
    pool is global, which is what makes it unable to.
    """
    n = z.size
    p = np.where(z == 1, p_treat, p_control)
    attempt = rng.random(n) < p
    y = np.zeros(n)

    if group is None or supply_per_group is None:
        idx = np.flatnonzero(attempt)
        if idx.size == 0:
            return y
        if idx.size <= supply:
            y[idx] = 1.0
        else:
            y[rng.choice(idx, size=supply, replace=False)] = 1.0
        return y

    for g in np.unique(group):
        idx = np.flatnonzero(attempt & (group == g))
        if idx.size == 0:
            continue
        if idx.size <= supply_per_group:
            y[idx] = 1.0
        else:
            y[rng.choice(idx, size=supply_per_group, replace=False)] = 1.0
    return y


def market_split_estimate(
    n: int,
    p_control: float,
    p_treat: float,
    supply: int,
    rng: np.random.Generator,
    share_treated: float = 0.5,
) -> Tuple[float, float]:
    """A conventional user-level A/B test inside ONE shared market.

    Returns (estimate, naive standard error).  The SE is the one the test reports -
    a two-proportion SE that knows nothing about the shared pool.
    """
    n_t = int(round(n * share_treated))
    z = np.zeros(n, dtype=int)
    z[rng.choice(n, size=n_t, replace=False)] = 1
    y = rationed_outcomes(z, p_control, p_treat, supply, rng)
    yt, yc = y[z == 1], y[z == 0]
    est = yt.mean() - yc.mean()
    se = float(np.sqrt(yt.var(ddof=1) / yt.size + yc.var(ddof=1) / yc.size))
    return float(est), se


def market_global_effect(
    n: int,
    p_control: float,
    p_treat: float,
    supply: int,
    rng: np.random.Generator,
    reps: int = 400,
) -> float:
    """The quantity the decision needs: everybody treated minus everybody control.

    Unobservable in a real experiment.  Here it is a second run of the same world.
    """
    ones, zeros = np.ones(n, dtype=int), np.zeros(n, dtype=int)
    t = np.mean([rationed_outcomes(ones, p_control, p_treat, supply, rng).mean() for _ in range(reps)])
    c = np.mean([rationed_outcomes(zeros, p_control, p_treat, supply, rng).mean() for _ in range(reps)])
    return float(t - c)


def tightness(n: int, p_control: float, supply: int) -> float:
    """Supply per expected control-side attempt.  >1 slack, <1 rationed."""
    return supply / (n * p_control)


# --------------------------------------------------------------------------- #
# 2.  A peer network: interference through positive spillover.
# --------------------------------------------------------------------------- #


def spillover_outcomes(
    z: np.ndarray,
    group: np.ndarray,
    tau: float,
    gamma: float,
    sigma: float,
    rng: np.random.Generator,
    group_sd: float = 0.0,
) -> np.ndarray:
    """Linear-in-means peer effects: my outcome moves with the share of my peers treated.

    ``tau`` is the direct effect, ``gamma`` the full indirect effect (the move a user
    gets when ALL of their peers are treated).  The global effect is ``tau + gamma``.

    ``group_sd`` adds a per-group intercept.  It cancels exactly in a within-group
    split (both arms sit in the same group) and does not cancel at all in a cluster
    randomisation, which is where the design effect comes from.
    """
    y = np.zeros(z.size) + rng.normal(0.0, sigma, z.size)
    if group_sd > 0:
        gs = np.unique(group)
        eff = dict(zip(gs.tolist(), rng.normal(0.0, group_sd, gs.size).tolist()))
        y += np.array([eff[g] for g in group.tolist()])
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        m = idx.size
        if m < 2:
            continue
        s = z[idx].sum()
        peer_frac = (s - z[idx]) / (m - 1)
        y[idx] += tau * z[idx] + gamma * peer_frac
    return y


def spillover_split_bias_closed_form(gamma: float, m: int) -> float:
    """Exact bias of a within-group 50/50 split under linear-in-means peer effects.

    A treated user has (m/2 - 1) of their (m - 1) peers treated; a control user has
    (m/2).  The split therefore recovers ``tau - gamma/(m-1)`` while the truth is
    ``tau + gamma``, so the bias is ``-gamma * m / (m - 1)`` - slightly MORE than the
    whole indirect effect, and with no sample size in it.
    """
    return -gamma * m / (m - 1)


# --------------------------------------------------------------------------- #
# 3.  The three designs.
# --------------------------------------------------------------------------- #


def assign_within_group(group: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exactly half of each group treated - the standard user-level split."""
    z = np.zeros(group.size, dtype=int)
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        z[rng.choice(idx, size=idx.size // 2, replace=False)] = 1
    return z


def assign_by_group(group: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Whole groups treated or control - cluster randomisation."""
    gs = np.unique(group)
    treated = set(rng.choice(gs, size=gs.size // 2, replace=False).tolist())
    return np.array([1 if g in treated else 0 for g in group], dtype=int)


def cluster_estimate(y: np.ndarray, z: np.ndarray, group: np.ndarray) -> Tuple[float, float]:
    """Difference in CLUSTER means with a cluster-level SE.

    The SE is computed across group means, not across users: with cluster assignment
    the group is the unit that was randomised, and n is the number of groups.
    """
    gs = np.unique(group)
    means = np.array([y[group == g].mean() for g in gs])
    zs = np.array([z[group == g][0] for g in gs])
    a, b = means[zs == 1], means[zs == 0]
    est = a.mean() - b.mean()
    se = float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))
    return float(est), se


def user_estimate(y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """Difference in USER means with a user-level (iid) SE."""
    a, b = y[z == 1], y[z == 0]
    est = a.mean() - b.mean()
    se = float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))
    return float(est), se


# --------------------------------------------------------------------------- #
# 4.  Switchback: randomise TIME instead of users.
# --------------------------------------------------------------------------- #


def switchback_run(
    n_periods: int,
    tau: float,
    carryover: float,
    sigma: float,
    rng: np.random.Generator,
    alternating: bool = False,
    burn_in: float = 0.0,
) -> Tuple[float, float]:
    """A switchback test where a fraction of each period still behaves like the last one.

    ``carryover`` c is the share of a period during which the system has not yet
    settled into its new condition.  ``burn_in`` is the share of each period the
    analyst DISCARDS before measuring; discarding at least c removes the bias and
    throws away that share of the data.
    """
    if alternating:
        z = np.arange(n_periods) % 2
    else:
        z = (rng.random(n_periods) < 0.5).astype(int)
    z_prev = np.concatenate(([z[0]], z[:-1]))

    keep = max(0.0, 1.0 - burn_in)
    if keep <= 0:
        raise ValueError("burn_in must be < 1")
    # Within the kept window, the share still under the previous condition.
    residual = max(0.0, carryover - burn_in) / keep
    w = (1.0 - residual) * z + residual * z_prev
    y = tau * w + rng.normal(0.0, sigma / np.sqrt(keep), n_periods)

    a, b = y[z == 1], y[z == 0]
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan")
    est = a.mean() - b.mean()
    se = float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))
    return float(est), se


def switchback_bias_closed_form(tau: float, carryover: float, alternating: bool, burn_in: float = 0.0) -> float:
    """Expected switchback estimate.

    Coin-flip randomisation: ``tau * (1 - r)``.  Strict ABAB alternation: ``tau * (1 - 2r)``,
    because a treated period's predecessor is ALWAYS a control period, so the carryover
    pushes the two arms apart in both directions instead of one.  ``r`` is the residual
    contamination left after burn-in.
    """
    keep = 1.0 - burn_in
    r = max(0.0, carryover - burn_in) / keep
    return tau * (1.0 - 2.0 * r) if alternating else tau * (1.0 - r)


# --------------------------------------------------------------------------- #
# 5.  The check people actually run: does the effect depend on the share treated?
# --------------------------------------------------------------------------- #


def dose_response_check(
    n: int,
    p_control: float,
    p_treat: float,
    supply: int,
    rng: np.random.Generator,
    shares: Tuple[float, float] = (0.1, 0.5),
) -> Dict[str, float]:
    """Run the SAME market at two treated shares and test whether the effect moved.

    Under no interference the effect is invariant to the share treated, so a
    significant difference is evidence of interference.  Returns the two estimates,
    the difference, its SE and a two-sided z p-value.
    """
    e1, s1 = market_split_estimate(n, p_control, p_treat, supply, rng, share_treated=shares[0])
    e2, s2 = market_split_estimate(n, p_control, p_treat, supply, rng, share_treated=shares[1])
    diff = e1 - e2
    se = float(np.sqrt(s1**2 + s2**2))
    from scipy import stats

    p = float(2 * (1 - stats.norm.cdf(abs(diff / se)))) if se > 0 else 1.0
    return {"est_low": e1, "est_high": e2, "diff": diff, "se": se, "p": p}


def power_of(flags: List[bool]) -> float:
    return float(np.mean(flags))
