"""Novelty and primacy decay in a long holdout, and the arithmetic behind the readout.

Three things live here and nothing else:

* generators for three worlds that all produce a lift DECLINING in user tenure - a real
  exponential decay, a permanent effect measured on a changing cohort mix, and a
  permanent effect measured on a surviving population - plus the enrollment process,
  which turns out to change the reported number on its own;
* the readouts an experiment actually publishes: the pooled lift, the lift-by-tenure
  curve, the early-window-vs-late-window guard, and a fitted exponential whose
  extrapolated asymptote is the quantity the launch decision needs;
* the two things that are arithmetic rather than tests - the user-day attenuation a test
  window applies to the novelty component, and the long-term holdout's MDE.

`cohort_profile` is the interesting one: it solves for the set of cohort effects, with NO
tenure decay in any of them, whose aggregate lift-by-tenure curve equals a given decay
curve EXACTLY. The two worlds are then observationally identical in the published plot
and 5x apart in the decision.

Nothing here reads results.json. Every number in the evidence run is computed live.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize, stats

# --------------------------------------------------------------------------- effect shapes


def tau_decay(t, tau0: float, tau_inf: float, lam: float):
    """Effect at tenure t. tau0 on the first day, tau_inf as t -> infinity."""
    return tau_inf + (tau0 - tau_inf) * np.exp(-np.asarray(t, dtype=float) / lam)


def cohort_profile(T: int, tau0: float, tau_inf: float, lam: float) -> np.ndarray:
    """Cohort effects m_0..m_{T-1}, CONSTANT in tenure, matching the decay tenure curve.

    Daily equal-sized cohorts. Tenure t is observed only for cohorts enrolled on or
    before day T-1-t, so the tenure-t lift is the mean of m over the prefix
    {0..T-1-t}. Pinning those prefix means to the decay curve gives
    C_j = tau_decay(T-1-j) and m_j = (j+1) C_j - j C_{j-1}.

    Note the consequence: mean(m) = C_{T-1} = tau_decay(0) = tau0. A world with no decay
    at all whose population average effect equals the OTHER world's first-day effect.
    """
    j = np.arange(T)
    C = tau_decay((T - 1 - j).astype(float), tau0, tau_inf, lam)
    m = np.empty(T)
    m[0] = C[0]
    m[1:] = (j[1:] + 1) * C[1:] - j[1:] * C[:-1]
    return m


def attenuation(T: int, lam: float, enroll: str = "uniform") -> float:
    """User-day weighted mean of exp(-t/lam) over a T-day window. No data in it.

    "bigbang": every user enrolled on day 0, so every tenure 0..T-1 is equally weighted.
    "uniform": daily equal-sized cohorts, so tenure t carries weight (T-t) - the
    population is younger and the novelty component is weighted UP.
    """
    t = np.arange(T)
    r = np.exp(-t / lam)
    wt = np.ones(T) if enroll == "bigbang" else (T - t).astype(float)
    return float((wt * r).sum() / wt.sum())


def attenuation_continuous(T: float, lam: float, enroll: str = "uniform") -> float:
    """The continuous-time version, for the n-free statement: bigbang is
    (lam/T)(1-e^{-T/lam}); uniform is (2 lam/T)(1 - (lam/T)(1-e^{-T/lam}))."""
    E = np.exp(-T / lam)
    if enroll == "bigbang":
        return float((lam / T) * (1 - E))
    return float((2 * lam / T) * (1 - (lam / T) * (1 - E)))


# --------------------------------------------------------------------------- worlds


def _skeleton(n_users: int, T: int, enroll: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if enroll == "bigbang":
        s = np.zeros(n_users, dtype=np.int64)
    elif enroll == "uniform":
        s = (np.arange(n_users) % T).astype(np.int64)
    else:
        raise ValueError(f"unknown enroll {enroll!r}")
    tenure = np.arange(T)[None, :] - s[:, None]
    obs = tenure >= 0
    return s, tenure, obs


def _finish(rng, y, w, s, tenure, obs, T, truth, day0) -> Dict[str, object]:
    y = np.where(obs, y, np.nan)
    return {
        "y": y,
        "w": w,
        "s": s,
        "tenure": np.where(obs, tenure, -1),
        "obs": obs,
        "T": int(T),
        "truth_long_run": float(truth),
        "truth_day0": float(day0),
    }


def panel_decay(
    rng,
    n_users: int = 4200,
    T: int = 28,
    enroll: str = "uniform",
    tau0: float = 0.10,
    tau_inf: float = 0.02,
    lam: float = 7.0,
    sigma: float = 1.0,
    p_treat: float = 0.5,
) -> Dict[str, object]:
    """A genuine novelty effect: the same user's effect decays with THEIR tenure."""
    s, tenure, obs = _skeleton(n_users, T, enroll)
    w = (rng.random(n_users) < p_treat).astype(np.int8)
    eff = tau_decay(np.clip(tenure, 0, None), tau0, tau_inf, lam)
    y = rng.normal(0.0, sigma, size=(n_users, T)) + w[:, None] * eff
    return _finish(rng, y, w, s, tenure, obs, T, tau_inf, tau0)


def panel_mixshift(
    rng,
    n_users: int = 4200,
    T: int = 28,
    tau0: float = 0.10,
    tau_inf: float = 0.02,
    lam: float = 7.0,
    sigma: float = 1.0,
    p_treat: float = 0.5,
) -> Dict[str, object]:
    """No decay anywhere. Each cohort has a PERMANENT effect; late cohorts have larger
    ones, and only early cohorts are old enough to appear at long tenures."""
    s, tenure, obs = _skeleton(n_users, T, "uniform")
    w = (rng.random(n_users) < p_treat).astype(np.int8)
    m = cohort_profile(T, tau0, tau_inf, lam)
    eff = np.repeat(m[s][:, None], T, axis=1)
    y = rng.normal(0.0, sigma, size=(n_users, T)) + w[:, None] * eff
    d = _finish(rng, y, w, s, tenure, obs, T, float(m.mean()), float(m.mean()))
    d["cohort_effects"] = m
    return d


def panel_survivor(
    rng,
    n_users: int = 4200,
    T: int = 28,
    tau: float = 0.05,
    sigma: float = 1.0,
    base_sd: float = 1.0,
    h0: float = 0.03,
    gamma: float = 0.8,
    p_treat: float = 0.5,
) -> Dict[str, object]:
    """A permanent effect plus differential churn. The treatment drives low-value users
    away, so the treated population improves with tenure through selection alone."""
    s, tenure, obs = _skeleton(n_users, T, "bigbang")
    w = (rng.random(n_users) < p_treat).astype(np.int8)
    base = rng.normal(0.0, base_sd, size=n_users)
    haz = np.where(w == 1, h0 * np.exp(-gamma * base), h0)
    churned = rng.random((n_users, T)) < haz[:, None]
    first = np.where(churned.any(axis=1), churned.argmax(axis=1), T)
    obs = obs & (np.arange(T)[None, :] < first[:, None])
    y = base[:, None] + w[:, None] * tau + rng.normal(0.0, sigma, size=(n_users, T))
    d = _finish(rng, y, w, s, tenure, obs, T, tau, tau)
    d["alive"] = obs
    return d


# --------------------------------------------------------------------------- readouts


def pooled_lift(d: Dict[str, object]) -> Tuple[float, float]:
    """The number on the dashboard: mean over all observed user-days, treated minus control."""
    y, w = d["y"], d["w"]
    t = y[w == 1]
    c = y[w == 0]
    nt = int(np.isfinite(t).sum())
    nc = int(np.isfinite(c).sum())
    se = np.sqrt(np.nanvar(t, ddof=1) / nt + np.nanvar(c, ddof=1) / nc)
    return float(np.nanmean(t) - np.nanmean(c)), float(se)


def window_lift(d: Dict[str, object], lo: int, hi: int) -> Tuple[float, float, int]:
    """Lift over the user-days whose tenure is in [lo, hi)."""
    ten, y, w = d["tenure"], d["y"], d["w"]
    cell = (ten >= lo) & (ten < hi)
    t = y[cell & (w[:, None] == 1)]
    c = y[cell & (w[:, None] == 0)]
    nt, nc = t.size, c.size
    if nt < 2 or nc < 2:
        return np.nan, np.nan, nt + nc
    se = np.sqrt(np.var(t, ddof=1) / nt + np.var(c, ddof=1) / nc)
    return float(t.mean() - c.mean()), float(se), nt + nc


def lift_by_tenure(d: Dict[str, object], min_arm: int = 20) -> Dict[str, np.ndarray]:
    """The published plot: one lift per tenure day, with its own SE."""
    ten, y, w = d["tenure"], d["y"], d["w"]
    tt, ll, ss, nn = [], [], [], []
    for t in range(int(d["T"])):
        cell = ten == t
        a = y[cell & (w[:, None] == 1)]
        b = y[cell & (w[:, None] == 0)]
        if a.size < min_arm or b.size < min_arm:
            continue
        tt.append(t)
        ll.append(a.mean() - b.mean())
        ss.append(np.sqrt(np.var(a, ddof=1) / a.size + np.var(b, ddof=1) / b.size))
        nn.append(a.size + b.size)
    return {
        "tenure": np.array(tt, dtype=float),
        "lift": np.array(ll),
        "se": np.array(ss),
        "n": np.array(nn),
    }


def guard_early_vs_late(
    d: Dict[str, object], early: Tuple[int, int] = (0, 7), late: Tuple[int, int] = (21, 28)
) -> Dict[str, float]:
    """The guard everyone runs: is week 1's lift different from week 4's?"""
    a, sa, _ = window_lift(d, *early)
    b, sb, _ = window_lift(d, *late)
    se = float(np.sqrt(sa**2 + sb**2))
    z = (a - b) / se
    return {"early": a, "late": b, "diff": a - b, "se": se, "z": float(z),
            "p": float(2 * stats.norm.sf(abs(z)))}


def fit_exponential(curve: Dict[str, np.ndarray], lam0: float = 7.0) -> Optional[Dict[str, float]]:
    """Weighted NLS for (tau0, tau_inf, lam). The launch decision wants tau_inf, which is
    the parameter the window never observes."""

    def f(t, a, b, lam):
        return b + (a - b) * np.exp(-t / lam)

    x, y, s = curve["tenure"], curve["lift"], curve["se"]
    if x.size < 4:
        return None
    try:
        p, cov = optimize.curve_fit(
            f, x, y, p0=[y[0], y[-1], lam0], sigma=s, absolute_sigma=True,
            maxfev=40000, bounds=([-1.0, -1.0, 0.5], [1.0, 1.0, 400.0]),
        )
    except Exception:
        return None
    resid = y - f(x, *p)
    tot = float(((y - y.mean()) ** 2).sum())
    se_inf = float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) and cov[1, 1] >= 0 else np.nan
    return {
        "tau0": float(p[0]),
        "tau_inf": float(p[1]),
        "lam": float(p[2]),
        "se_tau_inf": se_inf,
        "r2": float(1.0 - (resid**2).sum() / tot) if tot > 0 else np.nan,
        "chi2_per_df": float(((resid / s) ** 2).sum() / max(x.size - 3, 1)),
        "at_bound": bool(p[2] > 399.0 or p[2] < 0.51),
    }


def within_cohort_slope(d: Dict[str, object], min_arm: int = 10) -> Dict[str, float]:
    """Regress the per-(cohort, tenure) lift on tenure with cohort fixed effects.

    This is the separator, and it is a data requirement rather than a test: real tenure
    decay bends every cohort's own curve, a cohort mix shift bends none of them. It needs
    the enrollment date in the table, which is why a dashboard cannot run it.
    """
    ten, y, w, s = d["tenure"], d["y"], d["w"], d["s"]
    cohorts = np.unique(s)
    rows: List[Tuple[float, int, float, float]] = []
    for k in cohorts:
        who = s == k
        for t in range(int(d["T"])):
            cell = (ten[who] == t)
            a = y[who][cell & (w[who][:, None] == 1)]
            b = y[who][cell & (w[who][:, None] == 0)]
            if a.size < min_arm or b.size < min_arm:
                continue
            se = np.sqrt(np.var(a, ddof=1) / a.size + np.var(b, ddof=1) / b.size)
            rows.append((float(a.mean() - b.mean()), int(k), float(t), float(se)))
    if len(rows) < 10:
        return {"slope": np.nan, "se": np.nan, "z": np.nan, "cells": len(rows)}
    lift = np.array([r[0] for r in rows])
    kk = np.array([r[1] for r in rows])
    tt = np.array([r[2] for r in rows])
    sd = np.array([r[3] for r in rows])
    wgt = 1.0 / sd**2
    # cohort-demean tenure and lift with the same weights, then weighted least squares
    dt = np.empty_like(tt)
    dl = np.empty_like(lift)
    for k in np.unique(kk):
        m = kk == k
        wk = wgt[m]
        dt[m] = tt[m] - np.average(tt[m], weights=wk)
        dl[m] = lift[m] - np.average(lift[m], weights=wk)
    den = float((wgt * dt**2).sum())
    if den <= 0:
        return {"slope": np.nan, "se": np.nan, "z": np.nan, "cells": len(rows)}
    slope = float((wgt * dt * dl).sum() / den)
    se = float(np.sqrt(1.0 / den))
    return {"slope": slope, "se": se, "z": slope / se, "cells": len(rows)}


def share_weighted_identity(d: Dict[str, object]) -> Dict[str, float]:
    """Rebuild the pooled lift from the tenure curve, two ways.

    `gap` uses ARM-SPECIFIC tenure shares and is exact algebra on a fixed panel (machine
    precision), so a non-zero value means the readout dropped user-days somewhere.

    `naive_gap` uses one pooled share per tenure, the way a share-weighted rollup in a BI
    tool would. It closes only when the two arms have the same tenure composition, so its
    residual is a tenure-level randomisation check - a real diagnostic, and a much smaller
    quantity than either composition failure this build is about. Day 170's identity
    separated two worlds by 1,204 sd; this one does not generalise that far, and the point
    of computing both is to show where the family of free guards stops.
    """
    pooled, se = pooled_lift(d)
    curve = lift_by_tenure(d, min_arm=2)
    ten, y, w = d["tenure"], d["y"], d["w"]
    st, sc, lt, lc = [], [], [], []
    for t in curve["tenure"].astype(int):
        cell = ten == t
        aa = y[cell & (w[:, None] == 1)]
        bb = y[cell & (w[:, None] == 0)]
        st.append(aa.size)
        sc.append(bb.size)
        lt.append(aa.mean())
        lc.append(bb.mean())
    st = np.array(st, dtype=float)
    sc = np.array(sc, dtype=float)
    exact = float((st / st.sum() * np.array(lt)).sum() - (sc / sc.sum() * np.array(lc)).sum())
    share = curve["n"] / curve["n"].sum()
    naive = float((share * curve["lift"]).sum())
    return {"pooled": pooled, "exact": exact, "naive": naive,
            "gap": pooled - exact, "naive_gap": pooled - naive,
            "naive_gap_in_se": (pooled - naive) / se}


# --------------------------------------------------------------------------- the holdout


def mde(n: int, p_treat: float, sigma: float = 1.0, alpha: float = 0.05,
        power: float = 0.80) -> float:
    """Two-sided MDE for a difference in means at allocation p_treat."""
    z = stats.norm.isf(alpha / 2) + stats.norm.isf(1 - power)
    return float(z * sigma * np.sqrt(1.0 / (n * p_treat) + 1.0 / (n * (1 - p_treat))))


def mde_ratio(p_hold: float) -> float:
    """MDE at allocation p_hold divided by MDE at 50/50. No n and no sigma in it:
    1 / (2 sqrt(p (1-p)))."""
    return float(1.0 / (2.0 * np.sqrt(p_hold * (1.0 - p_hold))))


def holdout_attribution(ship_days, n_days: int, gap_se: float) -> Optional[Dict[str, object]]:
    """A single long-term holdout carrying K launches is a JOINT COST.

    The daily treated-minus-holdout gap is a sum of step functions. OLS recovers the total
    exactly and each launch's own share only as well as the ship dates allow, so the SE
    inflation is a property of the release calendar and can be read off before any data
    exists. Returns None when the calendar leaves the per-launch effects unidentified -
    which is a rank deficiency, not a large number, so it is checked rather than inferred
    from a failed inverse.

    No intercept: the pre-launch gap is zero by construction under the null, and a launch
    on day 0 would be collinear with a constant.
    """
    ship_days = np.asarray(ship_days)
    d = np.arange(n_days)
    X = np.column_stack([(d >= s).astype(float) for s in ship_days])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    xtx_inv = np.linalg.inv(X.T @ X)
    se = gap_se * np.sqrt(np.diag(xtx_inv))
    ones = np.ones(X.shape[1])
    se_total = gap_se * float(np.sqrt(ones @ xtx_inv @ ones))
    return {"se_per_launch": se, "se_total": se_total,
            "cond": float(np.linalg.cond(X.T @ X))}
