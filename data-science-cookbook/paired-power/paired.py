"""Paired data analysed as two independent samples: the power it throws away, exactly.

Model: n pairs (before, after), bivariate normal, each SD 1, correlation rho, true mean shift delta
(in SD units). Two analyses of the SAME data:

  paired t      t = mean(D) / (sd(D)/sqrt n),                 df n-1,  D = after - before
  independent   t = mean(D) / sqrt((s_b^2 + s_a^2)/n),        df 2n-2  (Student = Welch at equal n)

Paired power is a noncentral t. The independent analysis is not: its denominator mixes two
correlated variances. Under normality mean(D) is independent of the sample covariance matrix, and
(n-1)(s_b^2 + s_a^2) = (1+rho)A + (1-rho)B with A, B independent chi-square(n-1) - the eigenvalues
of the covariance matrix. So given (A, B) the rejection probability is two normal tail areas, and a
2-D Gauss-Legendre quadrature over the chi-square quantiles of A and B gives it exactly. At rho = 0
it must reduce to Student's noncentral t, which is how it is calibrated.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import optimize, stats

ALPHA = 0.05
DELTA = 0.5  # the effect in SD units of one measurement: a "medium" before/after shift
RHOS = (-0.5, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95)
NS = (5, 10, 20, 40, 80)
GL_NODES = 160
MC_REPS = 20000
Z99 = 2.5758
_U, _W = np.polynomial.legendre.leggauss(GL_NODES)
_U, _W = (_U + 1) / 2, _W / 2  # nodes and weights on (0, 1)


def wilson(k: int, n: int, z: float = Z99) -> tuple:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def sd_diff(rho: float) -> float:
    """SD of after - before when each has SD 1."""
    return float(np.sqrt(2 * (1 - rho)))


def power_paired(n: int, rho: float, delta: float = DELTA, alpha: float = ALPHA) -> float:
    """Exact two-sided power of the paired t-test: noncentral t with df n-1."""
    df = n - 1
    nc = delta * np.sqrt(n) / sd_diff(rho)
    c = stats.t.ppf(1 - alpha / 2, df)
    # lower tail via symmetry: scipy returns nan for nct.cdf(-c) far out in the tail (nc ~ 10)
    return float(stats.nct.sf(c, df, nc) + stats.nct.sf(c, df, -nc))


def power_independent(n: int, rho: float, delta: float = DELTA, alpha: float = ALPHA) -> float:
    """Exact two-sided rejection rate of Student's two-sample t run on PAIRED data (quadrature)."""
    df = 2 * n - 2
    c = stats.t.ppf(1 - alpha / 2, df)
    a = stats.chi2.ppf(_U, n - 1)[:, None]
    b = stats.chi2.ppf(_U, n - 1)[None, :]
    s2sum = ((1 + rho) * a + (1 - rho) * b) / (n - 1)  # s_b^2 + s_a^2
    k = c * np.sqrt(s2sum / n)  # critical |mean(D)|
    s = sd_diff(rho) / np.sqrt(n)  # true SD of mean(D)
    p = stats.norm.sf((k - delta) / s) + stats.norm.cdf((-k - delta) / s)
    return float(_W @ p @ _W)


def n_for_power(target: float, rho: float, analysis: str, delta: float = DELTA) -> int:
    """Smallest n (pairs) giving at least `target` power. Power rises in n, so bisection is valid."""
    f = power_paired if analysis == "paired" else power_independent
    lo, hi = 2, 4
    while f(hi, rho, delta) < target:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if f(mid, rho, delta) >= target else (mid, hi)
    return hi


def break_even_rho(n: int, delta: float = DELTA) -> float:
    """The rho at which pairing stops costing power: below it the paired test's lost df win."""
    g = lambda r: power_paired(n, r, delta) - power_independent(n, r, delta)  # noqa: E731
    return float(optimize.brentq(g, -0.3, 0.6, xtol=1e-6))


def analyse(before: Sequence[float], after: Sequence[float]) -> Dict:
    """Both analyses on one before/after dataset, plus what the correlation bought."""
    b, a = np.asarray(before, float), np.asarray(after, float)
    if len(b) != len(a):
        raise ValueError(f"before has {len(b)} values and after has {len(a)}: pairs must match")
    if len(b) < 3:
        raise ValueError("need at least 3 pairs")
    d = a - b
    if np.std(d) == 0 or np.std(b) == 0 or np.std(a) == 0:
        raise ValueError("a column (or every difference) is constant, so no test is defined")
    r = float(np.corrcoef(b, a)[0, 1])
    p_pair = float(stats.ttest_rel(a, b).pvalue)
    p_ind = float(stats.ttest_ind(a, b).pvalue)
    return {"n": len(b), "delta": float(d.mean()), "rho_hat": r, "p_paired": p_pair,
            "p_independent": p_ind, "variance_ratio": 1 / (1 - r) if r < 1 else float("inf")}


def grid() -> List[Dict]:
    rows = []
    for n in NS:
        for rho in RHOS:
            rows.append({"n": n, "rho": rho,
                         "size_paired": ALPHA, "size_independent": power_independent(n, rho, 0.0),
                         "power_paired": power_paired(n, rho), "power_independent": power_independent(n, rho)})
    return rows


def calibrate() -> Dict:
    """Three independent checks of the quadrature and of the vectorised statistics."""
    gaps = []
    for n in (3, 5, 10, 40, 200):
        for d in (0.0, 0.3, 0.8):
            df, c = 2 * n - 2, stats.t.ppf(1 - ALPHA / 2, 2 * n - 2)
            nc = d * np.sqrt(n) / np.sqrt(2)
            exact = stats.nct.sf(c, df, nc) + stats.nct.sf(c, df, -nc)
            gaps.append(abs(power_independent(n, 0.0, d) - exact))
    rng = np.random.default_rng(7)
    mism = 0
    for _ in range(400):
        n, rho = int(rng.integers(3, 60)), float(rng.uniform(-0.9, 0.95))
        x = _pairs(rng, n, rho, 0.4, 1)[0]
        b, a = x[:, 0], x[:, 1]
        mism += (_t_paired(b[None], a[None])[0] > 0) != (stats.ttest_rel(a, b).pvalue < ALPHA)
        mism += (_t_indep(b[None], a[None])[0] > 0) != (stats.ttest_ind(a, b).pvalue < ALPHA)
    nodes = abs(power_independent(20, 0.8) - _power_independent_nodes(20, 0.8, 320))
    return {"max_gap_vs_student_nct_at_rho0": float(max(gaps)), "scipy_decision_mismatches": int(mism),
            "decisions_compared": 800, "gap_160_vs_320_nodes": float(nodes)}


def _power_independent_nodes(n: int, rho: float, m: int, delta: float = DELTA) -> float:
    global _U, _W
    keep = _U, _W
    u, w = np.polynomial.legendre.leggauss(m)
    _U, _W = (u + 1) / 2, w / 2
    try:
        return power_independent(n, rho, delta)
    finally:
        _U, _W = keep


def _pairs(rng: np.random.Generator, n: int, rho: float, delta: float, reps: int) -> np.ndarray:
    z1 = rng.standard_normal((reps, n))
    z2 = rho * z1 + np.sqrt(1 - rho * rho) * rng.standard_normal((reps, n))
    return np.stack([z1, z2 + delta], axis=-1)


def _t_paired(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Signed margin past the critical value (positive = reject), one row per replicate."""
    n = b.shape[1]
    d = a - b
    t = d.mean(1) / (d.std(1, ddof=1) / np.sqrt(n))
    return np.abs(t) - stats.t.ppf(1 - ALPHA / 2, n - 1)


def _t_indep(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    n = b.shape[1]
    t = (a.mean(1) - b.mean(1)) / np.sqrt((a.var(1, ddof=1) + b.var(1, ddof=1)) / n)
    return np.abs(t) - stats.t.ppf(1 - ALPHA / 2, 2 * n - 2)


MC_DESIGNS = ((10, -0.5, 0.0), (10, 0.0, 0.5), (20, 0.5, 0.0), (20, 0.8, 0.5), (5, 0.9, 0.5), (40, 0.95, 0.5))


def monte_carlo(reps: int = MC_REPS, seed: int = 184) -> List[Dict]:
    """Raw bivariate-normal replicates; each exact rate must sit inside the 99% Wilson interval."""
    rng = np.random.default_rng(seed)
    out = []
    for n, rho, d in MC_DESIGNS:
        x = _pairs(rng, n, rho, d, reps)
        b, a = x[..., 0], x[..., 1]
        row = {"n": n, "rho": rho, "delta": d}
        for name, fn, exact in (("paired", _t_paired, power_paired(n, rho, d)),
                                ("independent", _t_indep, power_independent(n, rho, d))):
            k = int((fn(b, a) > 0).sum())
            lo, hi = wilson(k, reps)
            row[name] = {"exact": exact, "mc": k / reps, "inside_99": bool(lo <= exact <= hi)}
            if not row[name]["inside_99"]:
                # 12 checks at 99% miss once about 11% of the time by chance. A miss is re-run at
                # 10x the reps on a fresh seed and BOTH results are reported, never just the second.
                x2 = _pairs(np.random.default_rng(seed + 1000), n, rho, d, reps * 10)
                k2 = int((fn(x2[..., 0], x2[..., 1]) > 0).sum())
                lo2, hi2 = wilson(k2, reps * 10)
                row[name]["recheck"] = {"reps": reps * 10, "mc": k2 / (reps * 10),
                                        "inside_99": bool(lo2 <= exact <= hi2)}
        out.append(row)
    return out


def exemplar(seed: int = 0) -> Dict:
    """12 stores, weekly sales before/after a process change. Sales SD $4k, stores correlate 0.85
    week to week, the true lift is $2k (0.5 SD). The first seed where the two analyses disagree is
    shown, and the probability of that disagreement is reported next to it, so it is not a
    cherry-pick presented as typical."""
    n, rho, sd, lift = 12, 0.85, 4.0, 2.0
    for s in range(seed, seed + 500):
        rng = np.random.default_rng(s)
        z = _pairs(rng, n, rho, lift / sd, 1)[0]
        before = np.round(30 + sd * z[:, 0], 1)
        after = np.round(30 + sd * z[:, 1], 1)
        res = analyse(before, after)
        if res["p_paired"] < ALPHA <= res["p_independent"]:
            break
    return {"seed": s, "n": n, "rho": rho, "sd": sd, "true_lift": lift,
            "before": before.tolist(), "after": after.tolist(), **res,
            **_disagreement(n, rho, lift / sd)}


def _disagreement(n: int, rho: float, delta: float, reps: int = MC_REPS) -> Dict:
    """How often, on one dataset, each analysis rejects and the other does not (Monte Carlo:
    the joint event has no closed form)."""
    x = _pairs(np.random.default_rng(1840), n, rho, delta, reps)
    rp, ri = _t_paired(x[..., 0], x[..., 1]) > 0, _t_indep(x[..., 0], x[..., 1]) > 0
    return {"p_only_paired": float((rp & ~ri).mean()), "p_only_independent": float((ri & ~rp).mean())}
