"""p-value and Bayes factor on the same data: where they agree, where they cannot, and why.

The study model is a normal mean with known unit SD, so the test statistic z = mean * sqrt(n) is
exactly N(delta * sqrt(n), 1) and every verdict probability below is a closed-form normal integral.

    p-value          two-sided z-test of H0: delta = 0
    Bayes factor     H0: delta = 0  vs  H1: delta ~ N(0, tau^2), tau in SD units
                     BF01 = sqrt(1 + n tau^2) * exp(-z^2/2 * n tau^2 / (1 + n tau^2))

Real data uses the one-sample t and the JZS Bayes factor (Cauchy prior on delta, scale r), which is
what JASP, the R BayesFactor package and pingouin report. It is computed by quadrature over g and
checked against an independent quadrature over delta and against pingouin.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Sequence

import numpy as np
from scipy import integrate, optimize, stats

ALPHA = 0.05
Z_CRIT = float(stats.norm.isf(ALPHA / 2))
TAU = 1.0  # the unit-information prior: one observation's worth of prior information
R_JZS = float(np.sqrt(2) / 2)  # the default "medium" Cauchy scale in JASP / BayesFactor / pingouin
NS = (10, 30, 100, 300, 1000, 3000, 10000, 100000, 1000000)
TAUS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
MC_REPS = 20000
MC_DESIGNS = ((50, 0.0), (200, 0.1), (1000, 0.0), (1000, 0.07), (2000, 0.05))
Z99 = float(stats.norm.isf(0.005))


def wilson(k: int, n: int, z: float = Z99) -> List[float]:
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(mid - half), float(mid + half)]


def bf01_normal(z: float, n: int, tau: float = TAU) -> float:
    """BF01 for H0: delta = 0 against a N(0, tau^2) alternative, from z = mean * sqrt(n)."""
    s = n * tau * tau
    return float(np.sqrt(1 + s) * np.exp(-0.5 * z * z * s / (1 + s)))


def bf01_normal_by_quadrature(z: float, n: int, tau: float = TAU) -> float:
    """The same Bayes factor as an explicit integral over the prior - the calibration route."""
    lo, hi = z / np.sqrt(n) - 12 / np.sqrt(n), z / np.sqrt(n) + 12 / np.sqrt(n)
    m1, _ = integrate.quad(lambda d: stats.norm.pdf(z, d * np.sqrt(n)) * stats.norm.pdf(d, 0, tau),
                           lo, hi, epsabs=0, epsrel=1e-11, limit=200)
    return float(stats.norm.pdf(z) / m1)


def z_threshold_sq(n: int, k: float, tau: float = TAU) -> float:
    """BF01 > k  <=>  z^2 < this. Negative means BF01 > k is impossible at this n."""
    s = n * tau * tau
    return float((1 + s) / s * (np.log1p(s) - 2 * np.log(k)))


def p_abs_below(a2: float, mu: float) -> float:
    """P(|Z + mu| < sqrt(a2)) for Z standard normal."""
    if a2 <= 0:
        return 0.0
    a = np.sqrt(a2)
    return float(stats.norm.cdf(a - mu) - stats.norm.cdf(-a - mu))


def verdicts(n: int, delta: float, tau: float = TAU) -> Dict[str, float]:
    """Exact probability of each verdict when the true standardised effect is delta."""
    mu = delta * np.sqrt(n)
    zc2 = Z_CRIT ** 2
    below_crit = p_abs_below(zc2, mu)
    t1, t3, t_third = (z_threshold_sq(n, k, tau) for k in (1.0, 3.0, 1 / 3))
    return {
        "n": n, "delta": delta, "tau": tau,
        "p_significant": 1 - below_crit,
        "bf10_over_3": 1 - p_abs_below(t_third, mu),
        "bf01_over_3": p_abs_below(t3, mu),
        # the two analysts disagree: p < alpha AND the Bayes factor favours H0
        "sig_and_bf01_over_1": max(0.0, p_abs_below(t1, mu) - below_crit),
        "sig_and_bf01_over_3": max(0.0, p_abs_below(t3, mu) - below_crit),
    }


def bf01_at_p(p: float, n: int, tau: float = TAU) -> float:
    """Lindley's curve: the Bayes factor of a result whose p-value is held fixed while n grows."""
    return bf01_normal(float(stats.norm.isf(p / 2)), n, tau)


def n_where_p_supports_null(p: float, k: float = 1.0, tau: float = TAU) -> float:
    """Smallest (continuous) n past which a result at exactly this p-value has BF01 >= k.

    BF01 at fixed p is U-shaped in n: it starts at 1, dips to its minimum at n tau^2 = z^2 - 1, then
    grows like sqrt(n). A root search over the whole range can land on the descending crossing (the
    first version did, and reported n = 3.7 for a curve still at 0.58 when n = 10). Bracket from the
    minimum instead, and return n, not the log-n the search runs on (the second defect).
    """
    z = float(stats.norm.isf(p / 2))
    lo = np.log(max(z * z - 1, 1e-9) / tau ** 2)
    return float(np.exp(optimize.brentq(lambda ln: np.log(bf01_at_p(p, np.exp(ln), tau)) - np.log(k), lo, 40)))


def max_bf10_over_normal_priors(z: float) -> Dict[str, float]:
    """The most a N(0, tau^2) alternative can favour H1 at this z: closed form, and by search."""
    closed = float(np.exp((z * z - 1) / 2) / abs(z)) if abs(z) > 1 else 1.0
    res = optimize.minimize_scalar(lambda lt: bf01_normal(z, 1, np.exp(lt)), bounds=(-8, 8), method="bounded",
                                   options={"xatol": 1e-12})
    return {"closed_form": closed, "by_search": float(1 / res.fun), "best_tau_sqrt_n": float(np.exp(res.x))}


def sbb_bound(p: float) -> float:
    """Sellke-Bayarri-Berger (2001): BF10 <= 1 / (-e p ln p) for p < 1/e, over a broad class of alternatives."""
    return float(1 / (-np.e * p * np.log(p)))


def bf10_jzs(t: float, n: int, r: float = R_JZS) -> float:
    """JZS Bayes factor for a one-sample t (Rouder et al. 2009), integrated over g on a log scale."""
    nu = n - 1
    base = (1 + t * t / nu) ** (-(nu + 1) / 2)

    def f(u: float) -> float:
        g = np.exp(u)
        a = 1 + n * g * r * r
        lik = a ** -0.5 * (1 + t * t / (a * nu)) ** (-(nu + 1) / 2)
        prior = (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g))
        return lik / base * prior * g  # dg = g du

    val, _ = integrate.quad(f, -30, 30, epsabs=0, epsrel=1e-10, limit=400)
    return float(val)


def bf10_jzs_by_delta(t: float, n: int, r: float = R_JZS) -> float:
    """The same Bayes factor as an average of the noncentral-t likelihood over a Cauchy prior on delta."""
    nu, c = n - 1, t / np.sqrt(n)
    w = 10 / np.sqrt(n)
    f = lambda d: stats.nct.pdf(t, nu, d * np.sqrt(n)) * stats.cauchy.pdf(d, 0, r)  # noqa: E731
    with warnings.catch_warnings():  # nct.pdf far in the tails warns; those tails carry ~0 mass
        warnings.simplefilter("ignore")
        parts = [integrate.quad(f, a, b, epsabs=0, epsrel=1e-10, limit=400)[0]
                 for a, b in ((-np.inf, c - w), (c - w, c + w), (c + w, np.inf))]
    return float(sum(parts) / stats.t.pdf(t, nu))


def analyse(x: Sequence[float], r: float = R_JZS) -> Dict[str, float]:
    """One-sample (or paired differences) t: the p-value and the JZS Bayes factor on the same data."""
    x = np.asarray([v for v in x], dtype=float)
    if x.size < 3:
        raise ValueError("need at least 3 values")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite numbers")
    sd = float(np.std(x, ddof=1))
    if sd == 0:
        raise ValueError("all values are identical, so there is no variance to test against")
    n = int(x.size)
    t = float(np.mean(x) / sd * np.sqrt(n))
    bf10 = bf10_jzs(t, n, r)
    return {"n": n, "mean": float(np.mean(x)), "d": float(np.mean(x) / sd), "t": t,
            "p": float(2 * stats.t.sf(abs(t), n - 1)), "bf10": bf10, "bf01": 1 / bf10, "r": r}


def n_for_power(delta: float, key: str, target: float = 0.8, tau: float = TAU) -> int:
    """Smallest n whose verdict probability `key` reaches target (scan up, then bisect the bracket)."""
    lo, hi = 2, 2
    while verdicts(hi, delta, tau)[key] < target:
        lo, hi = hi, hi * 2
        if hi > 10 ** 9:
            raise ValueError("target not reached")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if verdicts(mid, delta, tau)[key] >= target else (mid, hi)
    return hi


def calibrate() -> Dict[str, float]:
    """Closed form vs quadrature, g-form JZS vs delta-form JZS vs pingouin, decisions vs scipy."""
    gap_closed = max(abs(np.log(bf01_normal(z, n, tau)) - np.log(bf01_normal_by_quadrature(z, n, tau)))
                     for z in (0.3, 1.96, 3.1) for n in (10, 1000, 100000) for tau in (0.1, 1.0))
    cases = [(t, n) for t in (0.5, 2.1, 3.4) for n in (8, 40, 300)]
    gap_forms = max(abs(np.log(bf10_jzs(t, n)) - np.log(bf10_jzs_by_delta(t, n))) for t, n in cases)
    out = {"max_log_gap_closed_vs_quadrature": float(gap_closed), "max_log_gap_jzs_g_vs_delta": float(gap_forms)}
    try:
        import pingouin as pg

        out["max_log_gap_jzs_vs_pingouin"] = float(max(
            abs(np.log(bf10_jzs(t, n)) - np.log(float(pg.bayesfactor_ttest(t, n, paired=True, r=R_JZS))))
            for t, n in cases))
    except ImportError:
        out["max_log_gap_jzs_vs_pingouin"] = float("nan")
    rng = np.random.default_rng(7)
    mism = 0
    for _ in range(400):
        x = rng.normal(rng.normal(0, 0.3), 1, int(rng.integers(5, 60)))
        a = analyse(x)
        mism += (a["p"] < ALPHA) != (stats.ttest_1samp(x, 0).pvalue < ALPHA)
    out["scipy_decision_mismatches"] = int(mism)
    out["decisions_compared"] = 400
    mx = max_bf10_over_normal_priors(Z_CRIT)
    out["max_bf10_closed_vs_search_gap"] = abs(mx["closed_form"] - mx["by_search"])
    return out


def monte_carlo(reps: int = MC_REPS, seed: int = 11) -> List[Dict]:
    """Raw N(delta, 1) data, z from the sample mean, verdict frequencies vs the exact probabilities."""
    rng = np.random.default_rng(seed)
    out = []
    for n, delta in MC_DESIGNS:
        z = np.empty(reps)
        for i in range(0, reps, 2000):
            k = min(2000, reps - i)
            z[i:i + k] = rng.normal(delta, 1, (k, n)).mean(axis=1) * np.sqrt(n)
        ex = verdicts(n, delta)
        bf = np.sqrt(1 + n) * np.exp(-0.5 * z * z * n / (1 + n))
        freq = {"p_significant": np.abs(z) > Z_CRIT, "bf01_over_3": bf > 3,
                "sig_and_bf01_over_1": (np.abs(z) > Z_CRIT) & (bf > 1)}
        row = {"n": n, "delta": delta}
        for key, hit in freq.items():
            k = int(hit.sum())
            lo, hi = wilson(k, reps)
            row[key] = {"exact": ex[key], "mc": k / reps, "inside_99": bool(lo <= ex[key] <= hi)}
        out.append(row)
    return out


def exemplar(n: int = 10000, delta: float = 0.02) -> Dict:
    """The first seed whose one-sample t lands at 0.03 < p < 0.05: a 'finding' at n = 10,000."""
    for seed in range(1000):
        x = np.random.default_rng(seed).normal(delta, 1, n)
        a = analyse(x)
        if 0.03 < a["p"] < 0.05:
            return {"seed": seed, "true_delta": delta, **a,
                    "bf01_normal_tau1": bf01_normal(a["t"], n, 1.0),
                    "bf01_by_tau": {str(t): bf01_normal(a["t"], n, t) for t in TAUS}}
    raise RuntimeError("no exemplar seed found")
