"""A lead is a claim about a lag, and a lag has to be estimated from the data.

The world below is written so the answer is known. Demand moves, and it reaches
revenue through a funnel - awareness, signups, activations - so every stage is a
genuine leading indicator of revenue with a known lead and a known causal gain.
Around that funnel sit metrics that correlate with revenue and carry no usable
information at all: a sensor that observes demand without being part of the
chain, a metric that follows revenue, a metric that only shares its calendar,
a random walk, and pure noise.

No ranker gets to see a latent variable, the true lead, or the causal gain.

Every constant is declared once in ``World``, and the closed forms below are
derived from those same constants, so no effect is typed in twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# 1. The world
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class World:
    """A four-stage funnel plus five distractors that correlate anyway.

    ``a_t = phi_a * a_{t-1} + eps``           latent demand
    ``s_t = c_s * a_{t-1} + eps``             signups
    ``v_t = c_v * s_{t-1} + eps``             activations
    ``y_t = c_y * v_{t-1} + season_t + eps``  revenue, the lagging KPI

    So revenue at ``t`` is driven by demand at ``t-3``: awareness leads by 3,
    signups by 2, activations by 1. Because each stage passes on a fraction of
    the one before it, the *earliest* stage is also the *weakest* signal - the
    trade-off between warning time and detectability is mechanical here, not
    a modelling choice.
    """

    T: int = 240              # observed periods (months)
    burn: int = 120           # discarded periods, so nothing depends on t=0
    phi_a: float = 0.75       # demand persistence
    sd_a: float = 1.00
    c_s: float = 0.80         # demand -> signups
    sd_s: float = 0.55
    c_v: float = 0.80         # signups -> activations
    sd_v: float = 0.55
    c_y: float = 0.80         # activations -> revenue
    sd_y: float = 0.60
    season_amp_y: float = 1.20    # revenue seasonality
    season_amp_m: float = 1.60    # marketing-spend seasonality
    period: int = 12

    load_web: float = 0.95    # web sessions: a low-noise sensor of demand
    sd_web: float = 0.30
    load_aware: float = 0.90  # awareness survey: a noisy sensor of demand
    sd_aware: float = 0.60
    c_tick: float = 0.80      # revenue -> support tickets (a LAGGING metric)
    sd_tick: float = 0.40
    sd_mkt: float = 0.50      # marketing spend: seasonality and nothing else
    phi_noise: float = 0.60   # the three placebo candidates
    sd_noise: float = 1.00
    seed: int = 20260902

    # ---- closed forms, so the tests can check the simulator ----

    @property
    def gain(self) -> Dict[str, float]:
        """Change in revenue per unit permanently added to each metric.

        Only nodes *inside* the funnel have one. A sensor that merely observes
        demand can be moved without moving anything downstream, and its gain is
        exactly zero however well it forecasts.
        """
        return {
            "signups": self.c_v * self.c_y,
            "activations": self.c_y,
            "web_sessions": 0.0,
            "awareness_index": 0.0,
            "support_tickets": 0.0,
            "marketing_spend": 0.0,
            "nps_trend": 0.0,
            "placebo_1": 0.0,
            "placebo_2": 0.0,
            "placebo_3": 0.0,
        }

    @property
    def true_lead(self) -> Dict[str, int]:
        """Periods of warning each metric actually carries about revenue."""
        return {
            "web_sessions": 3, "awareness_index": 3, "signups": 2,
            "activations": 1, "support_tickets": -1, "marketing_spend": 0,
            "nps_trend": 0, "placebo_1": 0, "placebo_2": 0, "placebo_3": 0,
        }

    @property
    def informative(self) -> List[str]:
        """The metrics that carry information about future revenue at all."""
        return ["web_sessions", "awareness_index", "signups", "activations"]

    @property
    def actionable(self) -> List[str]:
        """The metrics that are worth moving. A strict subset of the above."""
        return [k for k, g in self.gain.items() if g > 0]


CANDIDATES = [
    "web_sessions", "awareness_index", "signups", "activations",
    "support_tickets", "marketing_spend", "nps_trend",
    "placebo_1", "placebo_2", "placebo_3",
]

TRUTH_LABEL = {
    "web_sessions": "informative, not actionable (sensor of demand)",
    "awareness_index": "informative, not actionable (noisy sensor)",
    "signups": "informative and actionable (funnel stage)",
    "activations": "informative and actionable (funnel stage)",
    "support_tickets": "lags revenue - carries nothing new",
    "marketing_spend": "shares the calendar only",
    "nps_trend": "random walk - unrelated",
    "placebo_1": "unrelated AR(1)",
    "placebo_2": "unrelated AR(1)",
    "placebo_3": "unrelated AR(1)",
}


def season(t: np.ndarray, period: int = 12) -> np.ndarray:
    """One annual cycle. Two harmonics, so it is not a pure sine."""
    x = 2.0 * np.pi * t / period
    return np.sin(x) + 0.35 * np.sin(2.0 * x + 0.7)


def simulate(
    w: World,
    seed: Optional[int] = None,
    force: Optional[Tuple[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Run the world once.

    ``force=(name, delta)`` applies the do-operator: that metric is shifted by
    ``delta`` at every period *before* anything downstream reads it. For a
    funnel stage the shift propagates; for a sensor there is nothing downstream
    to propagate to, which is the whole point of the experiment.
    """
    rng = np.random.default_rng(w.seed if seed is None else seed)
    n = w.T + w.burn
    name, delta = force if force is not None else ("", 0.0)

    ea, es, ev, ey = (rng.normal(size=n) for _ in range(4))
    tt = np.arange(n)
    sy = w.season_amp_y * season(tt, w.period)

    a = np.zeros(n)
    s = np.zeros(n)
    v = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        a[t] = w.phi_a * a[t - 1] + w.sd_a * ea[t]
        s[t] = w.c_s * a[t - 1] + w.sd_s * es[t]
        if name == "signups":
            s[t] += delta
        v[t] = w.c_v * s[t - 1] + w.sd_v * ev[t]
        if name == "activations":
            v[t] += delta
        y[t] = w.c_y * v[t - 1] + sy[t] + w.sd_y * ey[t]

    web = w.load_web * a + w.sd_web * rng.normal(size=n)
    aware = w.load_aware * a + w.sd_aware * rng.normal(size=n)
    tick = np.zeros(n)
    tick[1:] = w.c_tick * y[:-1] + w.sd_tick * rng.normal(size=n - 1)
    mkt = w.season_amp_m * season(tt + 1, w.period) + w.sd_mkt * rng.normal(size=n)
    nps = np.cumsum(rng.normal(size=n) * 0.35)

    out: Dict[str, np.ndarray] = {
        "revenue": y, "web_sessions": web, "awareness_index": aware,
        "signups": s, "activations": v, "support_tickets": tick,
        "marketing_spend": mkt, "nps_trend": nps,
        "_demand": a,
    }
    for i in range(1, 4):
        z = np.zeros(n)
        e = rng.normal(size=n)
        for t in range(1, n):
            z[t] = w.phi_noise * z[t - 1] + w.sd_noise * e[t]
        out[f"placebo_{i}"] = z

    # A sensor is observed, not set. Forcing it changes only the column.
    if name in ("web_sessions", "awareness_index", "support_tickets",
                "marketing_spend", "nps_trend") or name.startswith("placebo"):
        out[name] = out[name] + delta

    return {k: val[w.burn:] for k, val in out.items()}


# --------------------------------------------------------------------------
# 2. Rankers. Each one sees only observed columns.
# --------------------------------------------------------------------------


def lagged_corr(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """Correlation between ``x`` lagged ``lag`` periods and ``y``.

    A positive lag means x is read *earlier* than y, i.e. x leads. A negative
    lag means y leads x. Nothing in the arithmetic prefers one sign.
    """
    if lag > 0:
        xa, ya = x[:-lag], y[lag:]
    elif lag < 0:
        xa, ya = x[-lag:], y[:lag]
    else:
        xa, ya = x, y
    if xa.size < 8 or np.std(xa) == 0 or np.std(ya) == 0:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def corr_profile(x: np.ndarray, y: np.ndarray, lags: range) -> Dict[int, float]:
    return {lag: lagged_corr(x, y, lag) for lag in lags}


def rank_pearson_lead(x: np.ndarray, y: np.ndarray, max_lag: int = 12) -> Tuple[float, int]:
    """Strongest positive correlation over lags 1..max_lag. The usual scan."""
    prof = corr_profile(x, y, range(1, max_lag + 1))
    lag = max(prof, key=lambda k: prof[k])
    return prof[lag], lag


def rank_pearson_abs_sym(x: np.ndarray, y: np.ndarray, max_lag: int = 12) -> Tuple[float, int]:
    """Strongest |correlation| over lags -max_lag..max_lag.

    This is what a scan written without sign discipline does, and it is a
    common shape: take the whole cross-correlation function, find the peak,
    call the peak the lag.
    """
    prof = corr_profile(x, y, range(-max_lag, max_lag + 1))
    lag = max(prof, key=lambda k: abs(prof[k]))
    return prof[lag], lag


def _design(cols: List[np.ndarray]) -> np.ndarray:
    return np.column_stack([np.ones(len(cols[0]))] + cols)


def _ols_rss(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return beta, float(r @ r)


def granger_f(
    x: np.ndarray, y: np.ndarray, p: int = 4, deseason: bool = True, period: int = 12
) -> Tuple[float, float]:
    """Does x's history reduce revenue's residual variance given revenue's own?

    This is the question raw correlation cannot answer. A metric that follows
    revenue is correlated with revenue's future only through revenue's own
    persistence, and conditioning on ``y_{t-1..t-p}`` removes exactly that.
    """
    n = len(y)
    t = np.arange(n)
    rows = slice(p, n)
    ylags = [y[p - k: n - k] for k in range(1, p + 1)]
    xlags = [x[p - k: n - k] for k in range(1, p + 1)]
    extra: List[np.ndarray] = []
    if deseason:
        sx = 2.0 * np.pi * t[rows] / period
        extra = [np.sin(sx), np.cos(sx), np.sin(2 * sx), np.cos(2 * sx)]
    target = y[rows]
    Xr = _design(ylags + extra)
    Xf = _design(ylags + xlags + extra)
    _, rss_r = _ols_rss(Xr, target)
    _, rss_f = _ols_rss(Xf, target)
    df_n = p
    df_d = len(target) - Xf.shape[1]
    if df_d <= 0 or rss_f <= 0:
        return 0.0, 1.0
    f = ((rss_r - rss_f) / df_n) / (rss_f / df_d)
    return float(f), float(stats.f.sf(f, df_n, df_d))


def deseasonalize(z: np.ndarray, period: int = 12) -> np.ndarray:
    """Remove two annual harmonics by least squares."""
    t = np.arange(len(z))
    sx = 2.0 * np.pi * t / period
    X = _design([np.sin(sx), np.cos(sx), np.sin(2 * sx), np.cos(2 * sx)])
    beta, _ = _ols_rss(X, z)
    return z - X @ beta


def rank_prewhitened(
    x: np.ndarray, y: np.ndarray, max_lag: int = 12, period: int = 12
) -> Tuple[float, int]:
    """Box-Jenkins style: strip the calendar and the level, then cross-correlate.

    Two series that only share a calendar have nothing left after this, and two
    random walks stop drifting together.
    """
    dx = np.diff(deseasonalize(x, period))
    dy = np.diff(deseasonalize(y, period))
    return rank_pearson_lead(dx, dy, max_lag)


def naive_corr_pvalue(r: float, n: int) -> float:
    """The p-value a spreadsheet reports for a correlation: iid assumed."""
    if n <= 3 or abs(r) >= 1:
        return 0.0 if abs(r) >= 1 else 1.0
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return float(2 * stats.t.sf(abs(t), n - 2))


def ac1(z: np.ndarray) -> float:
    return lagged_corr(z, z, 1)


def bartlett_corr_pvalue(r: float, n: int, rx: float, ry: float) -> float:
    """Bartlett's correction: two autocorrelated series share fewer than n facts.

    ``n_eff = n * (1 - rx*ry) / (1 + rx*ry)`` with rx, ry the lag-1
    autocorrelations. Cheap, well known, and it is the difference between a
    calibrated test and a test that fires on nothing.
    """
    rho = float(np.clip(rx * ry, -0.99, 0.99))
    n_eff = max(4.0, n * (1.0 - rho) / (1.0 + rho))
    return naive_corr_pvalue(r, int(round(n_eff)))


# --------------------------------------------------------------------------
# 3. The criterion that decides: does it beat doing nothing, at my horizon?
# --------------------------------------------------------------------------


def oos_gain(
    y: np.ndarray,
    x: np.ndarray,
    h: int,
    max_lag: int = 12,
    min_train: int = 96,
    p: int = 3,
    period: int = 12,
) -> Dict[str, float]:
    """Rolling-origin forecast of revenue ``h`` periods ahead.

    Standing at period ``t`` we know ``y`` up to ``t`` and ``x`` up to ``t``, so
    the only usable readings of ``x`` are lags of ``h`` or more relative to the
    target. A metric whose lead is shorter than the horizon cannot help at all,
    however strongly it correlates - and that constraint is the reason a lead
    has to be judged against the horizon somebody actually needs.

    The lag is chosen on the training window only. Choosing it on the full
    series is the leak that makes every candidate look useful.
    """
    n = len(y)
    t_all = np.arange(n)
    sx = 2.0 * np.pi * t_all / period
    harm = [np.sin(sx), np.cos(sx), np.sin(2 * sx), np.cos(2 * sx)]
    lags = [lag for lag in range(max(h, 1), max_lag + 1)]
    if not lags:
        return {"rmse_base": np.nan, "rmse_aug": np.nan, "gain_pct": 0.0, "lag": 0, "n": 0}

    err_b, err_a, err_sn, chosen = [], [], [], []
    for origin in range(min_train, n - h):
        # rows i are origins in the training window: target y[i+h]
        tr = np.arange(p, origin - h + 1)
        if tr.size < 40:
            continue
        ytr = y[tr + h]
        base_tr = [y[tr - k] for k in range(0, p)] + [c[tr + h] for c in harm]
        Xb = _design(base_tr)
        beta_b, rss_b = _ols_rss(Xb, ytr)

        best_lag, best_rss = lags[0], np.inf
        for lag in lags:
            xf = x[tr + h - lag]
            _, rss = _ols_rss(_design(base_tr + [xf]), ytr)
            if rss < best_rss:
                best_rss, best_lag = rss, lag
        beta_a, _ = _ols_rss(_design(base_tr + [x[tr + h - best_lag]]), ytr)

        row_b = np.array([1.0] + [y[origin - k] for k in range(0, p)]
                         + [c[origin + h] for c in harm])
        row_a = np.append(row_b, x[origin + h - best_lag])
        truth = y[origin + h]
        err_b.append(truth - row_b @ beta_b)
        err_a.append(truth - row_a @ beta_a)
        err_sn.append(truth - y[origin + h - period])
        chosen.append(best_lag)

    if not err_b:
        return {"rmse_base": np.nan, "rmse_aug": np.nan, "gain_pct": 0.0, "lag": 0, "n": 0}
    rb = float(np.sqrt(np.mean(np.square(err_b))))
    ra = float(np.sqrt(np.mean(np.square(err_a))))
    rs = float(np.sqrt(np.mean(np.square(err_sn))))
    dm = diebold_mariano(np.array(err_b), np.array(err_a))
    return {
        "rmse_base": rb, "rmse_aug": ra, "rmse_seasonal_naive": rs,
        "gain_pct": 100.0 * (rb - ra) / rb,
        "lag": int(stats.mode(np.array(chosen), keepdims=False).mode),
        "lag_spread": int(np.ptp(np.array(chosen))),
        "n": len(err_b), "dm_p": dm,
    }


def diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> float:
    """Is the RMSE difference bigger than the noise in the difference?

    A percentage improvement on 140 overlapping forecasts is not evidence on
    its own. This is the one-sided test that the augmented model is better,
    with a Newey-West variance because the loss differential is autocorrelated.
    """
    d = np.square(e1) - np.square(e2)
    n = len(d)
    dbar = float(np.mean(d))
    lag = max(1, int(round(n ** (1 / 3))))
    g0 = float(np.mean((d - dbar) ** 2))
    var = g0
    for k in range(1, lag + 1):
        gk = float(np.mean((d[k:] - dbar) * (d[:-k] - dbar)))
        var += 2.0 * (1 - k / (lag + 1)) * gk
    if var <= 0:
        return 1.0
    stat = dbar / np.sqrt(var / n)
    return float(stats.norm.sf(stat))


# --------------------------------------------------------------------------
# 4. A world with nothing in it, so every test can be graded
# --------------------------------------------------------------------------


def simulate_null(
    T: int,
    seed: int,
    kind: str = "ar1",
    k: int = 10,
    phi: float = 0.60,
    period: int = 12,
    season_amp: float = 1.20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Revenue that looks exactly like a real KPI, and candidates related to it
    in no way whatsoever.

    Revenue is persistent and seasonal because real revenue is. The candidates
    are drawn independently. Any indicator this world produces is false, so the
    rate at which a method produces one is that method's false-positive rate -
    measured, not nominal.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    y = np.zeros(T)
    e = rng.normal(size=T)
    for i in range(1, T):
        y[i] = 0.70 * y[i - 1] + e[i]
    y = y + season_amp * season(t, period)
    X = np.zeros((k, T))
    for j in range(k):
        z = np.zeros(T)
        ez = rng.normal(size=T)
        if kind == "rw":
            z = np.cumsum(ez)
        else:
            for i in range(1, T):
                z[i] = phi * z[i - 1] + ez[i]
        X[j] = z
    return y, X


def scan_flags(
    y: np.ndarray, X: np.ndarray, max_lag: int = 12, alpha: float = 0.05
) -> Dict[str, bool]:
    """Run six ways of declaring a leading indicator on one panel.

    ``fixed_lag`` tests a single pre-registered lag. ``scan`` takes the best of
    ``max_lag`` lags and reports its p-value as if that lag had been the plan.
    The rest add the corrections that are supposed to make the scan safe.
    """
    k, n = X.shape
    ry = ac1(y)
    fixed_lag = 3
    n_tests = k * max_lag

    best_p_naive, best_p_bart = 1.0, 1.0
    any_fixed, any_fixed_bart = False, False
    one_naive, one_bart = False, False
    for j in range(k):
        x = X[j]
        rx = ac1(x)
        rf = lagged_corr(x, y, fixed_lag)
        nf = n - fixed_lag
        any_fixed |= naive_corr_pvalue(rf, nf) < alpha
        any_fixed_bart |= bartlett_corr_pvalue(rf, nf, rx, ry) < alpha
        if j == 0:
            one_naive = naive_corr_pvalue(rf, nf) < alpha
            one_bart = bartlett_corr_pvalue(rf, nf, rx, ry) < alpha
        for lag in range(1, max_lag + 1):
            r = lagged_corr(x, y, lag)
            m = n - lag
            best_p_naive = min(best_p_naive, naive_corr_pvalue(r, m))
            best_p_bart = min(best_p_bart, bartlett_corr_pvalue(r, m, rx, ry))
    g_best = 1.0
    for j in range(k):
        g_best = min(g_best, granger_f(X[j], y)[1])
    return {
        "one_test_naive": one_naive,
        "one_test_bartlett": one_bart,
        "fixed_lag_naive": any_fixed,
        "fixed_lag_bartlett": any_fixed_bart,
        "scan_naive": best_p_naive < alpha,
        "scan_bonferroni": best_p_naive < alpha / n_tests,
        "scan_bartlett": best_p_bart < alpha,
        "scan_bartlett_bonferroni": best_p_bart < alpha / n_tests,
        "granger_best_of_k": g_best < alpha,
        "granger_bonferroni": g_best < alpha / k,
    }


FLAG_LABEL = {
    "one_test_naive": "one candidate, one lag, textbook p-value",
    "one_test_bartlett": "one candidate, one lag, Bartlett-corrected",
    "fixed_lag_naive": "one pre-registered lag, textbook p-value",
    "fixed_lag_bartlett": "one pre-registered lag, Bartlett-corrected",
    "scan_naive": "best of 10 x 12, textbook p-value",
    "scan_bonferroni": "best of 10 x 12, Bonferroni",
    "scan_bartlett": "best of 10 x 12, Bartlett only",
    "scan_bartlett_bonferroni": "best of 10 x 12, Bartlett + Bonferroni",
    "granger_best_of_k": "Granger F, best of 10 candidates",
    "granger_bonferroni": "Granger F, best of 10, Bonferroni",
}
