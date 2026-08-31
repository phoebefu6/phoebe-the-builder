"""A proxy metric is a bet that a correlation survives being optimised.

The world below is written so the answer is known: a latent quality driver feeds
both the outcome somebody cares about and a proxy somebody can see, and the proxy
has a second channel that moves it without moving the outcome. Nothing in the
detectors gets to look at the latent variables.

Every constant is declared once in ``World``. Both the honest channel and the
exploitable one are driven from it, so no effect is typed in twice.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# 1. The world
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class World:
    """One latent driver, one outcome, one proxy, one exploit.

    ``y_i = a_y * L_i + noise``            the thing that is actually wanted
    ``p_i = beta * L_i + gamma * u_i + noise``   the thing that is measured
    ``L_i = s_i + kappa * (1 - u_i)``      latent quality; effort share ``u_i``
                                            diverted to the exploit stops
                                            producing quality

    An agent raises the proxy by ``gamma - beta * kappa`` per unit of diverted
    effort while lowering the outcome by ``a_y * kappa``. When ``gamma`` exceeds
    ``beta * kappa`` the exploit is the cheaper way to move the number, and that
    single inequality is the whole of Goodhart's law in this model.
    """

    n_agents: int = 600
    kappa: float = 0.60      # quality bought by a unit of honest effort
    a_y: float = 1.00        # outcome loading on latent quality
    sigma_y: float = 0.50    # outcome noise
    beta: float = 1.00       # proxy loading on latent quality
    gamma: float = 1.10      # proxy loading on the exploit
    sigma_p: float = 0.50    # proxy noise
    scruple_median: float = 1.80   # exploit/honest payoff ratio an agent demands
    scruple_sigma: float = 0.50
    hazard: float = 0.35     # per-period adoption hazard among willing agents
    seed: int = 20260831

    # ---- closed forms, so the tests can check the simulator ----

    @property
    def exploit_edge(self) -> float:
        """Proxy points bought per unit of diverted effort."""
        return self.gamma - self.beta * self.kappa

    @property
    def outcome_cost(self) -> float:
        """Outcome points destroyed per unit of diverted effort."""
        return self.a_y * self.kappa

    @property
    def payoff_ratio(self) -> float:
        """How much better the exploit is than honest work, on the proxy."""
        return self.gamma / (self.beta * self.kappa)

    @property
    def rho_clean(self) -> float:
        """corr(p, y) when nobody games. var(L) = 1 by construction."""
        num = self.beta * self.a_y
        den = np.hypot(self.beta, self.sigma_p) * np.hypot(self.a_y, self.sigma_y)
        return float(num / den)


@dataclass(frozen=True)
class Panel:
    """A simulated history. ``t < t_target`` is the pre-target regime."""

    proxy: np.ndarray        # (T, n) observed proxy
    outcome: np.ndarray      # (T, n) observed outcome
    holdout: np.ndarray      # (T, n) a second proxy that was never a target
    diverted: np.ndarray     # (T, n) latent effort share on the exploit
    t_target: int
    threshold: Optional[float]

    @property
    def n_periods(self) -> int:
        return self.proxy.shape[0]

    def pre(self, arr: np.ndarray) -> np.ndarray:
        return arr[: self.t_target].ravel()

    def post(self, arr: np.ndarray, upto: Optional[int] = None) -> np.ndarray:
        end = self.n_periods if upto is None else upto
        return arr[self.t_target : end].ravel()


def simulate(
    world: World,
    n_pre: int = 6,
    n_post: int = 12,
    regime: str = "continuous",
    threshold_q: float = 0.75,
    margin: float = 0.15,
    gaming: bool = True,
) -> Panel:
    """Run ``n_pre`` periods before the proxy becomes a target and ``n_post`` after.

    ``regime="continuous"``: the target is "make the number go up", so a willing
    agent diverts everything.  ``regime="threshold"``: the target is "clear P*",
    so a willing agent diverts only as much as the gap requires and then stops.
    ``gaming=False`` runs the same world with the exploit switched off, which is
    the null the detectors have to be measured against.
    """
    if regime not in ("continuous", "threshold"):
        raise ValueError(f"unknown regime {regime!r}")
    rng = np.random.default_rng(world.seed)
    n = world.n_agents
    T = n_pre + n_post

    skill = rng.standard_normal(n)
    scruple = np.exp(np.log(world.scruple_median) + world.scruple_sigma * rng.standard_normal(n))
    # Nobody diverts effort unless the exploit actually moves the proxy further
    # than the honest work it replaces. Scruple only ranks who goes first.
    pays = gaming and world.exploit_edge > 0
    willing = scruple < world.payoff_ratio if pays else np.zeros(n, dtype=bool)
    # willing agents do not all switch on the same day
    adopted_at = np.full(n, T + 1, dtype=int)
    live = willing.copy()
    for t in range(n_pre, T):
        fires = live & (rng.random(n) < world.hazard)
        adopted_at[fires] = t
        live &= ~fires

    proxy = np.zeros((T, n))
    outcome = np.zeros((T, n))
    holdout = np.zeros((T, n))
    diverted = np.zeros((T, n))
    threshold: Optional[float] = None

    for t in range(T):
        active = t >= adopted_at
        eps_p = world.sigma_p * rng.standard_normal(n)
        eps_y = world.sigma_y * rng.standard_normal(n)
        eps_h = world.sigma_p * rng.standard_normal(n)
        if not active.any():
            u = np.zeros(n)
        elif regime == "continuous":
            u = active.astype(float)
        else:
            # A quota is topped up late, against the number as it actually stands.
            # The agent sees its realised proxy, not its latent quality, and aims
            # a margin past the line rather than at it.
            raw = world.beta * (skill + world.kappa) + eps_p
            gap = threshold + margin - raw
            need = gap / world.exploit_edge
            u = np.where(active & (need > 0) & (need <= 1.0), np.clip(need, 0.0, 1.0), 0.0)
        latent = skill + world.kappa * (1.0 - u)
        proxy[t] = world.beta * latent + world.gamma * u + eps_p
        outcome[t] = world.a_y * latent + eps_y
        # the holdout shares the latent driver but is not a target, so no exploit
        holdout[t] = world.beta * latent + eps_h
        diverted[t] = u
        if regime == "threshold" and t == n_pre - 1 and threshold is None:
            threshold = float(np.quantile(proxy[:n_pre].ravel(), threshold_q))

    return Panel(proxy, outcome, holdout, diverted, t_target=n_pre, threshold=threshold)


# --------------------------------------------------------------------------
# 2. Detectors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    name: str
    needs_outcome: bool
    stat: float
    pvalue: float

    def fires(self, alpha: float = 0.05) -> bool:
        return self.pvalue < alpha


def _fisher_z(r: float, n: int) -> Tuple[float, float]:
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r)), float(1.0 / np.sqrt(max(n - 3, 1)))


def corr_drop(panel: Panel, upto: Optional[int] = None) -> Verdict:
    """Did corr(proxy, outcome) fall? One-sided Fisher z on two samples."""
    pre = (panel.pre(panel.proxy), panel.pre(panel.outcome))
    post = (panel.post(panel.proxy, upto), panel.post(panel.outcome, upto))
    r1 = float(np.corrcoef(*pre)[0, 1])
    r2 = float(np.corrcoef(*post)[0, 1])
    z1, s1 = _fisher_z(r1, pre[0].size)
    z2, s2 = _fisher_z(r2, post[0].size)
    z = (z1 - z2) / np.hypot(s1, s2)
    return Verdict("corr_drop", True, r2 - r1, float(stats.norm.sf(z)))


def ratio_shift(panel: Panel, upto: Optional[int] = None, n_boot: int = 800) -> Verdict:
    """Did outcome-per-unit-of-proxy fall?

    Taken on the aggregate the way a dashboard takes it -- total outcome over
    total proxy -- because a per-agent ratio is undefined wherever the proxy is
    near zero. Bootstrapped over agent-periods, one-sided.
    """
    a_p, a_y = panel.pre(panel.proxy), panel.pre(panel.outcome)
    b_p, b_y = panel.post(panel.proxy, upto), panel.post(panel.outcome, upto)
    obs = (b_y.sum() / b_p.sum()) - (a_y.sum() / a_p.sum())
    rng = np.random.default_rng(11)
    i = rng.integers(0, a_p.size, (n_boot, a_p.size))
    j = rng.integers(0, b_p.size, (n_boot, b_p.size))
    d = (b_y[j].sum(1) / b_p[j].sum(1)) - (a_y[i].sum(1) / a_p[i].sum(1))
    hits = int((d >= 0.0).sum())
    return Verdict("ratio_shift", True, float(obs), float((hits + 1) / (n_boot + 1)))


def residual_trend(panel: Panel, upto: Optional[int] = None) -> Verdict:
    """Fit outcome ~ proxy on the pre-period, then test the post-period residual."""
    x, y = panel.pre(panel.proxy), panel.pre(panel.outcome)
    slope, intercept = np.polyfit(x, y, 1)
    resid = panel.post(panel.outcome, upto) - (slope * panel.post(panel.proxy, upto) + intercept)
    t, p_two = stats.ttest_1samp(resid, 0.0)
    return Verdict("residual_trend", True, float(resid.mean()),
                   float(p_two / 2 if t < 0 else 1 - p_two / 2))


def rank_reshuffle(panel: Panel, upto: Optional[int] = None) -> Verdict:
    """Do the proxy leaderboard and the outcome leaderboard still agree?"""
    r1 = stats.spearmanr(panel.pre(panel.proxy), panel.pre(panel.outcome)).statistic
    r2 = stats.spearmanr(panel.post(panel.proxy, upto), panel.post(panel.outcome, upto)).statistic
    z1, s1 = _fisher_z(float(r1), panel.pre(panel.proxy).size)
    z2, s2 = _fisher_z(float(r2), panel.post(panel.proxy, upto).size)
    z = (z1 - z2) / np.hypot(s1, s2)
    return Verdict("rank_reshuffle", True, float(r2 - r1), float(stats.norm.sf(z)))


def holdout_divergence(panel: Panel, upto: Optional[int] = None) -> Verdict:
    """Compare the target against a sibling proxy nobody was told about.

    Needs no outcome: if the two shared a driver before and stopped sharing it
    after, something moved the target that did not move the driver.
    """
    r1 = float(np.corrcoef(panel.pre(panel.proxy), panel.pre(panel.holdout))[0, 1])
    r2 = float(np.corrcoef(panel.post(panel.proxy, upto), panel.post(panel.holdout, upto))[0, 1])
    z1, s1 = _fisher_z(r1, panel.pre(panel.proxy).size)
    z2, s2 = _fisher_z(r2, panel.post(panel.proxy, upto).size)
    z = (z1 - z2) / np.hypot(s1, s2)
    return Verdict("holdout_divergence", False, r2 - r1, float(stats.norm.sf(z)))


def bunching(panel: Panel, upto: Optional[int] = None, width: float = 0.35) -> Verdict:
    """Excess mass just above the target line. Needs no outcome, needs a line."""
    if panel.threshold is None:
        return Verdict("bunching", False, float("nan"), 1.0)
    lo, hi = panel.threshold - width, panel.threshold + width

    def counts(v: np.ndarray) -> Tuple[int, int]:
        return int(((v >= panel.threshold) & (v < hi)).sum()), int(((v >= lo) & (v < panel.threshold)).sum())

    a_hi, a_lo = counts(panel.pre(panel.proxy))
    b_hi, b_lo = counts(panel.post(panel.proxy, upto))
    table = np.array([[b_hi, b_lo], [a_hi, a_lo]])
    if table.min() == 0:
        return Verdict("bunching", False, float("nan"), 1.0)
    chi2, p_two, _, _ = stats.chi2_contingency(table)
    ratio_post, ratio_pre = b_hi / b_lo, a_hi / a_lo
    return Verdict("bunching", False, float(ratio_post - ratio_pre),
                   float(p_two / 2 if ratio_post > ratio_pre else 1 - p_two / 2))


def dispersion_shift(panel: Panel, upto: Optional[int] = None) -> Verdict:
    """Gaming compresses the proxy toward wherever it is being pushed."""
    a, b = panel.pre(panel.proxy), panel.post(panel.proxy, upto)
    f = np.var(a, ddof=1) / np.var(b, ddof=1)
    p_two = 2 * min(
        stats.f.cdf(f, a.size - 1, b.size - 1), stats.f.sf(f, a.size - 1, b.size - 1)
    )
    return Verdict("dispersion_shift", False, float(np.std(b) - np.std(a)), float(p_two))


DETECTORS = [
    corr_drop,
    ratio_shift,
    residual_trend,
    rank_reshuffle,
    holdout_divergence,
    bunching,
    dispersion_shift,
]

DETECTOR_NAMES = [d.__name__ for d in DETECTORS]


def run_all(panel: Panel, upto: Optional[int] = None) -> Dict[str, Verdict]:
    return {d.__name__: d(panel, upto) for d in DETECTORS}


# --------------------------------------------------------------------------
# 3. Decomposition: how much of the proxy's rise was real?
# --------------------------------------------------------------------------


def decompose(world: World, panel: Panel) -> Dict[str, float]:
    """Split the proxy's movement into the part the outcome shared and the rest."""
    u_pre = panel.pre(panel.diverted).mean()
    u_post = panel.post(panel.diverted).mean()
    du = u_post - u_pre
    honest = -world.beta * world.kappa * du     # diverted effort also costs the honest channel
    exploit = world.gamma * du
    return {
        "diverted_share": float(u_post),
        "proxy_delta": float(panel.post(panel.proxy).mean() - panel.pre(panel.proxy).mean()),
        "proxy_from_honest": float(honest),
        "proxy_from_exploit": float(exploit),
        "outcome_delta": float(panel.post(panel.outcome).mean() - panel.pre(panel.outcome).mean()),
        "outcome_delta_true": float(-world.outcome_cost * du),
        "proxy_delta_true": float(honest + exploit),
        "exploit_share_of_gain": float(exploit / (exploit + honest)) if (exploit + honest) else float("nan"),
        "exchange_rate": float(
            (panel.post(panel.outcome).mean() - panel.pre(panel.outcome).mean())
            / (panel.post(panel.proxy).mean() - panel.pre(panel.proxy).mean())
        ),
        "exchange_rate_closed": float(-world.outcome_cost / world.exploit_edge),
    }


# --------------------------------------------------------------------------
# 4. Choosing a proxy: k candidates, keep the best-correlated one
# --------------------------------------------------------------------------


def candidate_worlds(world: World, k: int, rng: np.random.Generator) -> List[World]:
    """k plausible proxies with different honest loadings and different exploits."""
    betas = rng.uniform(0.55, 1.15, size=k)
    gammas = rng.uniform(0.45, 1.35, size=k)
    return [replace(world, beta=float(b), gamma=float(g), seed=int(rng.integers(1, 2**31)))
            for b, g in zip(betas, gammas)]


def simulate_candidates(
    world: World, betas: np.ndarray, gammas: np.ndarray, n_periods: int = 12, seed: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """One outcome, k competing proxies for it, nobody gaming anything.

    Used to ask what happens to the *chosen* proxy purely because it was chosen.
    Every candidate is a proxy for the same latent driver and the same outcome,
    which is the situation a metric review is actually in.
    """
    rng = np.random.default_rng(seed)
    n, k = world.n_agents, len(betas)
    skill = rng.standard_normal(n)
    latent = skill[None, :] + world.kappa          # no diverted effort anywhere
    latent = np.repeat(latent, n_periods, axis=0)
    outcome = world.a_y * latent + world.sigma_y * rng.standard_normal((n_periods, n))
    proxies = (
        betas[:, None, None] * latent[None, :, :]
        + world.sigma_p * rng.standard_normal((k, n_periods, n))
    )
    return outcome, proxies


def true_rho(world: World, beta: float) -> float:
    """corr(p, y) in the population, for a proxy with honest loading ``beta``."""
    return float(
        beta * world.a_y
        / (np.hypot(beta, world.sigma_p) * np.hypot(world.a_y, world.sigma_y))
    )
