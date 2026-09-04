"""CUPED and the alternatives people reach for instead, on worlds whose true
effect and true variance reduction are known in closed form.

Three parts, kept apart because they answer different questions:

* ``adjusters`` take (pre, post, arm) and return one effect estimate. They are
  the things being compared.
* ``World`` / ``simulate`` produce the data, from a process whose treatment
  effect and pre/post correlation are set rather than measured.
* ``analytic`` holds the closed forms - variance reduction is exactly rho^2,
  and that identity is what the simulation gets checked against.

Reference: Deng, Xu, Kohavi & Walker (2013), "Improving the Sensitivity of
Online Controlled Experiments by Utilizing Pre-Experiment Data" (WSDM).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# 1. The world
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class World:
    """A bivariate (pre-period, in-experiment) metric with a known effect.

    ``rho`` is the population correlation between the pre-period value and the
    in-experiment value on the SAME user. It is the only thing CUPED runs on,
    and everything below is a statement about it.

    ``new_user_share`` are users with no pre-period at all. Their covariate is
    imputed at the mean, which is what every real implementation does, and it
    is the main reason a measured reduction falls short of the promise.

    ``lognormal`` exponentiates both margins - the revenue-shaped case, where a
    strong relationship on the log scale is a weak correlation on the scale the
    metric is actually reported on.
    """

    per_arm: int = 3_000
    rho: float = 0.60
    mean: float = 10.0
    sd: float = 4.0
    sd_pre: Optional[float] = None  # defaults to sd; a longer pre-window has a LARGER sd
    true_rel_lift: float = 0.02
    new_user_share: float = 0.0
    lognormal: bool = False
    log_sigma: float = 1.0
    multiplicative: bool = False
    # Composition damage, for the cross-link to Day 165 `srm-detector`. Both
    # remove the same fraction of the treatment arm; one removes users the
    # covariate can see, the other removes users it cannot.
    drop_low_pre: float = 0.0
    drop_low_residual: float = 0.0

    @property
    def true_effect(self) -> float:
        return self.mean * self.true_rel_lift

    @property
    def sd_pre_eff(self) -> float:
        return self.sd if self.sd_pre is None else self.sd_pre


def lognormal_pearson_rho(log_rho: float, sigma: float) -> float:
    """Pearson correlation of two lognormals whose LOGS correlate at ``log_rho``.

    (exp(log_rho * sigma^2) - 1) / (exp(sigma^2) - 1).  A perfectly ordinary
    relationship on the log scale is a much weaker one on the reported scale,
    and CUPED only ever sees the reported scale.
    """
    return float((np.exp(log_rho * sigma * sigma) - 1.0) / (np.exp(sigma * sigma) - 1.0))


def simulate(
    world: World,
    trials: int,
    rng: np.random.Generator,
    effect_on_pre: bool = False,
) -> Dict[str, np.ndarray]:
    """Return per-trial arrays of pre/post values for both arms.

    ``effect_on_pre=True`` is the mistake in section 6: a covariate measured
    AFTER assignment, which is the same three lines of code and absorbs the
    treatment effect instead of the noise.
    """
    m = world.per_arm
    shape = (trials, m)

    def draw() -> Tuple[np.ndarray, np.ndarray]:
        z1 = rng.standard_normal(shape)
        z2 = rng.standard_normal(shape)
        pre_std = z1
        post_std = world.rho * z1 + np.sqrt(max(1.0 - world.rho ** 2, 0.0)) * z2
        if world.lognormal:
            s = world.log_sigma
            # exp(s*z) has mean exp(s^2/2); rescale so the reported metric keeps
            # the same mean and sd as the Gaussian case, which makes the two
            # worlds comparable on everything except tail shape.
            raw_pre = np.exp(s * pre_std)
            raw_post = np.exp(s * post_std)
            mu = np.exp(s * s / 2.0)
            sd = mu * np.sqrt(np.exp(s * s) - 1.0)
            pre = world.mean + world.sd_pre_eff * (raw_pre - mu) / sd
            post = world.mean + world.sd * (raw_post - mu) / sd
        else:
            pre = world.mean + world.sd_pre_eff * pre_std
            post = world.mean + world.sd * post_std
        return pre, post

    pre_c, post_c = draw()
    pre_t, post_t = draw()
    if world.multiplicative:
        # A relative lift applied per user, so the treatment scales each user's
        # value rather than adding a constant. This is the case where theta
        # genuinely differs between the two arms.
        post_t = post_t * (1.0 + world.true_rel_lift)
    else:
        post_t = post_t + world.true_effect
    if effect_on_pre:
        pre_t = pre_t + world.true_effect

    if world.drop_low_pre > 0 or world.drop_low_residual > 0:
        # Selection applied to the treatment arm only. `drop_low_pre` selects on
        # the covariate (so the adjustment can see it); `drop_low_residual`
        # selects on the part of the outcome the covariate cannot explain.
        rate = world.drop_low_pre if world.drop_low_pre > 0 else world.drop_low_residual
        key = pre_t if world.drop_low_pre > 0 else (post_t - world.rho * (world.sd / world.sd_pre_eff) * pre_t)
        cut = np.quantile(key, rate, axis=1, keepdims=True)
        keep = key > cut
        # Keep the array rectangular: replace dropped users with a resampled
        # survivor from the same trial, which is what a filter that removes rows
        # and leaves the rest looks like once the arm is re-read.
        idx = np.argsort(~keep, axis=1)  # survivors first
        n_keep = keep.sum(axis=1)
        take = np.zeros_like(idx)
        for i in range(trials):
            k = max(int(n_keep[i]), 1)
            take[i] = idx[i][rng.integers(0, k, m)]
        pre_t = np.take_along_axis(pre_t, take, axis=1)
        post_t = np.take_along_axis(post_t, take, axis=1)

    if world.new_user_share > 0:
        # New users have no pre-period. Mark them; the adjusters impute.
        new_c = rng.random(shape) < world.new_user_share
        new_t = rng.random(shape) < world.new_user_share
    else:
        new_c = np.zeros(shape, dtype=bool)
        new_t = np.zeros(shape, dtype=bool)

    return {"pre_c": pre_c, "post_c": post_c, "pre_t": pre_t, "post_t": post_t,
            "new_c": new_c, "new_t": new_t}


def _impute(pre: np.ndarray, is_new: np.ndarray) -> np.ndarray:
    """Replace missing pre-period values with the observed mean, per trial."""
    if not is_new.any():
        return pre
    out = pre.copy()
    for i in range(pre.shape[0]):
        mask = is_new[i]
        if mask.all():
            out[i] = 0.0
            continue
        out[i][mask] = pre[i][~mask].mean()
    return out


# --------------------------------------------------------------------------
# 2. The adjusters
#
# Each returns (estimate, se) per trial. They differ ONLY in what they do with
# the pre-period column, which is the whole subject.
# --------------------------------------------------------------------------


def _welch_se(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = a.shape[1], b.shape[1]
    return np.sqrt(a.var(axis=1, ddof=1) / na + b.var(axis=1, ddof=1) / nb)


def adj_none(d: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """No adjustment - the difference in means."""
    yc, yt = d["post_c"], d["post_t"]
    return yt.mean(axis=1) - yc.mean(axis=1), _welch_se(yt, yc)


def _cuped_core(d, theta_mode: str):
    yc, yt = d["post_c"], d["post_t"]
    xc = _impute(d["pre_c"], d["new_c"])
    xt = _impute(d["pre_t"], d["new_t"])

    def theta_of(x, y):
        xm = x - x.mean(axis=1, keepdims=True)
        ym = y - y.mean(axis=1, keepdims=True)
        var = (xm * xm).mean(axis=1)
        cov = (xm * ym).mean(axis=1)
        return np.divide(cov, var, out=np.zeros_like(cov), where=var > 0)

    if theta_mode == "pooled":
        x_all = np.concatenate([xc, xt], axis=1)
        y_all = np.concatenate([yc, yt], axis=1)
        # Centre each arm before pooling, so the treatment effect itself does
        # not leak into the covariance that sets theta.
        x_all = np.concatenate([xc - xc.mean(axis=1, keepdims=True),
                                xt - xt.mean(axis=1, keepdims=True)], axis=1)
        y_all = np.concatenate([yc - yc.mean(axis=1, keepdims=True),
                                yt - yt.mean(axis=1, keepdims=True)], axis=1)
        th = theta_of(x_all, y_all)
        th_c = th_t = th
    elif theta_mode == "per_arm":
        th_c, th_t = theta_of(xc, yc), theta_of(xt, yt)
    elif theta_mode == "unit":
        th_c = th_t = np.ones(yc.shape[0])
    else:
        raise ValueError(theta_mode)

    grand = np.concatenate([xc, xt], axis=1).mean(axis=1, keepdims=True)
    ac = yc - th_c[:, None] * (xc - grand)
    at = yt - th_t[:, None] * (xt - grand)
    return at.mean(axis=1) - ac.mean(axis=1), _welch_se(at, ac)


def adj_cuped(d):
    """CUPED with one theta estimated on both arms pooled - the paper's method."""
    return _cuped_core(d, "pooled")


def adj_cuped_per_arm(d):
    """CUPED with theta estimated separately in each arm - the tempting variant."""
    return _cuped_core(d, "per_arm")


def adj_diff_in_diff(d):
    """theta forced to 1: 'just subtract each user's pre-period value'."""
    return _cuped_core(d, "unit")


def adj_post_strat(d, n_bins: int = 10):
    """Post-stratification on pre-period deciles - the non-parametric cousin."""
    yc, yt = d["post_c"], d["post_t"]
    xc = _impute(d["pre_c"], d["new_c"])
    xt = _impute(d["pre_t"], d["new_t"])
    trials, m = yc.shape
    est = np.zeros(trials)
    var = np.zeros(trials)
    for i in range(trials):
        edges = np.quantile(np.concatenate([xc[i], xt[i]]), np.linspace(0, 1, n_bins + 1)[1:-1])
        bc = np.searchsorted(edges, xc[i])
        bt = np.searchsorted(edges, xt[i])
        num = 0.0
        v = 0.0
        for b in range(n_bins):
            mc, mt = bc == b, bt == b
            if mc.sum() < 2 or mt.sum() < 2:
                continue
            w = (mc.sum() + mt.sum()) / (2.0 * m)
            num += w * (yt[i][mt].mean() - yc[i][mc].mean())
            v += w * w * (yt[i][mt].var(ddof=1) / mt.sum() + yc[i][mc].var(ddof=1) / mc.sum())
        est[i] = num
        var[i] = v
    return est, np.sqrt(var)


def adj_cuped_stratified(d: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """CUPED within the users who HAVE a pre-period, plain difference within
    those who do not, combined by their share.

    Three lines more than mean-imputation, and it is the difference between
    (1-f)*rho^2 and a variance INCREASE once f passes 0.5.
    """
    yc, yt = d["post_c"], d["post_t"]
    xc, xt = d["pre_c"], d["pre_t"]
    nc, nt = d["new_c"], d["new_t"]
    trials = yc.shape[0]
    est = np.zeros(trials)
    var = np.zeros(trials)
    for i in range(trials):
        rc, rt = ~nc[i], ~nt[i]
        n_ret = int(rc.sum() + rt.sum())
        n_new = int(nc[i].sum() + nt[i].sum())
        n_all = n_ret + n_new
        parts = []
        if rc.sum() > 2 and rt.sum() > 2:
            xr = np.concatenate([xc[i][rc] - xc[i][rc].mean(), xt[i][rt] - xt[i][rt].mean()])
            yr = np.concatenate([yc[i][rc] - yc[i][rc].mean(), yt[i][rt] - yt[i][rt].mean()])
            th = float((xr * yr).mean() / (xr * xr).mean()) if (xr * xr).mean() > 0 else 0.0
            grand = np.concatenate([xc[i][rc], xt[i][rt]]).mean()
            ac = yc[i][rc] - th * (xc[i][rc] - grand)
            at = yt[i][rt] - th * (xt[i][rt] - grand)
            parts.append((n_ret / n_all, at.mean() - ac.mean(),
                          at.var(ddof=1) / rt.sum() + ac.var(ddof=1) / rc.sum()))
        if nc[i].sum() > 2 and nt[i].sum() > 2:
            parts.append((n_new / n_all, yt[i][nt[i]].mean() - yc[i][nc[i]].mean(),
                          yt[i][nt[i]].var(ddof=1) / nt[i].sum()
                          + yc[i][nc[i]].var(ddof=1) / nc[i].sum()))
        w_tot = sum(w for w, _, _ in parts) or 1.0
        est[i] = sum(w / w_tot * e for w, e, _ in parts)
        var[i] = sum((w / w_tot) ** 2 * v for w, _, v in parts)
    return est, np.sqrt(var)


ADJUSTERS: Dict[str, Callable] = {
    "none": adj_none,
    "cuped": adj_cuped,
    "cuped_per_arm": adj_cuped_per_arm,
    "diff_in_diff": adj_diff_in_diff,
    "cuped_stratified": adj_cuped_stratified,
    "post_strat": adj_post_strat,
}


# --------------------------------------------------------------------------
# 3. Closed forms
# --------------------------------------------------------------------------


def theta_star(world: World) -> float:
    """Cov(Y, X) / Var(X) = rho * sd_post / sd_pre.

    Note what this is NOT: 1. Forcing theta to 1 is the "subtract each user's
    own pre-period value" instinct, and section 4 prices what it costs.
    """
    return world.rho * world.sd / world.sd_pre_eff


def variance_reduction(rho: float) -> float:
    """The entire CUPED result: Var(Y_cuped) / Var(Y) = 1 - rho^2."""
    return rho * rho


def variance_ratio_unit_theta(rho: float, sd_pre: float, sd_post: float) -> float:
    """Var(Y - X) / Var(Y) when theta is forced to 1.

    sd_post^2 + sd_pre^2 - 2 rho sd_pre sd_post, over sd_post^2. Greater than 1
    whenever sd_pre > 2 rho sd_post - which is how the intuitive adjustment
    makes the test slower rather than faster.
    """
    return (sd_post ** 2 + sd_pre ** 2 - 2 * rho * sd_pre * sd_post) / sd_post ** 2


def effective_rho_with_new_users(rho: float, new_share: float) -> float:
    """Correlation between the MEAN-IMPUTED covariate and the metric.

    Cov is unchanged for the returning fraction and zero for the new one, while
    Var(X) shrinks by the same factor, so corr = sqrt(1 - f) * rho. Every write-up
    stops here and concludes the reduction is (1 - f) * rho^2. It is not - see
    :func:`reduction_mean_impute`.
    """
    return float(np.sqrt(max(1.0 - new_share, 0.0)) * rho)


def reduction_mean_impute(rho: float, new_share: float) -> float:
    """Variance reduction actually delivered by mean-imputing the covariate.

    rho^2 * (2 - 1/(1 - f)).

    The per-user variance does fall by (1 - f) * rho^2, which is where the usual
    claim comes from. But the estimator is a difference of ARM MEANS, and once
    the covariate is imputed, the arm's covariate mean is the mean of the
    RETURNING users only - its variance is sigma_x^2 / (n(1-f)), not
    sigma_x^2 / n. Writing out
    Var(dY - theta dX) = 2 sigma_y^2/n [1 + rho^2/(1-f) - 2 rho^2]
    leaves the expression above.

    It is zero at f = 0.5 and negative beyond, for ANY rho: past half new users,
    mean-imputation makes the experiment need MORE traffic than no adjustment at
    all. :func:`adj_cuped_stratified` is the fix and recovers (1-f) * rho^2.
    """
    f = float(new_share)
    if f >= 1.0:
        return float("-inf")
    return float(rho * rho * (2.0 - 1.0 / (1.0 - f)))


def reduction_stratified(rho: float, new_share: float) -> float:
    """Variance reduction from treating "has a pre-period" as a stratum.

    (1 - f) * rho^2 - the number the naive version is usually credited with.
    """
    return float((1.0 - new_share) * rho * rho)


def impute_breakeven_share() -> float:
    """New-user share at which mean-imputation stops helping. Independent of rho."""
    return 0.5


def sample_size_multiplier(rho: float) -> float:
    """Fraction of the original sample size CUPED needs: 1 - rho^2."""
    return 1.0 - rho * rho


def rho_for_saving(saving: float) -> float:
    """The correlation required to save a given fraction of the sample."""
    return float(np.sqrt(saving))


# --------------------------------------------------------------------------
# 4. Scoring
# --------------------------------------------------------------------------


def score(est: np.ndarray, se: np.ndarray, true_effect: float, alpha: float = 0.05) -> Dict[str, float]:
    """Bias, spread, power/size and coverage for one adjuster on one world."""
    z = stats.norm.isf(alpha / 2.0)
    reject = np.abs(est) > z * se
    lo, hi = est - z * se, est + z * se
    return {
        "mean_est": float(est.mean()),
        "bias": float(est.mean() - true_effect),
        "sd_est": float(est.std(ddof=1)),
        "mean_se": float(se.mean()),
        "reject_rate": float(reject.mean()),
        "coverage": float(((lo <= true_effect) & (true_effect <= hi)).mean()),
    }


def measured_reduction(sd_adjusted: float, sd_unadjusted: float) -> float:
    """1 - Var(adjusted)/Var(unadjusted), on the same trials."""
    return 1.0 - (sd_adjusted / sd_unadjusted) ** 2


def reduction_with_mc(
    est: np.ndarray,
    base: np.ndarray,
    resamples: int = 300,
    seed: int = 0,
) -> Tuple[float, float]:
    """Measured variance reduction, plus its own Monte Carlo standard error.

    A reduction is a ratio of two sample variances over the same trials, so it
    carries sampling error of its own. Reporting the point estimate alone makes
    ordinary MC noise look like a broken derivation - or, worse, lets a broken
    derivation hide inside it.

    The error is bootstrapped over trials rather than estimated by splitting
    them into batches, because the bootstrap resamples the statistic that is
    actually reported. A twelve-batch split was tried first; its own standard
    error is around 20%, so on any single seed it lands either side of the
    bootstrap and cannot be compared to it - which is exactly why it is not
    good enough to adjudicate a 5% gap. The gap that prompted this was settled
    the other way: re-running the same world at T=40,000 put the measured
    variance ratio on the closed form to four decimals (2.0444 and 2.0500
    across two seeds against 2.0500), so the T=6,000 discrepancy was noise.
    """
    t = min(len(est), len(base))
    est, base = np.asarray(est[:t]), np.asarray(base[:t])
    point = measured_reduction(float(est.std(ddof=1)), float(base.std(ddof=1)))
    if t < 200:
        return point, float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, t, size=(resamples, t))
    e_s = est[idx].std(axis=1, ddof=1)
    b_s = base[idx].std(axis=1, ddof=1)
    boot = 1.0 - (e_s / b_s) ** 2
    return point, float(boot.std(ddof=1))
