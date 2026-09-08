"""Segment scans and CATE trees on a randomised trial, and the arithmetic behind both.

Three things live here and nothing else:

* generators for four worlds - a homogeneous effect, one genuinely harmed segment,
  heterogeneity along a continuous covariate, and a segment defined AFTER assignment;
* the segment scan (difference in means per group, Welch SE, Bonferroni and
  Benjamini-Hochberg adjustment) that every experiment write-up ends with;
* a transformed-outcome CATE tree, fittable either in-sample or honestly (structure on
  one half, leaf effects on the other), because the difference between those two is the
  whole point of this build.

Nothing here reads results.json. Every number in the evidence run is computed live.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import integrate, stats

# --------------------------------------------------------------------------- worlds


def trial_partition(
    rng: np.random.Generator,
    n: int = 20000,
    k: int = 20,
    ate: float = 0.05,
    delta: float = 0.0,
    harmed: int = 0,
    sigma: float = 1.0,
    p_treat: float = 0.5,
) -> Dict[str, np.ndarray]:
    """K equal-sized pre-defined segments. `delta` is subtracted from segment `harmed`.

    delta=0 is the homogeneous world: every segment has the same true effect `ate`.
    """
    seg = rng.integers(0, k, size=n)
    w = (rng.random(n) < p_treat).astype(np.int8)
    tau = np.full(n, ate)
    if delta:
        tau[seg == harmed] = ate - delta
    y = rng.normal(0.0, sigma, size=n) + w * tau
    return {"seg": seg, "w": w, "y": y, "tau": tau, "k": np.int64(k)}


def trial_continuous(
    rng: np.random.Generator,
    n: int = 8000,
    ate: float = 0.05,
    beta: float = 0.20,
    k: int = 20,
    n_cov: int = 5,
    sigma: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Effect moves with x0 only: tau(x) = ate + beta * x0, so it changes SIGN.

    The `seg` column is a real categorical that is INDEPENDENT of the effect - the thing
    an analyst would actually slice on. A scan over it is looking in the wrong place.
    """
    x = rng.normal(size=(n, n_cov))
    seg = rng.integers(0, k, size=n)
    w = (rng.random(n) < 0.5).astype(np.int8)
    tau = ate + beta * x[:, 0]
    y = 0.5 * x[:, 1] + rng.normal(0.0, sigma, size=n) + w * tau
    return {"x": x, "seg": seg, "w": w, "y": y, "tau": tau, "k": np.int64(k)}


def trial_slices(
    rng: np.random.Generator,
    n: int = 20000,
    n_slices: int = 20,
    rho: float = 0.0,
    ate: float = 0.05,
    sigma: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Overlapping binary slices ("mobile", "new", "EU", ...), each about half the users.

    Correlation between slice memberships is `rho` through one shared latent factor, so
    "we looked at 20 cuts" and "we ran 20 independent tests" can be told apart.
    """
    u = rng.normal(size=(n, 1))
    e = rng.normal(size=(n, n_slices))
    z = rho * u + np.sqrt(max(0.0, 1.0 - rho * rho)) * e
    slices = z > 0.0
    w = (rng.random(n) < 0.5).astype(np.int8)
    y = rng.normal(0.0, sigma, size=n) + w * ate
    return {"slices": slices, "w": w, "y": y, "ate": np.float64(ate)}


def trial_post_gate(
    rng: np.random.Generator,
    n: int = 20000,
    ate: float = 0.05,
    gate_effect: float = 0.30,
    sigma: float = 1.0,
) -> Dict[str, np.ndarray]:
    """A segment measured AFTER assignment: "engaged users".

    Treatment raises the chance of being engaged (`gate_effect`), and engaged users have a
    higher outcome for reasons unrelated to treatment. The true effect is `ate` for
    everyone; there is no heterogeneity anywhere in this world.
    """
    latent = rng.normal(size=n)
    w = (rng.random(n) < 0.5).astype(np.int8)
    engaged = latent + gate_effect * w > 0.0
    y = 0.8 * latent + rng.normal(0.0, sigma, size=n) + w * ate
    return {"engaged": engaged, "w": w, "y": y, "ate": np.float64(ate)}


# --------------------------------------------------------------------------- estimators


def diff_means(y: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Difference in means with a Welch standard error. (nan, inf) if an arm is empty."""
    y1, y0 = y[w == 1], y[w == 0]
    if y1.size < 2 or y0.size < 2:
        return float("nan"), float("inf")
    est = float(y1.mean() - y0.mean())
    se = float(np.sqrt(y1.var(ddof=1) / y1.size + y0.var(ddof=1) / y0.size))
    return est, se


def scan(y: np.ndarray, w: np.ndarray, groups: np.ndarray) -> Dict[str, np.ndarray]:
    """Run diff_means inside every group. `groups` is (n, G) boolean, columns may overlap."""
    g = groups.shape[1]
    est = np.empty(g)
    se = np.empty(g)
    for j in range(g):
        m = groups[:, j]
        est[j], se[j] = diff_means(y[m], w[m])
    with np.errstate(invalid="ignore", divide="ignore"):
        t = est / se
    p = 2.0 * stats.norm.sf(np.abs(t))
    return {"est": est, "se": se, "t": t, "p": np.where(np.isfinite(p), p, 1.0)}


def partition_matrix(seg: np.ndarray, k: int) -> np.ndarray:
    return np.stack([seg == j for j in range(k)], axis=1)


def bonferroni(p: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, p * p.size)


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (monotone, same order as raw)."""
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(1, m + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(1.0, ranked)
    return out


def harm_flag(res: Dict[str, np.ndarray], alpha: float = 0.05, method: str = "raw") -> bool:
    """Did the scan report at least one segment as significantly HARMED?"""
    p = {"raw": res["p"], "bonferroni": bonferroni(res["p"]), "bh": bh(res["p"])}[method]
    return bool(np.any((res["est"] < 0.0) & (p < alpha)))


def any_flag(res: Dict[str, np.ndarray], alpha: float = 0.05, method: str = "raw") -> bool:
    p = {"raw": res["p"], "bonferroni": bonferroni(res["p"]), "bh": bh(res["p"])}[method]
    return bool(np.any(p < alpha))


# --------------------------------------------------------------------------- arithmetic


def fwer_independent(k: int, alpha: float = 0.05) -> float:
    """P(at least one of k independent tests fires) - arithmetic, no data in it."""
    return 1.0 - (1.0 - alpha) ** k


def expected_extreme_z(k: int, which: str = "min") -> float:
    """E[min] (or max) of k iid standard normals, by quadrature - the winner's curse size."""

    def f(z: float) -> float:
        return z * k * stats.norm.pdf(z) * stats.norm.cdf(z) ** (k - 1)

    val, _ = integrate.quad(f, -12.0, 12.0, limit=400)
    return float(val if which == "max" else -val)


def k_effective(alarm_rate: float, alpha: float = 0.05) -> float:
    """Back out how many INDEPENDENT tests a correlated family behaved like."""
    alarm_rate = min(max(alarm_rate, 1e-12), 1.0 - 1e-12)
    return float(np.log1p(-alarm_rate) / np.log1p(-alpha))


def mde(n_per_arm: float, sigma: float = 1.0, alpha: float = 0.05, power: float = 0.8) -> float:
    z_a = stats.norm.isf(alpha / 2.0)
    z_b = stats.norm.isf(1.0 - power)
    return float((z_a + z_b) * sigma * np.sqrt(2.0 / n_per_arm))


def segment_se(n: int, k: int, sigma: float = 1.0, p_treat: float = 0.5) -> float:
    """SE of a diff-in-means inside one of k equal segments of a trial of size n."""
    per_arm = n / k * min(p_treat, 1.0 - p_treat)
    return float(sigma * np.sqrt(2.0 / per_arm))


def harm_alarm_closed_form(k: int, n: int, ate: float, sigma: float = 1.0, alpha: float = 0.05) -> float:
    """P(a scan of k equal segments calls at least one HARMED) when the true effect is `ate`.

    The version everybody quotes, 1-(1-alpha/2)^k, is this expression at ate=0. A real
    positive effect shifts every segment's sampling distribution and shrinks the alarm rate,
    so how exposed a scan is to inventing harm depends on how strong the launch was.
    """
    se = segment_se(n, k, sigma)
    per_test = float(stats.norm.cdf(-stats.norm.isf(alpha / 2.0) - ate / se))
    return float(1.0 - (1.0 - per_test) ** k)


def consistency_gap(y: np.ndarray, w: np.ndarray, groups: np.ndarray) -> float:
    """(share-weighted mean of within-segment effects) - (overall effect), in overall SEs.

    For a partition fixed BEFORE assignment this is an algebraic near-identity and sits at
    zero with almost no sampling noise. Conditioning on anything measured after assignment
    breaks it. It needs no counterfactual and nobody runs it.
    """
    overall, se = diff_means(y, w)
    res = scan(y, w, groups)
    share = groups.sum(axis=0) / groups.shape[0]
    ok = np.isfinite(res["est"])
    inner = float(np.sum(share[ok] * res["est"][ok]) / np.sum(share[ok]))
    return float((inner - overall) / se)


# --------------------------------------------------------------------------- CATE tree


class Node:
    __slots__ = ("feat", "thr", "left", "right", "leaf_id")

    def __init__(self) -> None:
        self.feat: int = -1
        self.thr: float = 0.0
        self.left: Optional["Node"] = None
        self.right: Optional["Node"] = None
        self.leaf_id: int = -1


def transformed_outcome(y: np.ndarray, w: np.ndarray, p_treat: float = 0.5) -> np.ndarray:
    """Y* with E[Y*|X] = tau(X). The tree splits on this; it is an unbiased effect signal."""
    return y * (w - p_treat) / (p_treat * (1.0 - p_treat))


def _best_split(x: np.ndarray, ystar: np.ndarray, min_leaf: int, n_grid: int) -> Tuple[float, int, float]:
    """Greedy variance-reduction split on Y*. Returns (gain, feature, threshold)."""
    best = (0.0, -1, 0.0)
    n = ystar.size
    if n < 2 * min_leaf:
        return best
    total = ystar.sum()
    for j in range(x.shape[1]):
        col = x[:, j]
        qs = np.quantile(col, np.linspace(0.05, 0.95, n_grid))
        for thr in np.unique(qs):
            m = col <= thr
            n_l = int(m.sum())
            if n_l < min_leaf or n - n_l < min_leaf:
                continue
            s_l = ystar[m].sum()
            gain = s_l * s_l / n_l + (total - s_l) ** 2 / (n - n_l) - total * total / n
            if gain > best[0]:
                best = (float(gain), j, float(thr))
    return best


def causal_tree(
    x: np.ndarray,
    w: np.ndarray,
    y: np.ndarray,
    depth: int = 2,
    min_leaf: int = 200,
    n_grid: int = 24,
    p_treat: float = 0.5,
) -> Node:
    """Fit tree STRUCTURE by maximising split gain on the transformed outcome."""
    ystar = transformed_outcome(y, w, p_treat)
    counter = [0]

    def grow(idx: np.ndarray, d: int) -> Node:
        node = Node()
        gain, feat, thr = (0.0, -1, 0.0) if d == 0 else _best_split(x[idx], ystar[idx], min_leaf, n_grid)
        if feat < 0 or gain <= 0.0:
            node.leaf_id = counter[0]
            counter[0] += 1
            return node
        node.feat, node.thr = feat, thr
        m = x[idx, feat] <= thr
        node.left = grow(idx[m], d - 1)
        node.right = grow(idx[~m], d - 1)
        return node

    return grow(np.arange(x.shape[0]), depth)


def leaf_of(tree: Node, x: np.ndarray) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=np.int64)
    for i in range(x.shape[0]):
        node = tree
        while node.leaf_id < 0:
            node = node.left if x[i, node.feat] <= node.thr else node.right
        out[i] = node.leaf_id
    return out


def n_leaves(tree: Node) -> int:
    if tree.leaf_id >= 0:
        return 1
    return n_leaves(tree.left) + n_leaves(tree.right)


def leaf_effects(tree: Node, x: np.ndarray, w: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Estimate each leaf's effect from WHATEVER data is passed in.

    Pass the fitting data and you get the in-sample number an analyst would quote; pass a
    held-out half and you get the honest one. The gap between them is section 6.
    """
    lid = leaf_of(tree, x)
    ids = np.arange(n_leaves(tree))
    groups = np.stack([lid == j for j in ids], axis=1)
    res = scan(y, w, groups)
    res["n"] = groups.sum(axis=0).astype(float)
    return res


def tau_hat_rows(tree: Node, x_fit: np.ndarray, effects: Dict[str, np.ndarray], x_new: np.ndarray) -> np.ndarray:
    """Per-row CATE prediction: the leaf effect of the leaf a row lands in."""
    del x_fit
    return effects["est"][leaf_of(tree, x_new)]


def honest_fit(
    rng: np.random.Generator,
    x: np.ndarray,
    w: np.ndarray,
    y: np.ndarray,
    depth: int = 2,
    min_leaf: int = 100,
) -> Dict[str, object]:
    """Split the sample, fit structure on half A, estimate leaf effects on half B."""
    n = x.shape[0]
    idx = rng.permutation(n)
    a, b = idx[: n // 2], idx[n // 2 :]
    tree = causal_tree(x[a], w[a], y[a], depth=depth, min_leaf=min_leaf)
    return {
        "tree": tree,
        "in_sample": leaf_effects(tree, x[a], w[a], y[a]),
        "honest": leaf_effects(tree, x[b], w[b], y[b]),
        "a": a,
        "b": b,
    }


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def ci_covers(est: float, se: float, truth: float, z: float = 1.959963985) -> bool:
    return bool(abs(est - truth) <= z * se)


def summarise(rows: List[Dict[str, object]], keys: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array([float(r[k]) for r in rows if np.isfinite(float(r[k]))])
        out[k] = float(vals.mean())
        out[k + "_mcse"] = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float("nan")
    return out
