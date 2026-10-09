"""Exact latency of a RAG chain on a 1 ms grid, and what per-hop dashboards say about it.

A request walks five hops in sequence: gateway, query embedding, vector search (fanned out to SHARDS shards in
parallel, the step waits for the slowest), reranker, LLM generation. Each hop is a declared lognormal body plus
the one tail mechanism it really has (cold start, GC pause, hang-then-timeout-retry). Every distribution is a
probability mass function on integer milliseconds, so

    sequential hops   -> convolution of pmfs
    parallel fan-out  -> CDF ** SHARDS               (max of iid)
    hedged request    -> survival product            (min of the call and a delayed copy)
    timeout + retry   -> mixture of the body and (timeout + a fresh body)

and every percentile, mean and tail attribution below is exact on the grid. `simulate` draws the same model
continuously and rounds to the nearest ms, which is the grid's own convention, so a Monte Carlo check of the
exact numbers differs from them by sampling noise only.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

SEED = 194
MC_REPS = 200_000
GRID_MS = 8_000
SHARDS = 8
SLO_MS = 1_500
PCTS = (0.50, 0.95, 0.99)
HOPS = ("gateway", "embed", "search", "rerank", "llm")

BASE: Dict[str, float] = {
    "gateway_median": 8.0, "gateway_sigma": 0.30,
    "embed_median": 25.0, "embed_sigma": 0.25, "embed_cold_p": 0.02, "embed_cold_ms": 300.0,
    "shard_median": 30.0, "shard_sigma": 0.35, "shard_gc_p": 0.008, "shard_gc_ms": 250.0, "hedge_after_ms": 0.0,
    "rerank_median": 60.0, "rerank_sigma": 0.30, "rerank_hang_p": 0.03, "rerank_timeout_ms": 400.0,
    "llm_median": 900.0, "llm_sigma": 0.12,
}

# Four fixes a team could fund this quarter, each a change to ONE declared parameter, plus the three config ones together.
FIXES: Dict[str, Dict[str, float]] = {
    "smaller LLM (-15% median)": {"llm_median": 765.0},
    "warm embedding pool": {"embed_cold_p": 0.0},
    "hedge shard calls at 80 ms": {"hedge_after_ms": 80.0},
    "rerank timeout 400 -> 120 ms": {"rerank_timeout_ms": 120.0},
    "all three config fixes": {"embed_cold_p": 0.0, "hedge_after_ms": 80.0, "rerank_timeout_ms": 120.0},
}


def _erf(x: np.ndarray) -> np.ndarray:
    return np.frompyfunc(math.erf, 1, 1)(x).astype(float)


def lognormal_pmf(median: float, sigma: float, n: int = GRID_MS) -> np.ndarray:
    """Mass on integer ms k = P(k - 0.5 < X <= k + 0.5): the grid rounds to the nearest millisecond."""
    edges = np.arange(n + 1) - 0.5
    pos = np.clip(edges, 1e-12, None)
    cdf = 0.5 * (1.0 + _erf(np.log(pos / median) / (sigma * math.sqrt(2.0))))
    cdf[edges <= 0] = 0.0
    p = np.diff(cdf)
    p[-1] += 1.0 - cdf[-1]
    return p


def shift(p: np.ndarray, ms: float) -> np.ndarray:
    k = int(round(ms))
    if k <= 0:
        return p.copy()
    out = np.zeros_like(p)
    out[k:] = p[:-k]
    out[-1] += p[-k:].sum()
    return out


def conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    full = np.convolve(a, b)
    out = full[: len(a)].copy()
    out[-1] += full[len(a):].sum()
    return out


def cdf_of(p: np.ndarray) -> np.ndarray:
    return np.minimum(np.cumsum(p), 1.0)


def from_cdf(c: np.ndarray) -> np.ndarray:
    return np.diff(np.concatenate([[0.0], c]))


def shard_pmf(prm: Dict[str, float]) -> np.ndarray:
    """One shard: lognormal body, plus a GC pause with probability shard_gc_p. Hedged if hedge_after_ms > 0."""
    body = lognormal_pmf(prm["shard_median"], prm["shard_sigma"])
    one = (1 - prm["shard_gc_p"]) * body + prm["shard_gc_p"] * shift(body, prm["shard_gc_ms"])
    h = prm["hedge_after_ms"]
    if h > 0:
        surv = 1.0 - cdf_of(one)
        surv_copy = 1.0 - cdf_of(shift(one, h))
        one = from_cdf(1.0 - surv * surv_copy)
    return one


def hop_pmfs(prm: Dict[str, float]) -> Dict[str, np.ndarray]:
    g = lognormal_pmf(prm["gateway_median"], prm["gateway_sigma"])
    e_body = lognormal_pmf(prm["embed_median"], prm["embed_sigma"])
    e = (1 - prm["embed_cold_p"]) * e_body + prm["embed_cold_p"] * shift(e_body, prm["embed_cold_ms"])
    s = from_cdf(cdf_of(shard_pmf(prm)) ** SHARDS)
    r_body = lognormal_pmf(prm["rerank_median"], prm["rerank_sigma"])
    tau = int(round(prm["rerank_timeout_ms"]))
    h = prm["rerank_hang_p"]
    kept = r_body.copy()
    kept[tau + 1:] = 0.0
    p_retry = h + (1 - h) * r_body[tau + 1:].sum()
    r = (1 - h) * kept + p_retry * shift(r_body, tau)
    llm = lognormal_pmf(prm["llm_median"], prm["llm_sigma"])
    return {"gateway": g, "embed": e, "search": s, "rerank": r, "llm": llm}


def chain(pmfs: Dict[str, np.ndarray], skip: Optional[str] = None) -> np.ndarray:
    out: Optional[np.ndarray] = None
    for k in HOPS:
        if k == skip:
            continue
        out = pmfs[k] if out is None else conv(out, pmfs[k])
    assert out is not None
    return out


def percentile(p: np.ndarray, q: float) -> int:
    return int(np.searchsorted(cdf_of(p), q - 1e-12))


def mean(p: np.ndarray) -> float:
    return float(np.dot(np.arange(len(p)), p))


def summary(p: np.ndarray) -> Dict[str, float]:
    out = {f"p{int(q * 100)}": percentile(p, q) for q in PCTS}
    out["mean"] = mean(p)
    return out


def tail_attribution(pmfs: Dict[str, np.ndarray], t: int) -> Dict[str, float]:
    """E[X_i | T > t] - E[X_i] for each hop: how much MORE of hop i a slow request carries. Sums exactly to
    E[T | T > t] - E[T], because sum_i X_i = T.

    E[X_i 1{T>t}] = sum_x x p_i(x) P(T_-i > t - x), with T_-i the chain without hop i.
    """
    total = chain(pmfs)
    p_tail = float(total[t + 1:].sum())
    ks = np.arange(GRID_MS)
    out: Dict[str, float] = {}
    for h in HOPS:
        others_cdf = cdf_of(chain(pmfs, skip=h))
        need = t - ks
        surv = np.where(need < 0, 1.0, 1.0 - others_cdf[np.clip(need, 0, GRID_MS - 1)])
        out[h] = float(np.dot(ks * pmfs[h], surv)) / p_tail - mean(pmfs[h])
    return out


def study(prm: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    prm = dict(BASE if prm is None else prm)
    pm = hop_pmfs(prm)
    total = chain(pm)
    hops = {h: summary(pm[h]) for h in HOPS}
    e2e = summary(total)
    shard = summary(shard_pmf(prm))
    q99 = int(e2e["p99"])
    excess = tail_attribution(pm, q99)
    mean_share = {h: hops[h]["mean"] / e2e["mean"] for h in HOPS}
    excess_total = sum(excess.values())
    return {
        "hops": hops, "e2e": e2e, "shard": shard,
        "sum_of_hop_p99": sum(hops[h]["p99"] for h in HOPS),
        "sum_of_hop_p50": sum(hops[h]["p50"] for h in HOPS),
        "p_over_slo": float(total[SLO_MS + 1:].sum()),
        "p_any_shard_gc": 1 - (1 - prm["shard_gc_p"]) ** SHARDS,
        "mean_share": mean_share,
        "tail_excess_ms": excess,
        "tail_share": {h: excess[h] / excess_total for h in HOPS},
        "p_tail": float(total[q99 + 1:].sum()),
    }


def fix_table(base: Optional[Dict[str, float]] = None) -> List[Dict[str, object]]:
    base = dict(BASE if base is None else base)
    ref = study(base)
    rows = []
    for name, change in FIXES.items():
        s = study({**base, **change})
        rows.append({
            "fix": name,
            "d_mean": s["e2e"]["mean"] - ref["e2e"]["mean"],
            "d_p50": s["e2e"]["p50"] - ref["e2e"]["p50"],
            "d_p99": s["e2e"]["p99"] - ref["e2e"]["p99"],
            "p_over_slo": s["p_over_slo"],
            "p99": s["e2e"]["p99"],
        })
    return rows


def simulate(prm: Optional[Dict[str, float]] = None, n: int = MC_REPS, seed: int = SEED) -> Dict[str, np.ndarray]:
    """The same model drawn continuously, rounded to the nearest ms at the end of each hop."""
    prm = dict(BASE if prm is None else prm)
    rng = np.random.default_rng(seed)

    def ln(med: float, sig: float, size: Tuple[int, ...]) -> np.ndarray:
        return med * np.exp(sig * rng.standard_normal(size))

    g = ln(prm["gateway_median"], prm["gateway_sigma"], (n,))
    e = ln(prm["embed_median"], prm["embed_sigma"], (n,)) + prm["embed_cold_ms"] * (rng.random(n) < prm["embed_cold_p"])
    sh = ln(prm["shard_median"], prm["shard_sigma"], (n, SHARDS)) + prm["shard_gc_ms"] * (rng.random((n, SHARDS)) < prm["shard_gc_p"])
    sh = np.rint(sh)
    if prm["hedge_after_ms"] > 0:
        copy = ln(prm["shard_median"], prm["shard_sigma"], (n, SHARDS)) + prm["shard_gc_ms"] * (rng.random((n, SHARDS)) < prm["shard_gc_p"])
        sh = np.minimum(sh, np.rint(copy) + round(prm["hedge_after_ms"]))
    s = sh.max(axis=1)
    tau = round(prm["rerank_timeout_ms"])
    r1 = np.rint(ln(prm["rerank_median"], prm["rerank_sigma"], (n,)))
    hang = rng.random(n) < prm["rerank_hang_p"]
    r2 = np.rint(ln(prm["rerank_median"], prm["rerank_sigma"], (n,)))
    r = np.where(hang | (r1 > tau), tau + r2, r1)
    llm = ln(prm["llm_median"], prm["llm_sigma"], (n,))
    hops = {"gateway": np.rint(g), "embed": np.rint(e), "search": s, "rerank": r, "llm": np.rint(llm)}
    hops["total"] = sum(hops[h] for h in HOPS)
    return hops


def wilson(k: int, n: int, z: float = 2.576) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - w, c + w)


def calibrate(n: int = MC_REPS, seed: int = SEED) -> List[Dict[str, object]]:
    """Exact P(T > t) against Monte Carlo, at the thresholds the study quotes, for the base chain and each fix."""
    rows = []
    for label, prm in [("base", dict(BASE))] + [(k, {**BASE, **v}) for k, v in FIXES.items()]:
        total = chain(hop_pmfs(prm))
        sim = simulate(prm, n, seed)["total"]
        for t in (summary(total)["p95"], summary(total)["p99"], SLO_MS):
            exact = float(total[int(t) + 1:].sum())
            k = int((sim > t).sum())
            lo, hi = wilson(k, n)
            rows.append({"chain": label, "t": int(t), "exact": exact, "mc": k / n, "inside": lo <= exact <= hi})
    return rows


def parse_params(text: str) -> Dict[str, float]:
    """`key=value, key=value` overrides on BASE. Unknown keys and negative values are refused, not ignored."""
    prm = dict(BASE)
    for part in [x.strip() for x in text.split(",") if x.strip()]:
        if "=" not in part:
            raise ValueError(f"'{part}' is not key=value")
        k, v = (s.strip() for s in part.split("=", 1))
        if k not in BASE:
            raise ValueError(f"unknown parameter '{k}'")
        val = float(v)
        if val < 0 or (k.endswith("_p") and val > 1):
            raise ValueError(f"{k}={v} is out of range")
        prm[k] = val
    return prm
