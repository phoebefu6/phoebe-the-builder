"""ANOVA says something differs. Which pair? Five answers on the same data, and what each one costs.

A one-way ANOVA across `k` groups ends with a significant F and a question it cannot answer: WHICH
groups differ. There are `m = k(k-1)/2` pairs, and the procedures people use to hunt among them are
reported as though they were interchangeable:

  * none        - every pair at 0.05, no adjustment (what a spreadsheet of t-tests does)
  * lsd         - Fisher's protected LSD: the same unadjusted pairs, but only after a significant F
  * bonferroni  - every pair at 0.05 / m
  * holm        - step-down Bonferroni
  * tukey       - Tukey's HSD, the studentized-range critical value

Every pairwise statistic uses the pooled ANOVA error (MSE, df = k(n-1)), which is the textbook
post-hoc setting. Data are normal with equal n and equal variance - the configuration where Tukey's
HSD is EXACT - so no finding here can be blamed on a violated assumption.

What separates this from the built `stat-test-advisor` (which test to use) and `t-test-variants`
(two groups): nothing in the catalog measures the FAMILY - the error rate across all m pairs, how it
grows with k, and what controlling it costs in power.

Note on duplication: the Wilson interval and verdict band also appear in sibling builds. Deliberate
- every build must run standalone from a bare Colab link. One copy WITHIN this build; the chart,
the app and the notebook read this module's numbers rather than recomputing them.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

ALPHA = 0.05
PROCS = ("none", "lsd", "bonferroni", "holm", "tukey")
LABEL = {"none": "no adjustment", "lsd": "protected LSD", "bonferroni": "Bonferroni",
         "holm": "Holm", "tukey": "Tukey HSD"}

# A verdict is a 99% Wilson interval at the cell's own replicate count. INFLATED means the whole
# interval clears 0.055, CONSERVATIVE means it sits wholly below 0.045 (the Day 175 rule: a
# point estimate compared to a band flags the harness's own noise).
BAND_LO, BAND_HI = 0.045, 0.055
Z99 = 2.5758293035489

N_PER_GROUP = 20
REPS = 20_000
KS = (3, 4, 5, 6, 8, 10)

# Seeds are INDICES into these lists, never hash() of a design (Day 178: string hashing is salted
# per process, so hash-derived seeds made every run produce different numbers).
NULL_DESIGNS: List[Tuple[int, str, float]] = [(k, "null", 0.0) for k in KS]
PARTIAL_DESIGNS: List[Tuple[int, str, float]] = [(k, "one-out", 3.0) for k in KS]
POWER_DESIGNS: List[Tuple[int, str, float]] = (
    [(k, "one-out", 0.9) for k in (3, 5, 8)] + [(k, "spread", 1.0) for k in (3, 5, 8, 10)]
)
SEED_BASE = {"null": 1_000, "partial": 2_000, "power": 3_000}


def wilson(hits: int, n: int, z: float = Z99) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    den = 1.0 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(mid - half), float(mid + half))


def verdict(hits: int, n: int) -> str:
    lo, hi = wilson(hits, n)
    if lo > BAND_HI:
        return "INFLATED"
    if hi < BAND_LO:
        return "CONSERVATIVE"
    return "ok"


def means_for(k: int, shape: str, effect: float) -> np.ndarray:
    """True group means in SD units. `one-out`: one group shifted by `effect`, the rest at 0.
    `spread`: means evenly spaced from 0 to `effect` - a ladder, where every pair differs."""
    if shape == "null":
        return np.zeros(k)
    if shape == "one-out":
        mu = np.zeros(k)
        mu[0] = effect
        return mu
    if shape == "spread":
        return np.linspace(0.0, effect, k)
    raise ValueError(shape)


def pairs(k: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


# --------------------------------------------------------------------------- 1. the statistics

def summarise(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """data (reps, k, n) -> group means (reps, k), pooled MSE (reps,), error df."""
    reps, k, n = data.shape
    m = data.mean(axis=2)
    mse = ((data - m[:, :, None]) ** 2).sum(axis=(1, 2)) / (k * (n - 1))
    return m, mse, k * (n - 1)


def f_pvalue(m: np.ndarray, mse: np.ndarray, n: int) -> np.ndarray:
    k = m.shape[1]
    msb = n * ((m - m.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) / (k - 1)
    return stats.f.sf(msb / mse, k - 1, k * (n - 1))


def pair_t(m: np.ndarray, mse: np.ndarray, n: int) -> np.ndarray:
    """Pooled-MSE t statistic for every pair, shape (reps, m)."""
    idx = pairs(m.shape[1])
    diff = np.stack([m[:, i] - m[:, j] for i, j in idx], axis=1)
    return diff / np.sqrt(2.0 * mse / n)[:, None]


_QCRIT: Dict[Tuple[int, int, float], float] = {}


def q_crit(k: int, df: int, alpha: float = ALPHA) -> float:
    """Studentized-range critical value. Slow in scipy, so computed once per (k, df)."""
    key = (k, df, alpha)
    if key not in _QCRIT:
        _QCRIT[key] = float(stats.studentized_range.ppf(1.0 - alpha, k, df))
    return _QCRIT[key]


def holm_reject(p: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Step-down Holm, vectorised over rows. Reject the smallest p while p_(r) <= alpha/(m-r)."""
    m = p.shape[1]
    order = np.argsort(p, axis=1)
    ps = np.take_along_axis(p, order, axis=1)
    passed = np.cumprod(ps <= alpha / (m - np.arange(m)), axis=1).astype(bool)
    out = np.zeros_like(passed)
    np.put_along_axis(out, order, passed, axis=1)
    return out


def decisions(data: np.ndarray, alpha: float = ALPHA) -> Dict[str, np.ndarray]:
    """Every procedure's pairwise rejections, each (reps, m) bool, plus the omnibus F p-value."""
    m, mse, df = summarise(data)
    n = data.shape[2]
    k = m.shape[1]
    t = pair_t(m, mse, n)
    p = 2.0 * stats.t.sf(np.abs(t), df)
    pf = f_pvalue(m, mse, n)
    npairs = p.shape[1]
    return {
        "F": pf,
        "none": p < alpha,
        "lsd": (pf < alpha)[:, None] & (p < alpha),
        "bonferroni": p < alpha / npairs,
        "holm": holm_reject(p, alpha),
        # |t| * sqrt(2) is the studentized range statistic for one pair with equal n
        "tukey": np.abs(t) * np.sqrt(2.0) > q_crit(k, df, alpha),
    }


def simulate(k: int, shape: str, effect: float, seed: int, reps: int = REPS,
             n: int = N_PER_GROUP) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((reps, k, n)) + means_for(k, shape, effect)[None, :, None]


# --------------------------------------------------------------------------- 2. the measurements

def run_design(k: int, shape: str, effect: float, seed: int, reps: int = REPS,
               n: int = N_PER_GROUP) -> Dict[str, object]:
    mu = means_for(k, shape, effect)
    idx = pairs(k)
    null_pairs = np.array([mu[i] == mu[j] for i, j in idx])
    dec = decisions(simulate(k, shape, effect, seed, reps, n))
    f_sig = dec["F"] < ALPHA
    row: Dict[str, object] = {"k": k, "shape": shape, "effect": effect, "reps": reps, "seed": seed,
                              "pairs": len(idx), "null_pairs": int(null_pairs.sum()),
                              "F_sig": float(f_sig.mean())}
    if (~null_pairs).any():
        # Tukey vs Holm per-pair power as a PAIRED difference on the same replicates, so the gap is
        # read against its own noise: material only if it clears a 99% interval (the Day 176 rule
        # that a win/loss tally without a materiality threshold counts coin flips).
        gap = dec["tukey"][:, ~null_pairs].mean(axis=1) - dec["holm"][:, ~null_pairs].mean(axis=1)
        half = Z99 * gap.std(ddof=1) / np.sqrt(reps)
        row["tukey_minus_holm"] = float(gap.mean())
        row["tukey_minus_holm_half"] = float(half)
        row["tukey_vs_holm"] = ("tukey" if gap.mean() > half else
                                "holm" if gap.mean() < -half else "tie")
    for proc in PROCS:
        d = dec[proc]
        if null_pairs.any():
            hits = int(d[:, null_pairs].any(axis=1).sum())
            row[f"{proc}_fwer"] = hits / reps
            row[f"{proc}_fwer_ci"] = wilson(hits, reps)
            row[f"{proc}_fwer_verdict"] = verdict(hits, reps)
        if (~null_pairs).any():
            row[f"{proc}_any"] = float(d[:, ~null_pairs].any(axis=1).mean())
            row[f"{proc}_per_pair"] = float(d[:, ~null_pairs].mean())
            row[f"{proc}_all"] = float(d[:, ~null_pairs].all(axis=1).mean())
        found = d.any(axis=1)
        # "ANOVA says something differs, but this procedure names no pair" - and the reverse
        row[f"{proc}_F_sig_no_pair"] = float((f_sig & ~found).mean())
        row[f"{proc}_pair_no_F"] = float((~f_sig & found).mean())
        row[f"{proc}_no_pair_given_F"] = float((f_sig & ~found).sum() / max(int(f_sig.sum()), 1))
    return row


def null_table(reps: int = REPS) -> List[Dict[str, object]]:
    return [run_design(k, s, e, SEED_BASE["null"] + i, reps) for i, (k, s, e) in enumerate(NULL_DESIGNS)]


def partial_table(reps: int = REPS) -> List[Dict[str, object]]:
    return [run_design(k, s, e, SEED_BASE["partial"] + i, reps)
            for i, (k, s, e) in enumerate(PARTIAL_DESIGNS)]


def power_table(partial: Sequence[Dict[str, object]], reps: int = REPS) -> List[Dict[str, object]]:
    """Power, gated: a procedure is COMPARABLE at k only if its FWER on the partial-null design at
    the same k is not INFLATED. An inflated procedure 'finds more' because it rejects more of
    everything, so its power lead is the inflation restated (the Day 176 rule)."""
    inflated = {(r["k"], p) for r in partial for p in PROCS if r[f"{p}_fwer_verdict"] == "INFLATED"}
    rows = []
    for i, (k, s, e) in enumerate(POWER_DESIGNS):
        r = run_design(k, s, e, SEED_BASE["power"] + i, reps)
        for p in PROCS:
            r[f"{p}_comparable"] = (k, p) not in inflated
        rows.append(r)
    return rows


def identity_checks(reps: int = 20_000, seed: int = 99) -> Dict[str, int]:
    """Logical statements that must hold on EVERY replicate, counted rather than assumed.

    1. Holm rejects every pair Bonferroni rejects (its first step IS Bonferroni).
    2. Under the complete null, Holm rejects anything iff Bonferroni does - so their FWER is
       identical there by construction, and Holm's advantage can only show up as power.
    3. Protected LSD never rejects a pair without a significant F (the protection, by definition).
    """
    out = {"holm_misses_bonferroni_pair": 0, "holm_vs_bonferroni_any_differs_under_null": 0,
           "lsd_without_F": 0, "replicates": 0}
    for i, k in enumerate(KS):
        dec = decisions(simulate(k, "null", 0.0, seed + i, reps))
        out["holm_misses_bonferroni_pair"] += int((dec["bonferroni"] & ~dec["holm"]).sum())
        out["holm_vs_bonferroni_any_differs_under_null"] += int(
            (dec["bonferroni"].any(axis=1) != dec["holm"].any(axis=1)).sum())
        out["lsd_without_F"] += int((dec["lsd"].any(axis=1) & (dec["F"] >= ALPHA)).sum())
        out["replicates"] += reps
    return out


def calibrate_against_scipy(trials: int = 300, seed: int = 7) -> Dict[str, float]:
    """Our Tukey decisions vs scipy.stats.tukey_hsd, and our F vs scipy.stats.f_oneway."""
    rng = np.random.default_rng(seed)
    mismatches, f_gap, compared = 0, 0.0, 0
    for t in range(trials):
        k = KS[t % len(KS)]
        data = rng.standard_normal((1, k, N_PER_GROUP)) + rng.normal(0, 0.4, k)[None, :, None]
        dec = decisions(data)
        groups = [data[0, g] for g in range(k)]
        ref = stats.tukey_hsd(*groups).pvalue
        for c, (i, j) in enumerate(pairs(k)):
            mismatches += int(bool(dec["tukey"][0, c]) != bool(ref[i, j] < ALPHA))
            compared += 1
        f_gap = max(f_gap, abs(float(dec["F"][0]) - stats.f_oneway(*groups).pvalue))
    return {"tukey_decision_mismatches": mismatches, "pairs_compared": compared,
            "max_F_p_gap": f_gap}


def analyse(groups: Sequence[Sequence[float]], alpha: float = ALPHA) -> Dict[str, object]:
    """One real dataset (unequal n allowed): F, and each procedure's list of significant pairs.

    Unequal n uses Tukey-Kramer via scipy.stats.tukey_hsd and a per-pair pooled-MSE SE for the rest.
    Edge case: a group with fewer than 2 values has no variance and is rejected up front."""
    arrs = [np.asarray(g, dtype=float) for g in groups]
    if len(arrs) < 3:
        raise ValueError("need at least 3 groups for a post-hoc question")
    if any(a.size < 2 for a in arrs):
        raise ValueError("every group needs at least 2 values")
    k = len(arrs)
    ns = np.array([a.size for a in arrs])
    means = np.array([a.mean() for a in arrs])
    df = int(ns.sum() - k)
    mse = sum(((a - a.mean()) ** 2).sum() for a in arrs) / df
    idx = pairs(k)
    p = np.array([2.0 * stats.t.sf(abs(means[i] - means[j]) / np.sqrt(mse * (1 / ns[i] + 1 / ns[j])), df)
                  for i, j in idx])[None, :]
    pf = float(stats.f_oneway(*arrs).pvalue)
    tk = stats.tukey_hsd(*arrs).pvalue
    rej = {
        "none": p[0] < alpha,
        "lsd": (p[0] < alpha) & (pf < alpha),
        "bonferroni": p[0] < alpha / len(idx),
        "holm": holm_reject(p, alpha)[0],
        "tukey": np.array([tk[i, j] < alpha for i, j in idx]),
    }
    return {"F_p": pf, "means": means.tolist(), "pairs": idx, "raw_p": p[0].tolist(),
            "tukey_p": [float(tk[i, j]) for i, j in idx],
            "significant": {proc: [idx[c] for c in np.flatnonzero(rej[proc])] for proc in PROCS}}


def exemplar(k: int = 5, effect: float = 1.0, max_seed: int = 10_000) -> Dict[str, object]:
    """Chosen by RULE, not by hand: the first seed of a k-group ladder where the F test is
    significant and Tukey names no pair, while an unadjusted pair list names at least one."""
    for seed in range(max_seed):
        data = simulate(k, "spread", effect, seed, reps=1)
        groups = [np.round(data[0, g], 2).tolist() for g in range(k)]
        res = analyse(groups)
        if res["F_p"] < ALPHA and not res["significant"]["tukey"] and res["significant"]["none"]:
            return {"seed": seed, "k": k, "effect": effect, "groups": groups, **res}
    raise RuntimeError("no exemplar found")
