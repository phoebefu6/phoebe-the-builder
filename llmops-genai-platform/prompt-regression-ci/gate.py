"""A prompt CI gate is a test with a false-alarm rate. This module computes it exactly.

A pull request changes a prompt from A to B. CI runs a held-out set of N cases under both prompts, k runs
each, and a rule decides which cases are "broken" (passed under A, fail under B). The gate goes red when
at least T cases are broken. Every case i has a per-run pass probability pi_i under each prompt, and runs
are independent, so:

    P(case i flagged)          = a sum of binomial pmf products          (exact)
    number of flagged cases    = Poisson-binomial over the N cases        (exact, by convolution)
    P(gate red)                = P(count >= T)                            (exact)

Rules compared:
    naive k=1        one run each; flagged if A passed and B failed - the diff report most teams ship
    majority k       k runs each (k odd); flagged if A passed the majority and B failed the majority
    fisher k         k runs each; flagged if a one-sided Fisher exact test on (A passes, B passes) < 0.05
    quarantine + r   first re-run A `Q_RUNS` times on the baseline; any case whose runs disagree is
                     quarantined (never gated); rule r on the rest

The held-out set is synthetic and its structure is declared below.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

ALPHA_GATE = 0.05  # target false-red rate for a no-op PR
N_STABLE_PASS, N_STABLE_FAIL, N_FLAKY = 246, 24, 30
N = N_STABLE_PASS + N_STABLE_FAIL + N_FLAKY
STABLE_P = 0.995  # a "stable" pass still fails 1 run in 200
Q_RUNS = 4  # baseline re-runs used to find flaky cases
N_BREAK = 6  # cases a real side effect breaks
PRS_PER_MONTH = 40
SEED = 2026
RULES: Tuple[Tuple[str, str, int, bool], ...] = (
    ("naive k=1", "majority", 1, False),
    ("majority k=3", "majority", 3, False),
    ("majority k=5", "majority", 5, False),
    ("fisher k=5", "fisher", 5, False),
    ("quarantine + naive k=1", "majority", 1, True),
    ("quarantine + majority k=3", "majority", 3, True),
)
SCENARIOS = ("no-op", "hard break", "intermittent break", "break on flaky cases")
MC_REPS = 20000


def held_out_set(seed: int = SEED) -> np.ndarray:
    """Per-run pass probability of each case under prompt A. Flaky cases draw pi from Beta(2, 2)."""
    rng = np.random.default_rng(seed)
    return np.r_[np.full(N_STABLE_PASS, STABLE_P), np.full(N_STABLE_FAIL, 1 - STABLE_P), rng.beta(2, 2, N_FLAKY)]


def scenario(name: str, pi_a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(pi_b, is_really_broken). Breaks hit the first N_BREAK cases of the relevant block."""
    pi_b, real = pi_a.copy(), np.zeros(len(pi_a), dtype=bool)
    if name == "no-op":
        return pi_b, real
    idx = np.arange(N_BREAK) if name != "break on flaky cases" else N_STABLE_PASS + N_STABLE_FAIL + np.arange(N_BREAK)
    pi_b[idx] = {"hard break": 1 - STABLE_P, "intermittent break": 0.5, "break on flaky cases": 1 - STABLE_P}[name]
    real[idx] = True
    return pi_b, real


@lru_cache(maxsize=None)
def reject_table(kind: str, k: int) -> np.ndarray:
    """reject[a, b] = does the rule flag a case with a passes of k under A and b passes of k under B."""
    a, b = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
    if kind == "majority":
        return (a > k / 2) & (b < k / 2)
    if kind == "fisher":
        out = np.zeros((k + 1, k + 1), dtype=bool)
        for i in range(k + 1):
            for j in range(k + 1):
                if i > j:
                    out[i, j] = stats.fisher_exact([[i, k - i], [j, k - j]], alternative="greater")[1] < 0.05
        return out
    raise ValueError(kind)


def flag_prob(pi_a: np.ndarray, pi_b: np.ndarray, kind: str, k: int, quarantine: bool) -> np.ndarray:
    """Exact per-case probability of being listed as broken."""
    ks = np.arange(k + 1)
    pa = stats.binom.pmf(ks[None, :], k, np.asarray(pi_a)[:, None])
    pb = stats.binom.pmf(ks[None, :], k, np.asarray(pi_b)[:, None])
    p = np.einsum("ia,ab,ib->i", pa, reject_table(kind, k).astype(float), pb)
    return p * not_quarantined(pi_a) if quarantine else p


def not_quarantined(pi_a: np.ndarray, runs: int = Q_RUNS) -> np.ndarray:
    """A case survives quarantine if all baseline runs agree (baseline runs are separate from the gate's)."""
    pi_a = np.asarray(pi_a)
    return pi_a ** runs + (1 - pi_a) ** runs


def count_pmf(p: np.ndarray) -> np.ndarray:
    """Poisson-binomial pmf of the number of flagged cases, by exact convolution."""
    pmf = np.array([1.0])
    for pi in p:
        pmf = np.convolve(pmf, [1 - pi, pi])
    return pmf


def red_prob(p: np.ndarray, t: int) -> float:
    return float(count_pmf(p)[t:].sum())


def threshold(p_noop: np.ndarray, alpha: float = ALPHA_GATE) -> int:
    """Smallest T whose no-op false-red rate is <= alpha."""
    tail = np.cumsum(count_pmf(p_noop)[::-1])[::-1]
    return int(np.argmax(tail <= alpha))


def evaluate(pi_a: np.ndarray = None) -> List[Dict]:
    """Every rule under every scenario: false reds, the calibrated threshold, power, report precision, cost."""
    pi_a = held_out_set() if pi_a is None else pi_a
    out = []
    for label, kind, k, quar in RULES:
        p0 = flag_prob(pi_a, pi_a, kind, k, quar)
        t = threshold(p0)
        row = {"rule": label, "k": k, "quarantine": quar, "T": t, "noop_expected_flags": float(p0.sum()),
               "noop_red_at_T1": red_prob(p0, 1), "noop_red_at_T": red_prob(p0, t),
               "month_any_false_red_T1": 1 - (1 - red_prob(p0, 1)) ** PRS_PER_MONTH,
               "month_any_false_red_T": 1 - (1 - red_prob(p0, t)) ** PRS_PER_MONTH,
               "calls_per_pr": 2 * N * k, "gated_cases": float(not_quarantined(pi_a).sum()) if quar else float(N)}
        for sc in SCENARIOS[1:]:
            pi_b, real = scenario(sc, pi_a)
            p = flag_prob(pi_a, pi_b, kind, k, quar)
            row[sc] = {"red_at_T": red_prob(p, t), "red_at_T1": red_prob(p, 1),
                       "recall": float(p[real].mean()), "precision": float(p[real].sum() / p.sum()),
                       "expected_flags": float(p.sum())}
        out.append(row)
    return out


def simulate(pi_a: np.ndarray, pi_b: np.ndarray, kind: str, k: int, quarantine: bool, reps: int,
             seed: int = 7) -> np.ndarray:
    """Raw runs, raw rule: the flagged count in each replicate. The check on the exact engine."""
    rng = np.random.default_rng(seed)
    a = rng.binomial(k, pi_a, size=(reps, len(pi_a)))
    b = rng.binomial(k, pi_b, size=(reps, len(pi_a)))
    flagged = reject_table(kind, k)[a, b]
    if quarantine:
        base = rng.random((reps, len(pi_a), Q_RUNS)) < pi_a[None, :, None]
        flagged &= base.all(axis=2) | (~base).all(axis=2)
    return flagged.sum(axis=1)


def wilson(k: int, n: int, z: float = float(stats.norm.isf(0.005))) -> List[float]:
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(mid - half), float(mid + half)]


def calibrate(reps: int = MC_REPS) -> Dict:
    """Exact red rates vs raw simulation, and the Poisson-binomial vs scipy's binomial where they coincide."""
    pi_a = held_out_set()
    rows = []
    for label, kind, k, quar in RULES:
        t = threshold(flag_prob(pi_a, pi_a, kind, k, quar))
        for sc in ("no-op", "intermittent break"):
            pi_b, _ = scenario(sc, pi_a)
            ex = red_prob(flag_prob(pi_a, pi_b, kind, k, quar), t)
            hits = int((simulate(pi_a, pi_b, kind, k, quar, reps) >= t).sum())
            lo, hi = wilson(hits, reps)
            rows.append({"rule": label, "scenario": sc, "T": t, "exact": ex, "mc": hits / reps,
                         "inside_99": bool(lo <= ex <= hi)})
    same = np.full(50, 0.13)
    gap = float(np.max(np.abs(count_pmf(same) - stats.binom.pmf(np.arange(51), 50, 0.13))))
    return {"red_rates": rows, "poisson_binomial_vs_binom_gap": gap}


def audit(baseline_runs: Sequence[Sequence[int]], candidate_runs: Sequence[Sequence[int]],
          q_runs: Sequence[Sequence[int]] = ()) -> Dict:
    """The app's core: per-case 0/1 results for A and B (k runs each), optional extra baseline runs.

    Returns the naive k=1 diff (first run only), the majority diff, the Fisher diff, which cases a
    quarantine would drop, and the expected number of broken cases a no-op would produce at these
    pass rates - the noise floor to read the diff against.
    """
    a = np.asarray(baseline_runs, dtype=int)
    b = np.asarray(candidate_runs, dtype=int)
    if a.ndim != 2 or a.shape != b.shape or a.size == 0:
        raise ValueError("baseline and candidate must be the same cases x runs grid")
    if not np.isin(a, (0, 1)).all() or not np.isin(b, (0, 1)).all():
        raise ValueError("results must be 0 or 1")
    k = a.shape[1]
    sa, sb = a.sum(axis=1), b.sum(axis=1)
    naive = (a[:, 0] == 1) & (b[:, 0] == 0)
    maj = reject_table("majority", k)[sa, sb] if k % 2 else np.zeros(len(a), dtype=bool)
    fish = reject_table("fisher", k)[sa, sb]
    allbase = np.c_[a, np.asarray(q_runs, dtype=int)] if len(q_runs) else a
    flaky = ~((allbase == 1).all(axis=1) | (allbase == 0).all(axis=1))
    pi_hat = (allbase.sum(axis=1) + 0.5) / (allbase.shape[1] + 1)  # Jeffreys-style, never exactly 0 or 1
    floor = float(flag_prob(pi_hat, pi_hat, "majority", 1, False).sum())
    return {"k": k, "n": len(a), "naive": np.flatnonzero(naive).tolist(), "majority": np.flatnonzero(maj).tolist(),
            "fisher": np.flatnonzero(fish).tolist(), "flaky": np.flatnonzero(flaky).tolist(),
            "noop_floor_naive": floor, "pass_a": float(a.mean()), "pass_b": float(b.mean())}
