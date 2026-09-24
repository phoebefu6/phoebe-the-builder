"""Did conversion actually move? Four tests on one 2x2 table, and where they part company.

A two-arm conversion comparison is `x1` successes of `n1` against `x2` of `n2`. Four tests are
routinely run on it and reported as though they were interchangeable:

  * the two-proportion z-test (pooled)
  * Pearson's chi-square, with or without Yates' continuity correction
  * Fisher's exact test (conditional on BOTH margins)
  * Barnard's exact test (unconditional - conditions only on the arm sizes)

The whole build rests on one fact about the outcome space: for fixed `n1`, `n2` there are only
`(n1 + 1) * (n2 + 1)` possible tables. So a test's rejection region can be computed ONCE per design,
and its size and power at ANY true pair of rates are then an exact weighted sum over that grid.
No Monte Carlo anywhere in this module, so no number here carries sampling error or an interval -
the same strengthening `ci-overlap-fallacy` (Day 180) made for its proportion section.

What separates this from the built `crosstab-chi2` (Day 120): that tool runs one test well. This one
measures the DISAGREEMENT between four, and what each costs.

Note on duplication: `z_for_level()` style helpers also appear in sibling builds. Deliberate -
every build must run standalone from a bare Colab link. One copy WITHIN this build; the chart, the
app and the notebook read this module's numbers rather than recomputing them.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

ALPHA = 0.05
TESTS = ("z", "yates", "fisher", "barnard")
LABEL = {"z": "z-test = chi-square", "yates": "chi-square + Yates",
         "fisher": "Fisher exact", "barnard": "Barnard exact"}

# Exact numbers have no interval, so the band is a plain threshold: a nominal 5% test is called
# INFLATED above 5.5% and CONSERVATIVE below 4.5% (10% relative either way).
BAND_LO, BAND_HI = 0.045, 0.055

# Barnard maximises over the nuisance rate. A grid gives a LOWER bound on that supremum, so the
# grid is dense and the result is checked against scipy.stats.barnard_exact in the tests.
NUISANCE = np.linspace(0.0005, 0.9995, 400)


def verdict(size: float) -> str:
    if size > BAND_HI:
        return "INFLATED"
    if size < BAND_LO:
        return "CONSERVATIVE"
    return "ok"


# --------------------------------------------------------------------------- 1. the four tests
# Each returns a (n1+1, n2+1) array of two-sided p-values, one per possible table.

def grid(n1: int, n2: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.arange(n1 + 1)[:, None], np.arange(n2 + 1)[None, :]


def z_stat(n1: int, n2: int) -> np.ndarray:
    """Pooled two-proportion z. Tables with an empty column (0 or all successes) score 0."""
    x1, x2 = grid(n1, n2)
    pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pool * (1.0 - pool) * (1.0 / n1 + 1.0 / n2))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (x1 / n1 - x2 / n2) / np.where(se > 0, se, 1.0), 0.0)
    return z


def p_z(n1: int, n2: int) -> np.ndarray:
    return 2.0 * stats.norm.sf(np.abs(z_stat(n1, n2)))


def chi2_uncorrected(n1: int, n2: int) -> np.ndarray:
    """Pearson's chi-square written from its own formula, NOT from z - so the identity
    z^2 == chi-square is something the tests check rather than something the code assumes."""
    x1, x2 = grid(n1, n2)
    a, b, c, d = x1, n1 - x1, x2, n2 - x2
    n = n1 + n2
    c1, c2 = a + c, b + d
    den = (n1 * n2) * c1 * c2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, n * (a * d - b * c) ** 2 / np.where(den > 0, den, 1), 0.0)


def p_yates(n1: int, n2: int) -> np.ndarray:
    """Chi-square with Yates' correction, matching scipy.chi2_contingency(correction=True):
    each |O - E| is reduced by min(0.5, |O - E|), which for a 2x2 is max(0, |ad - bc| - N/2)."""
    x1, x2 = grid(n1, n2)
    a, b, c, d = x1, n1 - x1, x2, n2 - x2
    n = n1 + n2
    c1, c2 = a + c, b + d
    den = (n1 * n2) * c1 * c2
    num = n * np.maximum(0.0, np.abs(a * d - b * c) - n / 2.0) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    return stats.chi2.sf(chi, 1)


def p_fisher(n1: int, n2: int) -> np.ndarray:
    """Two-sided Fisher, vectorised by margin. Same rule as scipy: sum every table on that margin
    whose hypergeometric probability is <= the observed one, with scipy's 1e-7 relative slack."""
    out = np.ones((n1 + 1, n2 + 1))
    for m in range(n1 + n2 + 1):
        lo, hi = max(0, m - n2), min(n1, m)
        k = np.arange(lo, hi + 1)
        pmf = stats.hypergeom.pmf(k, n1 + n2, n1, m)
        # p(k) = sum of pmf over tables no more likely than k
        order = np.sort(pmf)
        csum = np.cumsum(order)
        idx = np.searchsorted(order, pmf * (1 + 1e-7), side="right") - 1
        out[k, m - k] = np.minimum(1.0, csum[idx])
    return out


def p_barnard(n1: int, n2: int, nuisance: np.ndarray = NUISANCE) -> np.ndarray:
    """Barnard's unconditional exact test with the pooled z statistic (scipy's default).

    p(table) = sup over pi of P(|Z| >= |Z_obs| ; both arms at rate pi). Tables are sorted by |Z|
    once; for each pi the tail probability of every table is one cumulative sum, with ties
    resolved so that tables sharing a statistic share a p-value.
    """
    t = np.abs(z_stat(n1, n2)).ravel()
    order = np.argsort(-t, kind="stable")
    ts = t[order]
    # last index of each tie group, so the tail includes every table at least as extreme
    last = np.searchsorted(-ts, -ts, side="right") - 1
    x1, x2 = grid(n1, n2)
    best = np.zeros(t.size)
    for pi in nuisance:
        w = (stats.binom.pmf(x1, n1, pi) * stats.binom.pmf(x2, n2, pi)).ravel()[order]
        tail = np.cumsum(w)[last]
        best = np.maximum(best, tail)
    out = np.empty(t.size)
    out[order] = np.minimum(best, 1.0)
    return out.reshape(n1 + 1, n2 + 1)


def all_pvalues(n1: int, n2: int) -> Dict[str, np.ndarray]:
    return {"z": p_z(n1, n2), "yates": p_yates(n1, n2),
            "fisher": p_fisher(n1, n2), "barnard": p_barnard(n1, n2)}


# ------------------------------------------------------------------ 2. exact size and power

def weights(n1: int, n2: int, p1: float, p2: float) -> np.ndarray:
    x1, x2 = grid(n1, n2)
    return stats.binom.pmf(x1, n1, p1) * stats.binom.pmf(x2, n2, p2)


def rate(reject: np.ndarray, n1: int, n2: int, p1: float, p2: float) -> float:
    """Exact probability of landing in a rejection region. No replicates, no seed."""
    return float(weights(n1, n2, p1, p2)[reject].sum())


def size_curve(pv: np.ndarray, n1: int, n2: int, ps: Sequence[float],
               alpha: float = ALPHA) -> np.ndarray:
    """Exact Type I rate at each common true rate p. The test's SIZE is the max of this curve."""
    rej = pv < alpha
    return np.array([rate(rej, n1, n2, p, p) for p in ps])


def min_expected(n1: int, n2: int, p: float) -> float:
    """Smallest expected cell count under the null at true rate p - the quantity the
    'use Fisher when an expected count is below 5' rule is written in."""
    return float(min(n1, n2) * min(p, 1.0 - p))


# ----------------------------------------------------------------------- 3. the study design

SIZE_PS: List[float] = [round(float(p), 3) for p in np.linspace(0.01, 0.5, 50)]

# (n1, n2) - balanced ladder, then the unbalanced shapes real tests take (a holdout, a ramp)
SIZE_DESIGNS: List[Tuple[int, int]] = [
    (10, 10), (20, 20), (30, 30), (50, 50), (100, 100), (200, 200),
    (20, 80), (10, 100), (50, 200),
]

# (p1, p2, n1, n2) - conversion comparisons
POWER_DESIGNS: List[Tuple[float, float, int, int]] = [
    (0.10, 0.30, 30, 30),
    (0.10, 0.20, 100, 100),
    (0.10, 0.20, 200, 200),
    (0.05, 0.12, 200, 200),
    (0.02, 0.08, 200, 200),
    (0.30, 0.45, 100, 100),
    (0.10, 0.25, 20, 80),
]

LATTICE_N = 20   # the design drawn as a rejection-region map in the chart


def size_table(designs: Sequence[Tuple[int, int]] = tuple(SIZE_DESIGNS),
               ps: Sequence[float] = tuple(SIZE_PS)) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for n1, n2 in designs:
        pv = all_pvalues(n1, n2)
        row: Dict[str, object] = {"n1": n1, "n2": n2}
        for t in TESTS:
            curve = size_curve(pv[t], n1, n2, ps)
            j = int(np.argmax(curve))
            row[t] = float(curve[j])
            row[t + "_at_p"] = float(ps[j])
            row[t + "_mean"] = float(curve.mean())
            row[t + "_curve"] = [float(v) for v in curve]
            row[t + "_verdict"] = verdict(float(curve[j]))
        rows.append(row)
    return rows


def power_table(designs: Sequence[Tuple[float, float, int, int]] = tuple(POWER_DESIGNS)
                ) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for p1, p2, n1, n2 in designs:
        pv = all_pvalues(n1, n2)
        row: Dict[str, object] = {"p1": p1, "p2": p2, "n1": n1, "n2": n2}
        for t in TESTS:
            row[t] = rate(pv[t] < ALPHA, n1, n2, p1, p2)
        # the disagreement a reader actually meets: z says moved, Fisher says not
        row["z_not_fisher"] = rate((pv["z"] < ALPHA) & (pv["fisher"] >= ALPHA), n1, n2, p1, p2)
        row["barnard_not_fisher"] = rate((pv["barnard"] < ALPHA) & (pv["fisher"] >= ALPHA),
                                         n1, n2, p1, p2)
        rows.append(row)
    return rows


def expected_count_rule(size_rows: Sequence[Dict[str, object]],
                        ps: Sequence[float] = tuple(SIZE_PS)) -> Dict[str, object]:
    """Score the rule 'the z / chi-square test is fine when every expected count is >= 5' as a
    classifier, cell by cell, against the z-test's EXACT Type I rate at that cell."""
    safe_but_broken, unsafe_but_fine, safe_ok, unsafe_broken = [], [], 0, 0
    for r in size_rows:
        n1, n2 = int(r["n1"]), int(r["n2"])
        for p, s in zip(ps, r["z_curve"]):
            e = min_expected(n1, n2, p)
            broken = s > BAND_HI
            cell = {"n1": n1, "n2": n2, "p": p, "min_expected": e, "size": s}
            if e >= 5 and broken:
                safe_but_broken.append(cell)
            elif e < 5 and not broken:
                unsafe_but_fine.append(cell)
            elif e >= 5:
                safe_ok += 1
            else:
                unsafe_broken += 1
    return {"safe_but_broken": safe_but_broken, "unsafe_but_fine": unsafe_but_fine,
            "safe_ok": safe_ok, "unsafe_broken": unsafe_broken}


def exemplar(n: int = LATTICE_N) -> Dict[str, object]:
    """One table on which the four tests give the four answers a reader would meet.

    Chosen deterministically: the table with the smallest x1 + x2 (then smallest x1) where the
    z-test rejects and Fisher's p is above 0.05. Never hand-picked.
    """
    pv = all_pvalues(n, n)
    cand = [(i + j, i, j) for i in range(n + 1) for j in range(i + 1, n + 1)
            if pv["z"][i, j] < ALPHA and pv["fisher"][i, j] >= ALPHA]
    _, i, j = min(cand)
    return {"n": n, "x1": i, "x2": j, **{t: float(pv[t][i, j]) for t in TESTS}}


def lattice_regions(n: int = LATTICE_N) -> Dict[str, object]:
    """For every table in an n/n design, which of z / Barnard / Fisher rejects. Nesting is
    MEASURED, not assumed: `other` counts any table that breaks z > Barnard > Fisher."""
    pv = all_pvalues(n, n)
    z, b, f = pv["z"] < ALPHA, pv["barnard"] < ALPHA, pv["fisher"] < ALPHA
    code = np.full((n + 1, n + 1), 0)
    code[z & ~b & ~f] = 1
    code[z & b & ~f] = 2
    code[z & b & f] = 3
    other = ~((code > 0) | (~z & ~b & ~f))
    code[other] = 4
    return {"n": n, "code": code.tolist(), "counts": {
        "none": int((code == 0).sum()), "z_only": int((code == 1).sum()),
        "z_barnard": int((code == 2).sum()), "all": int((code == 3).sum()),
        "other": int((code == 4).sum())}}
