"""Monotone optimal binning with WOE/IV - and an IV number you can defend.

The mechanics of binning are easy. The reason binned scorecards ship broken features is
that the standard workflow measures a feature's strength on the same rows that chose its
cut points, so the strength is partly the fit.

Five mechanisms:

1. SEPARATION - missing values and sentinel codes (-999, 0-as-not-applicable) get their own
   bins and are never merged into a numeric range. Imputing them to the median merges two
   populations and destroys a signal that is often real (a thin credit file defaults
   differently).
2. SMOOTHED WOE - an empty or near-empty bin sends log-odds to infinity. Additive smoothing
   fixes the arithmetic, but the constant changes IV, so it is a declared parameter rather
   than a hidden 0.5.
3. CONSTRAINED MERGE - greedy adjacent merging under a minimum bin population and a minimum
   event count, choosing the merge that costs the least information at each step.
4. MONOTONICITY - scorecards need monotone WOE to survive review. Enforcing it costs IV.
   The size of that cost is diagnostic: small means the wiggle was noise, large means the
   relationship is genuinely non-monotone and you are destroying signal to get a shape.
5. HONEST IV - fit the cut points on train, recompute IV on holdout with those cut points
   frozen. The gap is the optimism. On pure noise it is nearly the whole number.

Plus PSI, because a bin scheme fit on last year's population drifts.

No database, no network, no sklearn. numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MISSING = "missing"
SPECIAL = "special"
NUMERIC = "numeric"


# --------------------------------------------------------------------------------------
# Bins
# --------------------------------------------------------------------------------------


@dataclass
class Bin:
    """One bin. `lo`/`hi` are the numeric edges; ignored for missing/special bins."""

    kind: str  # NUMERIC | MISSING | SPECIAL
    lo: float = -math.inf
    hi: float = math.inf
    special_value: Optional[float] = None
    n: int = 0
    events: int = 0

    @property
    def nonevents(self) -> int:
        return self.n - self.events

    @property
    def event_rate(self) -> float:
        return self.events / self.n if self.n else float("nan")

    @property
    def label(self) -> str:
        if self.kind == MISSING:
            return "missing"
        if self.kind == SPECIAL:
            return f"= {self.special_value:g} (special)"
        lo = "-inf" if self.lo == -math.inf else f"{self.lo:,.4g}"
        hi = "inf" if self.hi == math.inf else f"{self.hi:,.4g}"
        return f"({lo}, {hi}]"


@dataclass
class Binning:
    """A fitted scheme. Cut points are frozen here; applying them elsewhere is what makes
    the holdout IV honest."""

    feature: str
    bins: List[Bin]
    smoothing: float
    monotone: str  # "none" | "increasing" | "decreasing"
    total_events: int = 0
    total_nonevents: int = 0
    notes: List[str] = field(default_factory=list)

    # -- WOE / IV ----------------------------------------------------------------------

    def woe(self, b: Bin) -> float:
        """Smoothed log-odds of this bin against the population.

        Additive smoothing on both counts. Without it a zero-event bin returns -inf and
        every downstream sum becomes nan; with it, a 3-event bin stops pretending to be
        a 300-event bin.
        """
        a = self.smoothing
        p_event = (b.events + a) / (self.total_events + 2 * a)
        p_nonevent = (b.nonevents + a) / (self.total_nonevents + 2 * a)
        return math.log(p_event / p_nonevent)

    def share_event(self, b: Bin) -> float:
        a = self.smoothing
        return (b.events + a) / (self.total_events + 2 * a)

    def share_nonevent(self, b: Bin) -> float:
        a = self.smoothing
        return (b.nonevents + a) / (self.total_nonevents + 2 * a)

    @property
    def iv(self) -> float:
        return sum(
            (self.share_event(b) - self.share_nonevent(b)) * self.woe(b) for b in self.bins
        )

    @property
    def numeric_bins(self) -> List[Bin]:
        return [b for b in self.bins if b.kind == NUMERIC]

    @property
    def cuts(self) -> List[float]:
        """Interior cut points - the only thing that transfers to new data."""
        return [b.hi for b in self.numeric_bins[:-1]]

    @property
    def specials(self) -> List[float]:
        return [b.special_value for b in self.bins if b.kind == SPECIAL]

    def is_monotone(self, tol: float = 1e-12) -> Optional[str]:
        """Direction of monotone WOE across the numeric bins, or None if it wobbles.

        Missing and special bins are excluded by design: they are not on the numeric
        scale, so requiring them to sit in order is meaningless.
        """
        woes = [self.woe(b) for b in self.numeric_bins]
        if len(woes) < 2:
            return "increasing"
        diffs = [b - a for a, b in zip(woes, woes[1:])]
        if all(d >= -tol for d in diffs):
            return "increasing"
        if all(d <= tol for d in diffs):
            return "decreasing"
        return None

    def table(self) -> List[Dict[str, object]]:
        total = sum(b.n for b in self.bins)
        rows = []
        for b in self.bins:
            rows.append(
                {
                    "bin": b.label,
                    "kind": b.kind,
                    "n": b.n,
                    "share": b.n / total if total else float("nan"),
                    "events": b.events,
                    "event_rate": b.event_rate,
                    "woe": self.woe(b),
                    "iv_part": (self.share_event(b) - self.share_nonevent(b)) * self.woe(b),
                }
            )
        return rows

    # -- transform ---------------------------------------------------------------------

    def bin_index(self, value: float) -> int:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            for i, b in enumerate(self.bins):
                if b.kind == MISSING:
                    return i
            return -1
        for i, b in enumerate(self.bins):
            if b.kind == SPECIAL and b.special_value == value:
                return i
        for i, b in enumerate(self.bins):
            if b.kind == NUMERIC and b.lo < value <= b.hi:
                return i
        return -1

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Replace raw values with the WOE of their bin. This is what feeds the model."""
        woes = np.array([self.woe(b) for b in self.bins])
        idx = np.array([self.bin_index(v) for v in x])
        out = np.zeros(len(x))
        known = idx >= 0
        out[known] = woes[idx[known]]
        return out


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------


def _assign(
    x: np.ndarray, y: np.ndarray, edges: Sequence[float], specials: Sequence[float]
) -> Tuple[List[Bin], List[Bin], List[Bin]]:
    """Split rows into numeric prebins, one missing bin, and one bin per special value."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    is_missing = np.isnan(x)
    is_special = np.zeros(len(x), dtype=bool)
    for value in specials:
        is_special |= x == value

    missing_bins: List[Bin] = []
    if is_missing.any():
        missing_bins.append(
            Bin(MISSING, n=int(is_missing.sum()), events=int(y[is_missing].sum()))
        )

    special_bins: List[Bin] = []
    for value in specials:
        mask = x == value
        if mask.any():
            special_bins.append(
                Bin(SPECIAL, special_value=value, n=int(mask.sum()), events=int(y[mask].sum()))
            )

    keep = ~(is_missing | is_special)
    xs, ys = x[keep], y[keep]

    numeric_bins: List[Bin] = []
    lo = -math.inf
    for hi in list(edges) + [math.inf]:
        mask = (xs > lo) & (xs <= hi)
        numeric_bins.append(Bin(NUMERIC, lo=lo, hi=hi, n=int(mask.sum()), events=int(ys[mask].sum())))
        lo = hi
    return numeric_bins, missing_bins, special_bins


def _iv_of(bins: Sequence[Bin], smoothing: float) -> float:
    total_e = sum(b.events for b in bins)
    total_ne = sum(b.nonevents for b in bins)
    if total_e == 0 or total_ne == 0:
        return 0.0
    out = 0.0
    for b in bins:
        pe = (b.events + smoothing) / (total_e + 2 * smoothing)
        pne = (b.nonevents + smoothing) / (total_ne + 2 * smoothing)
        out += (pe - pne) * math.log(pe / pne)
    return out


def _merge(bins: List[Bin], i: int) -> List[Bin]:
    """Merge bin i with bin i+1."""
    a, b = bins[i], bins[i + 1]
    merged = Bin(NUMERIC, lo=a.lo, hi=b.hi, n=a.n + b.n, events=a.events + b.events)
    return bins[:i] + [merged] + bins[i + 2 :]


def _cheapest_merge(bins: List[Bin], smoothing: float) -> int:
    """Index of the adjacent pair whose merge loses the least information."""
    best, best_iv = 0, -math.inf
    for i in range(len(bins) - 1):
        iv = _iv_of(_merge(bins, i), smoothing)
        if iv > best_iv:
            best, best_iv = i, iv
    return best


def _woe_seq(bins: Sequence[Bin], smoothing: float) -> List[float]:
    total_e = sum(b.events for b in bins)
    total_ne = sum(b.nonevents for b in bins)
    out = []
    for b in bins:
        pe = (b.events + smoothing) / (total_e + 2 * smoothing)
        pne = (b.nonevents + smoothing) / (total_ne + 2 * smoothing)
        out.append(math.log(pe / pne))
    return out


def fit(
    x: np.ndarray,
    y: np.ndarray,
    feature: str = "x",
    max_bins: int = 6,
    max_prebins: int = 20,
    min_bin_share: float = 0.05,
    min_bin_events: int = 20,
    smoothing: float = 0.5,
    monotone: bool = True,
    specials: Sequence[float] = (),
) -> Binning:
    """Fit a binning scheme.

    Order matters. Size constraints come before monotonicity, because a monotone sequence
    built out of 4-row bins is monotone noise.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    notes: List[str] = []

    finite = x[~np.isnan(x)]
    for value in specials:
        finite = finite[finite != value]
    if len(finite) == 0:
        raise ValueError(f"{feature}: no numeric values left after removing missing/specials")

    quantiles = np.linspace(0, 100, max_prebins + 1)[1:-1]
    edges = sorted(set(np.percentile(finite, quantiles).tolist()))

    numeric, missing_bins, special_bins = _assign(x, y, edges, specials)
    if missing_bins:
        notes.append(
            f"{missing_bins[0].n} missing rows kept as their own bin "
            f"(event rate {missing_bins[0].event_rate:.1%})"
        )
    for b in special_bins:
        notes.append(
            f"{b.n} rows at sentinel {b.special_value:g} kept out of the numeric scale "
            f"(event rate {b.event_rate:.1%})"
        )

    n_numeric = sum(b.n for b in numeric)
    min_n = max(1, int(min_bin_share * n_numeric))

    # Stage 1 - satisfy the size floors.
    guard = 0
    while len(numeric) > 1 and guard < 500:
        guard += 1
        weak = [
            i
            for i, b in enumerate(numeric)
            if b.n < min_n or b.events < min_bin_events or b.nonevents < min_bin_events
        ]
        if not weak:
            break
        i = weak[0]
        j = i - 1 if i == len(numeric) - 1 else i
        numeric = _merge(numeric, j)

    # Stage 2 - respect max_bins, giving up the least information each time.
    while len(numeric) > max_bins:
        numeric = _merge(numeric, _cheapest_merge(numeric, smoothing))

    iv_unconstrained = _iv_of(numeric + missing_bins + special_bins, smoothing)
    direction = "none"

    # Stage 3 - monotonicity, by merging adjacent violators.
    if monotone and len(numeric) > 2:
        woes = _woe_seq(numeric, smoothing)
        ups = sum(1 for a, b in zip(woes, woes[1:]) if b > a)
        downs = len(woes) - 1 - ups
        direction = "increasing" if ups >= downs else "decreasing"
        guard = 0
        while len(numeric) > 2 and guard < 500:
            guard += 1
            woes = _woe_seq(numeric, smoothing)
            diffs = [b - a for a, b in zip(woes, woes[1:])]
            violations = [
                i
                for i, d in enumerate(diffs)
                if (d < 0 if direction == "increasing" else d > 0)
            ]
            if not violations:
                break
            worst = max(violations, key=lambda i: abs(diffs[i]))
            numeric = _merge(numeric, worst)

    bins = numeric + missing_bins + special_bins
    result = Binning(
        feature=feature,
        bins=bins,
        smoothing=smoothing,
        monotone=direction if monotone else "none",
        total_events=int(sum(b.events for b in bins)),
        total_nonevents=int(sum(b.nonevents for b in bins)),
        notes=notes,
    )
    if monotone:
        cost = iv_unconstrained - result.iv
        share = (
            f" ({cost / iv_unconstrained:.0%} of the unconstrained {iv_unconstrained:.4f})"
            if iv_unconstrained > 1e-12
            else " (unconstrained IV was already zero)"
        )
        result.notes.append(f"monotone ({direction}) cost {cost:.4f} IV{share}")
    return result


def refit_counts(scheme: Binning, x: np.ndarray, y: np.ndarray) -> Binning:
    """Re-count a FROZEN scheme on new rows. Cut points do not move.

    This is the whole honesty mechanism: same bins, different data, recomputed IV.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    fresh = [
        Bin(b.kind, lo=b.lo, hi=b.hi, special_value=b.special_value) for b in scheme.bins
    ]
    for value, label in zip(x, y):
        idx = scheme.bin_index(value)
        if idx < 0:
            continue
        fresh[idx].n += 1
        fresh[idx].events += int(label)
    return Binning(
        feature=scheme.feature,
        bins=fresh,
        smoothing=scheme.smoothing,
        monotone=scheme.monotone,
        total_events=int(sum(b.events for b in fresh)),
        total_nonevents=int(sum(b.nonevents for b in fresh)),
        notes=["counts recomputed on held-out rows; cut points frozen"],
    )


def psi(reference: Binning, current: Binning) -> float:
    """Population Stability Index between two count sets over the same bins."""
    ref_total = sum(b.n for b in reference.bins) or 1
    cur_total = sum(b.n for b in current.bins) or 1
    out = 0.0
    for rb, cb in zip(reference.bins, current.bins):
        r = max(rb.n / ref_total, 1e-6)
        c = max(cb.n / cur_total, 1e-6)
        out += (c - r) * math.log(c / r)
    return out


def null_iv(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 40,
    seed: int = 0,
    **kwargs: object,
) -> np.ndarray:
    """IV this binning procedure produces from a feature with NO relationship to the target.

    Shuffle the labels, refit with identical settings, record IV. Repeat.

    This is the reference the conventional 0.02 / 0.1 / 0.3 bands are missing. Those
    thresholds assume a fixed, modest number of bins. IV is not comparable across bin
    counts: more bins mechanically raise it, and additive smoothing on a near-empty bin
    manufactures a large positive IV contribution out of a count of zero. A permutation
    null absorbs both effects, because the null is measured through the same procedure.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    out = []
    for _ in range(n_permutations):
        try:
            out.append(fit(x, rng.permutation(y), **kwargs).iv)  # type: ignore[arg-type]
        except ValueError:
            continue
    return np.array(out)


SETTINGS_LADDER = (
    ("6 bins, 20-event floor", dict(max_bins=6, min_bin_events=20, min_bin_share=0.05)),
    ("10 bins, 5-event floor", dict(max_bins=10, min_bin_events=5, min_bin_share=0.01)),
    ("20 bins, 1-event floor", dict(max_bins=20, min_bin_events=1, min_bin_share=0.002)),
    (
        "20 bins, 1-event, no monotone",
        dict(max_bins=20, min_bin_events=1, min_bin_share=0.002, monotone=False),
    ),
)


def noise_screen(
    y: np.ndarray,
    n_columns: int = 12,
    n_permutations: int = 40,
    iv_threshold: float = 0.10,
    alpha: float = 0.05,
    seed: int = 99,
) -> List[Dict[str, object]]:
    """How many pure-noise columns each screening rule lets through, per settings ladder.

    Generates independent rng.normal() columns - guaranteed zero relationship to `y` - and
    screens each one two ways: by raw IV against the conventional threshold, and by the
    permutation p-value. Averaging over columns instead of showing one is deliberate; a
    single noise column can land anywhere in its own null distribution.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    columns = [rng.normal(0, 1, len(y)) for _ in range(n_columns)]
    out: List[Dict[str, object]] = []

    for label, kwargs in SETTINGS_LADDER:
        ivs, excesses, bins, passed_iv, passed_perm = [], [], [], 0, 0
        for i, x in enumerate(columns):
            scheme = fit(x, y, **kwargs)  # type: ignore[arg-type]
            nulls = null_iv(x, y, n_permutations=n_permutations, seed=i, **kwargs)
            p = (np.sum(nulls >= scheme.iv) + 1) / (len(nulls) + 1)
            ivs.append(scheme.iv)
            excesses.append(scheme.iv - float(np.median(nulls)))
            bins.append(len(scheme.bins))
            passed_iv += scheme.iv >= iv_threshold
            passed_perm += p <= alpha
        out.append(
            {
                "settings": label,
                "mean_bins": float(np.mean(bins)),
                "mean_iv": float(np.mean(ivs)),
                "mean_excess": float(np.mean(excesses)),
                "kept_by_iv": passed_iv / n_columns,
                "kept_by_permutation": passed_perm / n_columns,
            }
        )
    return out


def sparse_bin_warning(scheme: Binning, floor: int = 20) -> Optional[str]:
    """IV stops meaning anything once bins are this thin - it becomes a smoothing artifact."""
    thin = [b for b in scheme.bins if b.events < floor or b.nonevents < floor]
    if not thin:
        return None
    return (
        f"{len(thin)} of {len(scheme.bins)} bins hold fewer than {floor} events or non-events; "
        "their WOE is driven by the smoothing constant, not by the data"
    )


IV_BANDS = (
    (0.02, "unpredictive"),
    (0.10, "weak"),
    (0.30, "medium"),
    (0.50, "strong"),
    (math.inf, "suspiciously strong - check for leakage"),
)


def iv_band(value: float) -> str:
    for threshold, label in IV_BANDS:
        if value < threshold:
            return label
    return "unknown"


# --------------------------------------------------------------------------------------
# Honesty report
# --------------------------------------------------------------------------------------


@dataclass
class Audit:
    feature: str
    iv_train: float
    iv_holdout: float
    monotone_train: Optional[str]
    monotone_holdout: Optional[str]
    psi: float
    n_bins: int
    scheme: Binning
    holdout: Binning
    null: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def optimism(self) -> float:
        return self.iv_train - self.iv_holdout

    @property
    def shrinkage(self) -> float:
        return self.optimism / self.iv_train if self.iv_train else 0.0

    @property
    def null_median(self) -> float:
        return float(np.median(self.null)) if len(self.null) else float("nan")

    @property
    def excess_iv(self) -> float:
        """IV above what this procedure produces from shuffled labels. The number to quote."""
        return self.iv_train - self.null_median if len(self.null) else float("nan")

    @property
    def p_value(self) -> float:
        """Fraction of permutations reaching the observed IV (add-one smoothed)."""
        if not len(self.null):
            return float("nan")
        return float((np.sum(self.null >= self.iv_train) + 1) / (len(self.null) + 1))

    @property
    def sparse_warning(self) -> Optional[str]:
        return sparse_bin_warning(self.scheme)

    @property
    def verdict(self) -> str:
        if len(self.null) and self.p_value > 0.05:
            return "DROP - a shuffled target scores this well"
        if self.iv_holdout < 0.02:
            return "DROP - no out-of-sample signal"
        if self.shrinkage > 0.5:
            return "SUSPECT - most of the in-sample IV did not survive"
        if self.sparse_warning:
            return "SUSPECT - IV is smoothing-driven, bins too thin"
        if self.psi > 0.25:
            return "REVIEW - population shifted (PSI > 0.25)"
        if self.monotone_holdout is None:
            return "REVIEW - monotone on train, wobbles on holdout"
        return "OK"


def audit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_holdout: np.ndarray,
    y_holdout: np.ndarray,
    feature: str = "x",
    n_permutations: int = 40,
    **kwargs: object,
) -> Audit:
    """Fit on train, then judge three ways: against a holdout, against a permutation null,
    and against the bin-thinness floor. The raw in-sample IV is the least useful of them."""
    scheme = fit(x_train, y_train, feature=feature, **kwargs)  # type: ignore[arg-type]
    holdout = refit_counts(scheme, x_holdout, y_holdout)
    null = (
        null_iv(x_train, y_train, n_permutations=n_permutations, **kwargs)
        if n_permutations
        else np.array([])
    )
    return Audit(
        feature=feature,
        iv_train=scheme.iv,
        iv_holdout=holdout.iv,
        monotone_train=scheme.is_monotone(),
        monotone_holdout=holdout.is_monotone(),
        psi=psi(scheme, holdout),
        n_bins=len(scheme.bins),
        scheme=scheme,
        holdout=holdout,
        null=null,
    )


# --------------------------------------------------------------------------------------
# Deterministic credit-style dataset
# --------------------------------------------------------------------------------------

SENTINEL_NO_BUREAU = -999.0


def build_dataset(n: int = 12000, seed: int = 11) -> Dict[str, object]:
    """A small application-scorecard dataset with the four awkward cases planted.

    - utilization      : genuinely monotone, strong
    - income           : genuinely monotone, moderate
    - age              : genuinely NON-monotone (U-shaped) - monotonicity costs real signal
    - months_employed  : 18% missing, and missingness is predictive (thin file)
    - n_inquiries      : sentinel -999 for 'no bureau record', which is the riskiest group
    - noise            : pure noise, no relationship whatsoever
    """
    rng = np.random.default_rng(seed)

    utilization = np.clip(rng.beta(2.0, 3.0, n), 0.001, 0.999)
    income = np.exp(rng.normal(10.7, 0.45, n))
    age = rng.integers(19, 76, n).astype(float)
    months_employed = np.clip(rng.gamma(2.0, 30.0, n), 0, 480)
    n_inquiries = rng.poisson(1.4, n).astype(float)
    noise = rng.normal(0, 1, n)

    # Log-odds of default.
    logit = -2.9
    logit = logit + 2.6 * (utilization - 0.4)
    logit = logit - 0.9 * (np.log(income) - 10.7)
    logit = logit + 0.0016 * (age - 42.0) ** 2 - 0.35  # U-shaped: young and old are riskier
    logit = logit + 0.10 * np.minimum(n_inquiries, 6)

    # Thin file: no employment record on file, and that group is riskier.
    thin = rng.random(n) < 0.18
    months_employed[thin] = np.nan
    logit = logit + 0.75 * thin

    # No bureau record at all: sentinel, and the riskiest group on the book.
    no_bureau = rng.random(n) < 0.06
    n_inquiries[no_bureau] = SENTINEL_NO_BUREAU
    logit = logit + 1.15 * no_bureau

    # Employment length reduces risk where it is known.
    known = ~thin
    logit[known] = logit[known] - 0.004 * (months_employed[known] - 60.0)

    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)

    # Time index, so drift is demonstrable: utilization creeps up in the later period.
    period = np.where(np.arange(n) < n // 2, "H1", "H2")
    late = period == "H2"
    utilization[late] = np.clip(utilization[late] + 0.10, 0.001, 0.999)

    features = {
        "utilization": utilization,
        "income": income,
        "age": age,
        "months_employed": months_employed,
        "n_inquiries": n_inquiries,
        "noise": noise,
    }
    specials = {"n_inquiries": (SENTINEL_NO_BUREAU,)}

    idx = rng.permutation(n)
    split = int(0.6 * n)
    train_idx, holdout_idx = idx[:split], idx[split:]

    return {
        "features": features,
        "y": y,
        "specials": specials,
        "period": period,
        "train_idx": train_idx,
        "holdout_idx": holdout_idx,
        "base_rate": float(y.mean()),
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def format_table(scheme: Binning) -> str:
    lines = [
        f"{'bin':<24} {'n':>6} {'share':>7} {'events':>7} {'rate':>7} {'WOE':>8} {'IV part':>8}"
    ]
    lines.append("-" * 72)
    for row in scheme.table():
        lines.append(
            f"{str(row['bin']):<24} {row['n']:>6} {row['share']:>6.1%} {row['events']:>7} "
            f"{row['event_rate']:>6.1%} {row['woe']:>8.3f} {row['iv_part']:>8.4f}"
        )
    lines.append("-" * 72)
    lines.append(f"{'IV':<24} {scheme.iv:>6.4f}  ({iv_band(scheme.iv)})")
    mono = scheme.is_monotone()
    lines.append(f"{'monotone':<24} {mono or 'NO - WOE wobbles'}")
    for note in scheme.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def audit_report(audits: Sequence[Audit]) -> str:
    lines = [
        f"{'feature':<17} {'IV train':>9} {'IV null':>8} {'excess':>8} {'p':>6} "
        f"{'IV hold':>8} {'PSI':>6}  verdict"
    ]
    lines.append("-" * 104)
    for a in sorted(audits, key=lambda a: -a.excess_iv):
        lines.append(
            f"{a.feature:<17} {a.iv_train:>9.4f} {a.null_median:>8.4f} {a.excess_iv:>8.4f} "
            f"{a.p_value:>6.3f} {a.iv_holdout:>8.4f} {a.psi:>6.3f}  {a.verdict}"
        )
    lines.append("")
    lines.append("  IV null = median IV this same procedure gets from SHUFFLED labels.")
    lines.append("  excess  = IV train - IV null. Quote this, not IV train.")
    for a in audits:
        if a.sparse_warning:
            lines.append(f"  ! {a.feature}: {a.sparse_warning}")
    return "\n".join(lines)


if __name__ == "__main__":
    data = build_dataset()
    y = data["y"]
    tr, ho = data["train_idx"], data["holdout_idx"]
    print(f"n={len(y)}  base default rate={data['base_rate']:.2%}\n")

    audits = []
    for name, values in data["features"].items():
        audits.append(
            audit(
                values[tr],
                y[tr],
                values[ho],
                y[ho],
                feature=name,
                specials=data["specials"].get(name, ()),
            )
        )
    print(audit_report(audits))

    print("\n\nutilization - the honest case")
    print(format_table([a for a in audits if a.feature == "utilization"][0].scheme))

    print("\n\nnoise - what in-sample IV screening ships")
    noise = [a for a in audits if a.feature == "noise"][0]
    print(format_table(noise.scheme))
    print(f"\n  holdout IV with those same cut points: {noise.iv_holdout:.4f} ({iv_band(noise.iv_holdout)})")

    print("\n\nIV is not comparable across bin counts")
    print("12 independent pure-noise columns, 480 training rows each, four binner settings.")
    print("Every column has ZERO relationship to the target by construction.\n")
    small = build_dataset(n=800, seed=3)
    ys, ts = small["y"], small["train_idx"]
    print(
        f"  {'settings':<32} {'bins':>5} {'mean IV':>8} {'excess':>8} "
        f"{'kept by IV>=0.1':>16} {'kept by perm':>13}"
    )
    print("  " + "-" * 88)
    for row in noise_screen(ys[ts]):
        print(
            f"  {str(row['settings']):<32} {row['mean_bins']:>5.1f} {row['mean_iv']:>8.4f} "
            f"{row['mean_excess']:>8.4f} {row['kept_by_iv']:>15.0%} {row['kept_by_permutation']:>12.0%}"
        )
    print(
        "\n  Raw IV on pure noise climbs 10x with the bin count. Drop the event floor and\n"
        "  every rng.normal() column clears the conventional 0.10 'medium predictor' bar.\n"
        "  Excess IV stays around 0.01-0.02 the whole way, and the permutation screen stays\n"
        "  near its nominal 5% - both because the null is measured through the same procedure.\n"
        "  (At 12 columns those keep-rates are themselves noisy to about +/-10pp.)"
    )

    print("\n\nn_inquiries - sentinel kept off the numeric scale")
    print(format_table([a for a in audits if a.feature == "n_inquiries"][0].scheme))

    print("\n\nage - genuinely non-monotone, so the constraint costs real signal")
    age = data["features"]["age"]
    free = fit(age[tr], y[tr], feature="age", monotone=False)
    forced = fit(age[tr], y[tr], feature="age", monotone=True)
    print(f"  unconstrained IV {free.iv:.4f} in {len(free.bins)} bins, monotone={free.is_monotone()}")
    print(f"  monotone      IV {forced.iv:.4f} in {len(forced.bins)} bins")
    print(f"  cost of the shape: {free.iv - forced.iv:.4f} IV ({(free.iv - forced.iv) / free.iv:.0%})")
