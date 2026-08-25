"""Scoring a decision log.

A decision log without a *prediction* attached is a diary: it records what
was chosen and cannot say whether choosing it was any good.  Attach a
probability and it becomes an instrument - and an instrument has a
**scoring rule**, which is a choice, and which almost nobody makes
deliberately.

That choice has a consequence most teams never see: a scoring rule can be
**improper**, meaning the forecast that maximises a person's expected
score is not the forecast they actually believe.  Under an improper rule,
telling you the truth costs your team points.  Three of the six rules
below are improper, two of them are the ones teams invent themselves, and
this module computes the exact lie each one pays for.

Nothing here is modelled.  Propriety is established by optimising the
expected score over a grid of reports for every true belief; the
decomposition is Murphy's; the power figures come from a paired bootstrap
over simulated-but-fully-specified forecasters.  The only authored object
is the decision corpus in `RECORDS`, which is illustrative - its
resolvability rate is a property of the corpus, not a measurement of the
world.  The linter that produces that rate is the reusable part.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

RNG_SEED = 20260825


# --------------------------------------------------------------------------
# Scoring rules
# --------------------------------------------------------------------------
#
# Every rule below is expressed as a LOSS: lower is better, so they can be
# compared on one axis.  `q` is the reported probability that the event
# happens; `y` is the outcome, 1 or 0.


@dataclass(frozen=True)
class Rule:
    name: str
    loss: Callable[[np.ndarray, np.ndarray], np.ndarray]
    proper: Optional[bool]  # filled in by `propriety()`, not asserted here
    bounded: bool
    seen_in: str
    note: str = ""


def _clip(q: np.ndarray) -> np.ndarray:
    """Log loss is infinite at 0 and 1; every real scorer clips."""
    return np.clip(q, 1e-9, 1 - 1e-9)


def brier(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(q - y)^2. The default, and the one most people mean by 'Brier'."""
    return (q - y) ** 2


def log_loss(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """-ln P(observed). Local, unbounded, and unforgiving of confident misses."""
    q = _clip(q)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def spherical(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1 - the spherical score. Proper, bounded, rarely used."""
    p_obs = np.where(y == 1, q, 1 - q)
    return 1.0 - p_obs / np.sqrt(q**2 + (1 - q) ** 2)


def absolute(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|q - y|. Looks like Brier, is not proper, and is very commonly used."""
    return np.abs(q - y)


def threshold_01(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """'Were you right?' - 0/1 loss at a 50% cut. Discards confidence entirely."""
    return ((q >= 0.5).astype(float) != y).astype(float)


def confidence_points(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The homebrew: +q points when right, -q when wrong. Expressed as a loss.

    Every team that has ever run a prediction game has invented this one.
    """
    right = np.where(y == 1, q, 1 - q) > 0.5
    signed = np.where(right, q, -q)
    return -signed


RULES: Tuple[Rule, ...] = (
    Rule("brier", brier, None, True, "sklearn brier_score_loss, weather verification"),
    Rule("log", log_loss, None, False, "cross-entropy, every ML training loop"),
    Rule("spherical", spherical, None, True, "forecasting literature, rarely in practice"),
    Rule("absolute", absolute, None, True, "'average error' in a spreadsheet"),
    Rule("threshold_01", threshold_01, None, True, "'what was our hit rate?'"),
    Rule("confidence_points", confidence_points, None, True,
         "the in-house prediction game everybody builds"),
)

RULES_BY_NAME: Dict[str, Rule] = {r.name: r for r in RULES}


# --------------------------------------------------------------------------
# Propriety - computed, not asserted
# --------------------------------------------------------------------------

GRID = np.linspace(0.0, 1.0, 1001)


def expected_loss(rule: Rule, p: float, reports: np.ndarray = GRID) -> np.ndarray:
    """E[loss] for each possible report `q`, given true belief `p`.

    A forecaster who believes `p` and wants the best expected score will
    report the `q` that minimises this. If that `q` is not `p`, the rule
    pays them to misreport.
    """
    y1 = np.ones_like(reports)
    y0 = np.zeros_like(reports)
    return p * rule.loss(reports, y1) + (1 - p) * rule.loss(reports, y0)


def optimal_report(rule: Rule, p: float) -> float:
    """The report that minimises expected loss - the honest answer or the lie."""
    return float(GRID[int(np.argmin(expected_loss(rule, p)))])


def optimal_report_set(rule: Rule, p: float, tol: float = 1e-12) -> Tuple[float, float, float]:
    """(lo, hi, width) of the reports that tie for the best expected score.

    Reporting a single "optimal q" hides the more damaging property: under
    `threshold_01` every report on one side of 0.5 scores identically, so
    the rule is indifferent to confidence across half its own range. A rule
    with a wide optimal plateau does not reward precision - it cannot see it.
    """
    e = expected_loss(rule, p)
    best = e.min()
    tied = GRID[e <= best + tol]
    return float(tied.min()), float(tied.max()), float(tied.max() - tied.min())


@lru_cache(maxsize=None)
def propriety(rule_name: str, tol: float = 0.005) -> Tuple[bool, float, Tuple[float, ...]]:
    """(is_proper, worst_gap, the beliefs at which the optimum is not p).

    Proper means: for every true belief p, reporting p is optimal.
    """
    rule = RULES_BY_NAME[rule_name]
    beliefs = np.round(np.linspace(0.01, 0.99, 99), 2)
    gaps, bad = [], []
    for p in beliefs:
        q = optimal_report(rule, float(p))
        gap = abs(q - p)
        gaps.append(gap)
        if gap > tol:
            bad.append(float(p))
    return (len(bad) == 0, float(max(gaps)), tuple(bad))


def optimal_lie_table(rule_name: str,
                      beliefs: Sequence[float] = (0.55, 0.6, 0.7, 0.8, 0.9)) -> List[Tuple[float, float]]:
    """What an optimising forecaster reports, for each thing they actually believe."""
    rule = RULES_BY_NAME[rule_name]
    return [(float(p), optimal_report(rule, float(p))) for p in beliefs]


# --------------------------------------------------------------------------
# Forecasters - fully specified, so the numbers are reproducible
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Forecaster:
    name: str
    bias: float = 0.0        # added to the log-odds of the true probability
    sharpen: float = 1.0     # >1 pushes reports toward 0/1, <1 toward 0.5
    noise: float = 0.0       # sd of log-odds noise
    blind: bool = False      # ignores the signal entirely, reports the base rate
    description: str = ""


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


FORECASTERS: Tuple[Forecaster, ...] = (
    Forecaster("calibrated", description="reports the true probability"),
    Forecaster("overconfident", sharpen=2.2,
               description="right direction, always too sure - the default human"),
    Forecaster("underconfident", sharpen=0.45,
               description="hedges everything toward 50%"),
    Forecaster("optimist", bias=0.9,
               description="calibrated shape, systematically too rosy"),
    Forecaster("noisy_expert", noise=1.1,
               description="unbiased and inconsistent - real signal, real variance"),
    Forecaster("base_rate", blind=True,
               description="ignores the case, always reports the base rate"),
)


@lru_cache(maxsize=1)
def simulate(n: int = 4000) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """One shared set of events, forecast by everybody. Returns (outcomes, reports).

    The true probability of each event is drawn from a Beta so the base
    rate is realistic and the events are genuinely uncertain.
    """
    rng = np.random.default_rng(RNG_SEED)
    truth = rng.beta(2.0, 2.5, size=n)
    outcomes = (rng.random(n) < truth).astype(float)
    base = float(truth.mean())

    reports: Dict[str, np.ndarray] = {}
    for f in FORECASTERS:
        if f.blind:
            q = np.full(n, base)
        else:
            z = _logit(truth) * f.sharpen + f.bias
            if f.noise:
                z = z + rng.normal(0.0, f.noise, size=n)
            q = _sigmoid(z)
        reports[f.name] = np.clip(q, 0.001, 0.999)
    return outcomes, reports


def score_table() -> Dict[str, Dict[str, float]]:
    """Mean loss for every (forecaster, rule) pair on the shared event set."""
    outcomes, reports = simulate()
    return {
        f.name: {r.name: float(np.mean(r.loss(reports[f.name], outcomes))) for r in RULES}
        for f in FORECASTERS
    }


def ranking(rule_name: str) -> List[str]:
    """Forecasters ordered best-to-worst under one rule."""
    table = score_table()
    return sorted(table, key=lambda f: table[f][rule_name])


def ranking_disagreement() -> Dict[Tuple[str, str], int]:
    """Pairs of rules, and how many forecaster pairs they order differently.

    A scoring rule does not measure who is best. It *defines* who is best.
    """
    names = [f.name for f in FORECASTERS]
    out: Dict[Tuple[str, str], int] = {}
    table = score_table()
    for a in RULES:
        for b in RULES:
            flips = 0
            for i, fa in enumerate(names):
                for fb in names[i + 1:]:
                    oa = table[fa][a.name] < table[fb][a.name]
                    ob = table[fa][b.name] < table[fb][b.name]
                    if oa != ob:
                        flips += 1
            out[(a.name, b.name)] = flips
    return out


# --------------------------------------------------------------------------
# Murphy decomposition
# --------------------------------------------------------------------------


def murphy(q: np.ndarray, y: np.ndarray, bins: int = 10) -> Dict[str, float]:
    """Brier = reliability - resolution + uncertainty.

    Reliability is "when you said 70%, did it happen 70% of the time" -
    the thing everybody calls calibration, and the only part a
    recalibration step can fix.  Resolution is "did you separate the cases
    at all", and it is the part that carries the information.  A forecaster
    can be perfectly reliable and useless: report the base rate every time
    and reliability is zero, resolution is also zero.
    """
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(q, edges[1:-1]), 0, bins - 1)
    n = len(y)
    ybar = float(y.mean())
    rel = res = 0.0
    for k in range(bins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue
        qk, yk = float(q[m].mean()), float(y[m].mean())
        rel += nk * (qk - yk) ** 2
        res += nk * (yk - ybar) ** 2
    rel, res = rel / n, res / n
    unc = ybar * (1 - ybar)
    return {"brier": float(np.mean((q - y) ** 2)), "reliability": rel,
            "resolution": res, "uncertainty": unc, "check": rel - res + unc}


def decompositions(bins: int = 10) -> Dict[str, Dict[str, float]]:
    outcomes, reports = simulate()
    return {f.name: murphy(reports[f.name], outcomes, bins) for f in FORECASTERS}


def reliability_curve(name: str, bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean report, observed rate, count) per bin - the calibration plot."""
    outcomes, reports = simulate()
    q = reports[name]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(q, edges[1:-1]), 0, bins - 1)
    xs, ys, ns = [], [], []
    for k in range(bins):
        m = idx == k
        if m.sum() == 0:
            continue
        xs.append(q[m].mean())
        ys.append(outcomes[m].mean())
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def reliability_beats_resolution() -> List[Tuple[str, str]]:
    """Pairs where the MORE reliable forecaster has the WORSE Brier score.

    If this list is non-empty, "improve your calibration" is not the same
    instruction as "become a better forecaster".
    """
    d = decompositions()
    names = list(d)
    out = []
    for a in names:
        for b in names:
            if a == b:
                continue
            if d[a]["reliability"] < d[b]["reliability"] and d[a]["brier"] > d[b]["brier"]:
                out.append((a, b))
    return out


# --------------------------------------------------------------------------
# Resulting - judging a decision by its outcome
# --------------------------------------------------------------------------


def resulting(n: int = 20000, p_win: float = 0.62,
              win: float = 100.0, lose: float = -100.0) -> Dict[str, float]:
    """A positive-EV decision, reviewed by its outcome.

    Every decision here is *correct* - the expected value is positive and
    it was taken with full knowledge of the odds. An outcome-based review
    marks every loss as a bad decision.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    won = rng.random(n) < p_win
    ev = p_win * win + (1 - p_win) * lose
    return {
        "expected_value": ev,
        "p_win": p_win,
        "share_judged_bad": float(1 - won.mean()),
        "n": n,
    }


def resulting_portfolio(n_decisions: int = 200) -> Dict[str, float]:
    """A realistic mixed portfolio: some decisions good, some genuinely bad.

    Then review every one of them purely on outcome, and count how often
    that review reaches the wrong verdict.
    """
    rng = np.random.default_rng(RNG_SEED + 2)
    # Each decision has a win probability and a symmetric payoff; a decision
    # is GOOD if its expected value is positive, i.e. p > 0.5.
    p = rng.uniform(0.25, 0.75, size=n_decisions)
    good = p > 0.5
    won = rng.random(n_decisions) < p
    verdict_good = won  # outcome-based review
    wrong = verdict_good != good
    return {
        "n": n_decisions,
        "truly_good": int(good.sum()),
        "misjudged": int(wrong.sum()),
        "misjudged_rate": float(wrong.mean()),
        "good_called_bad": int((good & ~verdict_good).sum()),
        "bad_called_good": int((~good & verdict_good).sum()),
    }


# --------------------------------------------------------------------------
# How many decisions before the log says anything
# --------------------------------------------------------------------------


def decisions_needed(a: str = "noisy_expert", b: str = "overconfident",
                     alpha: float = 0.05, power: float = 0.8) -> Dict[str, float]:
    """Paired sample size to tell forecaster `a` from forecaster `b` by Brier.

    Uses the observed per-decision Brier difference: its mean and sd give
    a paired-design n. This is the number that decides whether a decision
    log can conclude anything at all.
    """
    from scipy import stats

    outcomes, reports = simulate()
    da = brier(reports[a], outcomes)
    db = brier(reports[b], outcomes)
    diff = da - db
    mean, sd = float(diff.mean()), float(diff.std(ddof=1))
    if sd == 0:
        return {"n_required": 0.0, "mean_diff": mean, "sd_diff": sd}
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    n = ((z_a + z_b) * sd / abs(mean)) ** 2
    return {"n_required": float(math.ceil(n)), "mean_diff": mean, "sd_diff": sd,
            "brier_a": float(da.mean()), "brier_b": float(db.mean())}


def cheapest_and_dearest_comparison() -> Tuple[Tuple[str, str], float, Tuple[str, str], float]:
    """The easiest and the hardest forecaster pair to tell apart."""
    m = power_matrix()
    lo = min(m, key=m.get)
    hi = max(m, key=m.get)
    return lo, m[lo], hi, m[hi]


def power_matrix(alpha: float = 0.05, power: float = 0.8) -> Dict[Tuple[str, str], float]:
    names = [f.name for f in FORECASTERS]
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[(a, b)] = decisions_needed(a, b, alpha, power)["n_required"]
    return out


# --------------------------------------------------------------------------
# The record, and whether it can be scored at all
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """One decision, as somebody actually wrote it down."""

    id: str
    decision: str
    claim: str
    probability: Optional[float] = None
    resolve_by: Optional[str] = None
    metric: Optional[str] = None
    threshold: Optional[str] = None


#: Illustrative, not sampled. The rate this produces is a property of this
#: corpus; the LINTER is the reusable part.
RECORDS: Tuple[Record, ...] = (
    Record("D-001", "Migrate the warehouse to the new engine",
           "query costs will fall", None, None, None, None),
    Record("D-002", "Migrate the warehouse to the new engine",
           "median dashboard query latency drops below 4s", 0.7, "2026-12-01",
           "p50_query_latency_s", "< 4"),
    Record("D-003", "Hire a second analytics engineer",
           "the team will move faster", None, None, None, None),
    Record("D-004", "Hire a second analytics engineer",
           "dbt model lead time falls under 5 days", 0.55, "2027-02-01",
           "model_lead_time_days", "< 5"),
    Record("D-005", "Adopt the vendor's managed catalog",
           "this is the right strategic direction", None, None, None, None),
    Record("D-006", "Adopt the vendor's managed catalog",
           "60% of certified tables have an owner recorded", 0.4, "2026-11-15",
           "pct_certified_with_owner", ">= 0.6"),
    Record("D-007", "Ship the pricing change",
           "revenue will improve", 0.8, None, None, None),
    Record("D-008", "Ship the pricing change",
           "net revenue retention rises by 2pp", 0.35, "2027-01-31",
           "nrr_delta_pp", ">= 2"),
    Record("D-009", "Rewrite the ingestion service in Go",
           "it will be more maintainable", None, "2026-10-01", None, None),
    Record("D-010", "Rewrite the ingestion service in Go",
           "on-call pages from ingestion fall below 2 per month", 0.5, "2027-03-01",
           "ingestion_pages_per_month", "< 2"),
    Record("D-011", "Freeze the schema for Q4",
           "fewer incidents", None, None, "incidents", None),
    Record("D-012", "Freeze the schema for Q4",
           "zero P1 data incidents attributable to schema change", 0.65, "2027-01-01",
           "p1_schema_incidents", "== 0"),
    Record("D-013", "Buy rather than build the CDC pipeline",
           "cheaper in the long run", None, None, None, None),
    Record("D-014", "Buy rather than build the CDC pipeline",
           "total 12-month cost under 90k", 0.45, "2027-08-01",
           "twelve_month_cost_usd", "< 90000"),
    Record("D-015", "Roll out the LLM support assistant",
           "customers will love it", 0.9, None, None, None),
    Record("D-016", "Roll out the LLM support assistant",
           "deflection rate above 25% with CSAT not down more than 2pp", 0.3,
           "2027-04-01", "deflection_rate", ">= 0.25"),
    Record("D-017", "Deprecate the legacy dashboards",
           "nobody is using them", None, "2026-09-30", None, None),
    Record("D-018", "Deprecate the legacy dashboards",
           "fewer than 5 unique viewers in the 90 days before cutoff", 0.75,
           "2026-09-30", "unique_viewers_90d", "< 5"),
    Record("D-019", "Standardise on one metrics layer",
           "one number, one definition", None, None, None, None),
    Record("D-020", "Standardise on one metrics layer",
           "revenue reported identically by all 3 certified dashboards", 0.6,
           "2027-01-15", "revenue_definition_variants", "== 1"),
)

#: A resolution date must be a real ISO date, not "soon" or "Q4".
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def lint(record: Record) -> Dict[str, bool]:
    """The four things a record needs before it can ever be scored."""
    return {
        "has_probability": record.probability is not None
        and 0.0 < record.probability < 1.0,
        "has_resolution_date": record.resolve_by is not None
        and bool(_ISO.match(record.resolve_by)),
        "has_metric": bool(record.metric),
        "has_threshold": bool(record.threshold),
    }


def resolvable(record: Record) -> bool:
    return all(lint(record).values())


def resolvability_report() -> Dict[str, object]:
    checks = [lint(r) for r in RECORDS]
    n = len(RECORDS)
    return {
        "n": n,
        "resolvable": sum(1 for r in RECORDS if resolvable(r)),
        "per_field": {
            field_name: sum(1 for c in checks if c[field_name])
            for field_name in ("has_probability", "has_resolution_date",
                               "has_metric", "has_threshold")
        },
        "unscoreable": [r.id for r in RECORDS if not resolvable(r)],
    }
