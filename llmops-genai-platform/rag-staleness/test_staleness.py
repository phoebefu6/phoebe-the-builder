"""Engine tests - closed form against numeric integration and raw simulation, and the claims the README makes."""

from __future__ import annotations

import math

import numpy as np
import pytest
import staleness as T


def test_closed_form_matches_numeric_integration() -> None:
    for mu, period in ((0.04, 30.0), (0.0008, 90.0), (0.3, 1.0)):
        xs = np.linspace(0, period, 200_001)
        num = float(np.trapz(1 - np.exp(-mu * xs), xs) / period)
        assert math.isclose(T.stale(mu, period), num, rel_tol=1e-6)


def test_small_argument_is_accurate() -> None:
    """1 - (1 - e^-x)/x loses every digit near x = 0; expm1 keeps them. stale ~ x/2 - x^2/6."""
    for x in (1e-10, 1e-8, 1e-5):
        assert math.isclose(T.stale(x, 1.0), x / 2 - x * x / 6, rel_tol=1e-6)
    assert T.stale(0.0, 1.0) == 0.0


def test_stale_is_monotone_in_period_and_rate() -> None:
    xs = [T.stale(0.02, t) for t in (1, 7, 30, 90, 365)]
    assert xs == sorted(xs)
    ys = [T.stale(m, 30) for m in (0.001, 0.01, 0.1)]
    assert ys == sorted(ys)


def test_simulation_agrees_with_closed_form() -> None:
    cal = T.calibrate(reps=40_000)
    assert sum(r["inside_99"] for r in cal) >= len(cal) - 1


def test_simulation_check_can_fail() -> None:
    """v2b's simulated answer staleness must land outside v1's exact value, or the check proves nothing."""
    sim = T.simulate(T.POLICIES["v2b tiered 1/7/90"], reps=20_000)
    k, n = sim["answer_staleness"]
    lo, hi = T.wilson(k, n)
    assert not (lo <= T.evaluate(T.POLICIES["v1 monthly full"])["answer_staleness"] <= hi)


def test_three_numbers_are_ordered_as_the_readme_says() -> None:
    e = T.evaluate(T.POLICIES["v1 monthly full"])
    assert e["hash_freshness"] < e["fact_freshness"]
    assert e["answer_staleness"] > 1 - e["fact_freshness"]


def test_ranking_flips_between_metrics() -> None:
    wk, ti = T.evaluate(T.POLICIES["v2a weekly full"]), T.evaluate(T.POLICIES["v2b tiered 1/7/90"])
    assert wk["hash_freshness"] > ti["hash_freshness"]
    assert wk["answer_staleness"] > 2 * ti["answer_staleness"]
    assert wk["reindex_per_day"] > ti["reindex_per_day"]


def test_cycle_ends_above_the_average_and_starts_at_zero() -> None:
    cyc = T.cycle(T.POLICIES["v1 monthly full"])
    avg = T.evaluate(T.POLICIES["v1 monthly full"])["answer_staleness"]
    assert cyc[0]["answer_staleness"] == 0.0
    assert cyc[30]["answer_staleness"] > avg > cyc[7]["answer_staleness"]


def test_archive_docs_raise_freshness_and_leave_answers_alone() -> None:
    base = T.evaluate(T.POLICIES["v1 monthly full"])
    cls, pol = T.with_archive(1000, T.POLICIES["v1 monthly full"])
    e = T.evaluate(pol, cls)
    assert e["hash_freshness"] > base["hash_freshness"] + 0.03
    assert math.isclose(e["answer_staleness"], base["answer_staleness"])


def test_audit_sample_estimators_name_different_numbers() -> None:
    e = T.evaluate(T.POLICIES["v1 monthly full"])
    assert math.isclose(T.audit_sample(T.POLICIES["v1 monthly full"], "docs"), 1 - e["fact_freshness"])
    assert math.isclose(T.audit_sample(T.POLICIES["v1 monthly full"], "queries"), e["answer_staleness"])


def test_similarity_gate_is_blind() -> None:
    sg = T.similarity_gap()
    assert sg["stale_passed_by_gate_keeping_all_fresh"] == 1.0
    assert sg["max_abs_gap"] < 0.05
    assert sg["stale_at_least_as_similar"] >= sg["n"] - 1


def test_query_never_contains_the_value() -> None:
    for f in T.FACTS:
        q = set(T.tokens(T.query(f)))
        assert not (q & set(T.tokens(f[2]))) and not (q & set(T.tokens(f[3])))


def test_parse_classes_normalises_and_refuses() -> None:
    cls = T.parse_classes("a, 10, 14, 0.6, 2\nb, 20, 30, 0.5, 2")
    assert [c["query_share"] for c in cls] == [0.5, 0.5]
    for bad in ("a, 10, 14", "a, x, 14, 0.6, 1", "a, 10, 14, 1.5, 1", "a, 10, 14, .5, 0", "a,1,1,1,1\na,1,1,1,1", ""):
        with pytest.raises(ValueError):
            T.parse_classes(bad)


def test_parse_policy_forms() -> None:
    cls = T.parse_classes("hot, 10, 14, 0.6, 1\ncold, 10, 300, 0.3, 1")
    assert T.parse_policy("30", cls) == {"hot": 30.0, "cold": 30.0}
    assert T.parse_policy("hot=1, cold=90", cls) == {"hot": 1.0, "cold": 90.0}
    for bad in ("0", "hot=1", "hot=x, cold=1", "hot 1, cold 2", "hot=-1, cold=1"):
        with pytest.raises(ValueError):
            T.parse_policy(bad, cls)
