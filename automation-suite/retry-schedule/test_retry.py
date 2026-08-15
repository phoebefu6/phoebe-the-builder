"""Tests for retry.py.

The ones that matter are at the bottom: an independent, obviously-correct
arrival counter cross-checked against the simulator's histogram, a Monte Carlo
check that the closed-form load floor matches the process it claims to predict,
and the two properties every backoff is assumed to have and only some of it
does.

Run: python3 -m pytest test_retry.py -q
"""

from __future__ import annotations

import math
import random

import pytest

import retry as R

# ---------------------------------------------------------------------------
# Policy behaviour - the published algorithms, pinned
# ---------------------------------------------------------------------------


def test_no_jitter_is_capped_exponential():
    s = R.Schedule("no_jitter", base=0.1, cap=20.0, max_attempts=10)
    d = s.delays(random.Random(0))
    assert d[:4] == [0.1, 0.2, 0.4, 0.8]
    assert d[-2:] == [20.0, 20.0]  # clamped
    assert sum(d) == pytest.approx(65.5)


def test_no_jitter_is_identical_for_every_client():
    # The whole problem in one assertion: the seed is irrelevant.
    s = R.Schedule("no_jitter", 0.1, 20.0, 8)
    assert s.delays(random.Random(1)) == s.delays(random.Random(99999))


def test_jittered_policies_are_not_identical_across_clients():
    for name in ("full_jitter", "equal_jitter", "decorrelated_jitter"):
        s = R.Schedule(name, 0.1, 20.0, 8)
        assert s.delays(random.Random(1)) != s.delays(random.Random(2)), name


def test_full_jitter_stays_inside_its_window():
    s = R.Schedule("full_jitter", 0.1, 20.0, 12)
    for _ in range(200):
        for i, d in enumerate(s.delays(random.Random(random.randrange(1 << 20)))):
            assert 0.0 <= d <= s.window(i) + 1e-12


def test_equal_jitter_never_goes_below_half_its_window():
    s = R.Schedule("equal_jitter", 0.1, 20.0, 12)
    for seed in range(50):
        for i, d in enumerate(s.delays(random.Random(seed))):
            assert s.window(i) / 2 - 1e-12 <= d <= s.window(i) + 1e-12


def test_decorrelated_jitter_can_decrease():
    # Documented behaviour, not a defect - it is what decorrelates two clients.
    s = R.Schedule("decorrelated_jitter", 0.1, 20.0, 20)
    drops = 0
    for seed in range(40):
        d = s.delays(random.Random(seed))
        drops += sum(1 for i in range(1, len(d)) if d[i] < d[i - 1])
    assert drops > 0


def test_decorrelated_jitter_respects_the_cap():
    s = R.Schedule("decorrelated_jitter", 0.1, 5.0, 40)
    for seed in range(30):
        assert max(s.delays(random.Random(seed))) <= 5.0 + 1e-12


# ---------------------------------------------------------------------------
# Closed forms vs. sampling - the analysis has to match the process
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", R.POLICY_ORDER)
def test_expected_total_matches_monte_carlo(name):
    s = R.Schedule(name, 0.1, 20.0, 10)
    rng = random.Random(4)
    sampled = sum(sum(s.delays(rng)) for _ in range(6000)) / 6000
    assert sampled == pytest.approx(s.expected_total(), rel=0.06)


@pytest.mark.parametrize("name", R.POLICY_ORDER)
def test_worst_case_total_is_an_upper_bound(name):
    s = R.Schedule(name, 0.1, 20.0, 10)
    worst = s.worst_case_total()
    for seed in range(300):
        assert sum(s.delays(random.Random(seed))) <= worst + 1e-9


def test_full_jitter_expectation_is_half_the_unjittered_schedule():
    a = R.Schedule("no_jitter", 0.1, 20.0, 10)
    b = R.Schedule("full_jitter", 0.1, 20.0, 10)
    assert b.expected_total() == pytest.approx(a.expected_total() / 2)


def test_median_and_mean_agree_for_ladder_policies_and_not_for_the_walk():
    # Sums of independent uniforms concentrate, so mean ~ median. The
    # decorrelated walk compounds, so it does not - which is why the finding
    # only fires for one policy.
    for name in ("full_jitter", "equal_jitter"):
        med, mean = R.sampled_totals(R.Schedule(name, 0.1, 20.0, 10), n=4000, seed=2)
        assert med / mean > 0.95, name
    med, mean = R.sampled_totals(
        R.Schedule("decorrelated_jitter", 0.1, 20.0, 10), n=4000, seed=2)
    assert med / mean < 0.75


# ---------------------------------------------------------------------------
# The load floor - a closed form is only worth having if it predicts
# ---------------------------------------------------------------------------


def _independent_arrival_count(schedule, fleet, lo, hi, seed):
    """Count arrivals in [lo, hi) with no simulator, no shedding, no state.

    Deliberately naive: build every client's arrival list independently and
    count. If this disagrees with `steady_state_rate`, the closed form is wrong.
    """
    rng = random.Random(seed)
    n = 0
    for _ in range(fleet):
        for t in schedule.arrivals(rng):
            if lo <= t < hi:
                n += 1
    return n


@pytest.mark.parametrize("cap", [10.0, 20.0, 40.0])
def test_steady_state_rate_predicts_the_arrival_process(cap):
    fleet = 400
    s = R.Schedule("full_jitter", 0.1, cap, max_attempts=60)
    # Start the window well after every client has reached the cap.
    lo, hi = 4 * cap, 8 * cap
    measured = _independent_arrival_count(s, fleet, lo, hi, seed=5) / (hi - lo)
    assert measured == pytest.approx(s.steady_state_rate(fleet), rel=0.15)


def test_the_floor_does_not_decay_with_time():
    s = R.Schedule("full_jitter", 0.1, 20.0, max_attempts=80)
    fleet = 400
    early = _independent_arrival_count(s, fleet, 100, 160, seed=6) / 60
    late = _independent_arrival_count(s, fleet, 400, 460, seed=6) / 60
    assert late == pytest.approx(early, rel=0.2)


# ---------------------------------------------------------------------------
# Simulator cross-check - a second, obviously-correct implementation
# ---------------------------------------------------------------------------


SCALE = 10 ** 9  # simulate() rounds arrival times to 9 decimal places


def _exact_histogram(arrivals, width, since, nbuckets):
    """Reference counter in integer arithmetic. No float grid, no edge cases.

    The obvious reference - loop over buckets, test `i*width <= t < (i+1)*width`
    - is itself wrong: `i * width` does not tile the real line, so adjacent
    buckets overlap by an ulp and a value sitting exactly on an edge matches
    two of them. Scaling to integers removes the grid entirely, which is the
    only way this is a genuinely independent check rather than a restatement.
    """
    w = round(width * SCALE)
    s0 = round(since * SCALE)
    counts = [0] * nbuckets
    for t in arrivals:
        ti = round(t * SCALE)
        if ti < s0:
            continue
        i = (ti - s0) // w
        if 0 <= i < nbuckets:
            counts[i] += 1
    return counts


def test_bucket_index_matches_integer_arithmetic():
    # The specific values that broke the first two implementations.
    for t, w in ((1.5, 0.1), (20.0, 0.1), (0.7, 0.1), (2.9, 0.1),
                 (60.0, 0.5), (13.0, 2.0), (0.0, 0.1)):
        assert R.bucket_index(t, w) == round(t * SCALE) // round(w * SCALE), (t, w)


def test_bucket_index_is_monotone_and_partitions():
    ts = [i * 0.017 for i in range(3000)]
    idx = [R.bucket_index(t, 0.1) for t in ts]
    assert idx == sorted(idx)
    assert all(R.bucket_index(t, 0.1) == round(t * SCALE) // 10 ** 8 for t in ts)


@pytest.mark.parametrize("name", R.POLICY_ORDER)
def test_histogram_matches_brute_force(name):
    s = R.Schedule(name, 0.1, 20.0, 8)
    sim = R.simulate(s, fleet=120, outage_s=10.0, capacity_rps=30.0, seed=3)
    for width, since in ((0.1, 0.0), (1.0, 0.0), (0.5, 10.0), (2.0, 10.0)):
        edges, fast = sim.histogram(width=width, since=since)
        slow = _exact_histogram(sim.arrivals, width, since, len(fast))
        assert fast == slow, (name, width, since)


@pytest.mark.parametrize("name", R.POLICY_ORDER)
def test_every_request_is_accounted_for(name):
    s = R.Schedule(name, 0.1, 20.0, 8)
    sim = R.simulate(s, fleet=200, outage_s=10.0, capacity_rps=40.0, seed=8)
    assert len(sim.admitted) + len(sim.rejected) == len(sim.arrivals)
    assert sim.succeeded + sim.gave_up == sim.fleet
    assert sim.succeeded == len(sim.admitted)


@pytest.mark.parametrize("name", R.POLICY_ORDER)
def test_no_client_exceeds_its_attempt_budget(name):
    s = R.Schedule(name, 0.1, 20.0, 6)
    sim = R.simulate(s, fleet=150, outage_s=12.0, capacity_rps=20.0, seed=2)
    assert len(sim.arrivals) <= sim.fleet * s.max_attempts


def test_simulation_is_deterministic_given_a_seed():
    s = R.Schedule("full_jitter", 0.1, 20.0, 10)
    a = R.simulate(s, fleet=200, seed=13)
    b = R.simulate(s, fleet=200, seed=13)
    assert a.arrivals == b.arrivals
    assert (a.succeeded, a.gave_up) == (b.succeeded, b.gave_up)


def test_nothing_is_admitted_during_the_outage():
    s = R.Schedule("full_jitter", 0.1, 20.0, 10)
    sim = R.simulate(s, fleet=300, outage_s=15.0, capacity_rps=50.0, seed=1)
    assert all(t >= 15.0 for t in sim.admitted)


def test_zero_capacity_admits_nothing():
    # The isolation case: a positive capacity always admits at least one
    # request per bucket, but zero means zero.
    s = R.Schedule("full_jitter", 0.1, 20.0, 10)
    sim = R.simulate(s, fleet=100, outage_s=5.0, capacity_rps=0.0, seed=1)
    assert sim.admitted == []
    assert sim.gave_up == 100


# ---------------------------------------------------------------------------
# The headline claims
# ---------------------------------------------------------------------------


def test_deterministic_policies_put_the_whole_fleet_in_one_bucket():
    s = R.Schedule("no_jitter", 0.1, 20.0, 10)
    sim = R.simulate(s, fleet=500, outage_s=20.0, capacity_rps=50.0, seed=7)
    _, counts = sim.histogram(width=0.1, since=20.0)
    assert max(counts) == 500


def test_jitter_does_not_reduce_total_work_much_but_flattens_the_peak():
    a = R.simulate(R.Schedule("no_jitter", 0.1, 20.0, 10), fleet=500,
                   outage_s=20.0, capacity_rps=50.0, seed=7)
    b = R.simulate(R.Schedule("full_jitter", 0.1, 20.0, 10), fleet=500,
                   outage_s=20.0, capacity_rps=50.0, seed=7)
    assert b.recovery_peak_rps() < a.recovery_peak_rps() / 5
    # total work is the same order - backoff decides that, not jitter
    assert 0.5 < b.total_requests() / a.total_requests() < 2.0


def test_the_lowest_peak_is_not_the_best_outcome():
    """The result the whole project exists for."""
    full = R.simulate(R.Schedule("full_jitter", 0.1, 20.0, 10), fleet=500,
                      outage_s=20.0, capacity_rps=50.0, seed=7)
    equal = R.simulate(R.Schedule("equal_jitter", 0.1, 20.0, 10), fleet=500,
                       outage_s=20.0, capacity_rps=50.0, seed=7)
    assert full.recovery_peak_rps() < equal.recovery_peak_rps()
    assert full.gave_up > equal.gave_up


def test_verdicts_are_three_valued_and_ordered():
    cfg = dict(fleet=500, outage_s=20.0, capacity_rps=50.0)
    got = {}
    for name in R.POLICY_ORDER:
        s = R.Schedule(name, 0.1, 20.0, 10)
        v, _ = R.audit(s, **cfg)
        got[name] = v
    assert got["no_jitter"] is R.Verdict.HERDING
    assert got["fixed_interval"] is R.Verdict.HERDING
    assert got["full_jitter"] is R.Verdict.DISPERSED


def test_dispersed_does_not_mean_healthy():
    # A dispersed verdict is a statement about the arrival process, not about
    # whether the clients recovered. Both are reported; they disagree here.
    s = R.Schedule("decorrelated_jitter", 0.1, 20.0, 10)
    v, findings = R.audit(s, fleet=500, outage_s=20.0, capacity_rps=50.0)
    assert v is R.Verdict.DISPERSED
    assert any(f.code == "CLIENTS_GAVE_UP" for f in findings)


def test_short_budget_findings_are_reachable():
    # Neither fires in the headline run, so pin a config where they do -
    # a finding nothing can trigger is not a finding.
    _, f6 = R.audit(R.Schedule("full_jitter", 0.1, 20.0, 6),
                    fleet=500, outage_s=20.0, capacity_rps=50.0)
    assert "BUDGET_SHORTER_THAN_OUTAGE" in {x.code for x in f6}
    _, f8 = R.audit(R.Schedule("full_jitter", 0.1, 20.0, 8),
                    fleet=500, outage_s=20.0, capacity_rps=50.0)
    assert "JITTER_SHORTENS_COVERAGE" in {x.code for x in f8}


def test_budget_underuse_is_reachable():
    _, f = R.audit(R.Schedule("full_jitter", 0.1, 2.0, 4), fleet=10,
                   outage_s=0.5, capacity_rps=500.0, deadline_s=120.0)
    assert "BUDGET_UNDERUSE" in {x.code for x in f}


def test_fixed_interval_wastes_nothing_only_by_coincidence():
    # Its interval equals the outage, so the first retry lands on recovery.
    # Move either number and the coincidence goes away.
    s = R.Schedule("fixed_interval", 0.1, 20.0, 10)
    exact = R.simulate(s, fleet=100, outage_s=20.0, capacity_rps=50.0, seed=7)
    assert exact.wasted_requests() == 0
    off = R.simulate(s, fleet=100, outage_s=25.0, capacity_rps=50.0, seed=7)
    assert off.wasted_requests() > 0


def test_amplification_is_the_product():
    assert R.amplification([3, 3, 3]) == 27
    assert R.amplification([3, 3, 3, 2]) == 54
    assert R.amplification([]) == 1
    assert R.amplification([0, 5]) == 5  # a layer that does not retry is 1x


def test_quantise_collapses_narrow_jitter_windows():
    s = R.Schedule("full_jitter", 0.1, 20.0, 10)
    d = s.delays(random.Random(5))
    q = R.quantise(d, 1.0)
    narrow = [i for i in range(10) if s.window(i) < 1.0]
    # Every delay whose whole window fits inside one tick lands on the same value.
    assert len({q[i] for i in narrow}) == 1


def test_audit_reports_the_alignment_and_amplification_findings():
    s = R.Schedule("full_jitter", 0.1, 20.0, 10)
    _, findings = R.audit(s, fleet=500, outage_s=20.0, capacity_rps=50.0,
                          deadline_s=30.0, nested_layers=[3, 3], tick_s=1.0)
    codes = {f.code for f in findings}
    assert "CLOCK_ALIGNMENT" in codes
    assert "RETRY_AMPLIFICATION" in codes
    assert "CAP_PLATEAU" in codes


def test_reducing_the_fleet_is_what_actually_lowers_the_floor():
    s = R.Schedule("equal_jitter", 0.1, 20.0, 10)
    big = R.simulate(s, fleet=500, outage_s=20.0, capacity_rps=50.0, seed=7)
    small = R.simulate(s, fleet=100, outage_s=20.0, capacity_rps=50.0, seed=7)
    assert small.recovery_peak_rps() < big.recovery_peak_rps()
    assert s.steady_state_rate(100) == pytest.approx(s.steady_state_rate(500) / 5)


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def test_single_attempt_schedule():
    s = R.Schedule("full_jitter", 0.1, 20.0, 1)
    sim = R.simulate(s, fleet=10, outage_s=100.0, capacity_rps=50.0, seed=1)
    assert sim.total_requests() == 10
    assert sim.gave_up == 10


def test_zero_fleet_is_not_a_crash():
    sim = R.simulate(R.Schedule("full_jitter"), fleet=0)
    assert sim.total_requests() == 0
    assert sim.peak_rps() == 0.0
    assert sim.completion_time() is None


def test_cap_below_base_still_caps():
    s = R.Schedule("no_jitter", base=5.0, cap=1.0, max_attempts=4)
    assert s.delays(random.Random(0)) == [1.0, 1.0, 1.0, 1.0]


def test_cap_reached_at_is_none_when_the_ladder_never_reaches_it():
    s = R.Schedule("no_jitter", base=0.1, cap=1e6, max_attempts=5)
    assert s.cap_reached_at() is None


def test_horizon_truncates_rather_than_hangs():
    s = R.Schedule("fixed_interval", 0.1, 60.0, 100)
    sim = R.simulate(s, fleet=20, outage_s=10.0, capacity_rps=0.0,
                     seed=1, horizon_s=120.0)
    assert all(t <= 120.0 for t in sim.arrivals)
