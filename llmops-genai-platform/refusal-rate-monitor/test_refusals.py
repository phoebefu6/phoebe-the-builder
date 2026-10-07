"""Engine tests - closed form against raw simulation, exact tails against the cumulative sum, and the README's claims."""

from __future__ import annotations

import math

import numpy as np
import pytest
import refusals as R

MIX = R.mix_of()
KW = R.DETECTORS["keyword list (written on v1)"]
V1, V2, V3 = (R.VERSIONS[k] for k in R.VERSIONS)


def test_mix_normalises_and_overrides() -> None:
    assert math.isclose(sum(MIX.values()), 1.0)
    m = R.mix_of(override={"harmful": 0.5})
    assert math.isclose(sum(m.values()), 1.0) and m["harmful"] > MIX["harmful"]


def test_true_rates_are_a_weighted_sum() -> None:
    tr = R.true_rates(V1, MIX)
    assert math.isclose(tr["refusal_rate"], sum(MIX[c] * V1["refuse"][c] for c in MIX))
    assert math.isclose(tr["under_refusal"], 1 - V1["refuse"]["harmful"])
    assert 0 < tr["over_refusal"] < tr["per_class"]["sensitive"]


def test_recall_is_a_property_of_the_pair() -> None:
    assert R.recall(KW, V1) > 0.8 and R.recall(KW, V2) < 0.25
    assert R.recall(R.DETECTORS["LLM judge"], V1) == R.recall(R.DETECTORS["LLM judge"], V2)
    assert math.isclose(R.recall(R.DETECTORS["keyword list, extended"], V2), 1.0)


def test_upgrade_sign_is_wrong_on_the_keyword_detector() -> None:
    assert R.true_rates(V2, MIX)["refusal_rate"] > R.true_rates(V1, MIX)["refusal_rate"]
    assert R.measured_rate(V2, MIX, KW) < R.measured_rate(V1, MIX, KW)
    for d in ("keyword list, extended", "LLM judge"):
        dd = R.DETECTORS[d]
        assert R.measured_rate(V2, MIX, dd) > R.measured_rate(V1, MIX, dd)


def test_campaign_moves_the_rate_and_not_the_errors() -> None:
    cm = R.mix_of(override=R.CAMPAIGN_MIX)
    a, b = R.true_rates(V1, MIX), R.true_rates(V1, cm)
    assert b["refusal_rate"] > 1.8 * a["refusal_rate"]
    assert abs(b["over_refusal"] - a["over_refusal"]) < 0.001
    assert b["under_refusal"] == a["under_refusal"]


def test_flat_twin_holds_the_aggregate_and_moves_under_refusal_the_other_way() -> None:
    tw = R.flat_aggregate_twin(V1, MIX, 2.0)
    a, b = R.true_rates(V1, MIX), R.true_rates(tw, MIX)
    assert math.isclose(a["refusal_rate"], b["refusal_rate"])
    assert math.isclose(b["over_refusal"], 2 * a["over_refusal"])
    assert b["under_refusal"] > 5 * a["under_refusal"]
    assert 0 <= tw["refuse"]["harmful"] <= 1


def test_tails_match_cumulative_sum_and_edges() -> None:
    n, p = 40, 0.3
    pmf = np.exp(R.log_pmf(np.arange(n + 1), n, p))
    assert math.isclose(pmf.sum(), 1.0, rel_tol=1e-12)
    for k in (0, 5, 12, 40, 41):
        assert math.isclose(R.upper_tail(k, n, p), pmf[k:].sum() if k <= n else 0.0, abs_tol=1e-12)
    assert R.upper_tail(0, n, 0.0) == 1.0 and R.upper_tail(1, n, 0.0) == 0.0 and R.upper_tail(n, n, 1.0) == 1.0


def test_critical_k_controls_size() -> None:
    for n, p0 in ((100, 0.02), (300, 0.376), (20_000, 0.0614)):
        k = R.critical_k(n, p0, 0.05)
        assert R.upper_tail(k, n, p0) <= 0.05 < R.upper_tail(k - 1, n, p0)


def test_refusal_sample_beats_traffic_sample() -> None:
    ad = R.audit_designs(V1, V2, MIX, [100, 1000], KW)
    assert ad["refusals"][0]["power"] > 0.95 > 0.5 > ad["traffic"][0]["power"]
    assert ad["traffic"][1]["power"] > 0.95
    assert all(r["size"] <= 0.05 for r in ad["traffic"] + ad["refusals"])


def test_detected_legit_share_is_inflated_by_false_positives() -> None:
    ad = R.audit_designs(V1, V2, MIX, [100], KW)
    assert ad["refusals"][0]["p0"] > R.true_rates(V1, MIX)["legit_share_of_refusals"]
    assert ad["refusals"][0]["p1"] > R.true_rates(V2, MIX)["legit_share_of_refusals"]


def test_monitor_alarms_where_the_readme_says() -> None:
    mon = {r["week"]: r for r in R.monitor(KW)}
    assert mon[1]["p_alarm_high"] < 0.01 and mon[1]["p_audit_alarm"] < 0.01
    assert mon[9]["p_alarm_high"] > 0.99 and mon[9]["p_audit_alarm"] < 0.01
    assert mon[19]["p_alarm_high"] < 0.01 and mon[19]["p_alarm_low"] > 0.99 and mon[19]["p_audit_alarm"] > 0.99
    assert mon[19]["measured"] < mon[1]["measured"] < mon[1]["true_rate"] < mon[19]["true_rate"]


def test_simulation_agrees_with_closed_form() -> None:
    cal = R.calibrate(reps=40_000)
    assert sum(r["inside_99"] for r in cal) >= len(cal) - 1


def test_simulation_check_can_fail() -> None:
    """v2's simulated measured rate must land outside v1's exact measured rate, or the check proves nothing."""
    sim = R.simulate(V2, MIX, KW, reps=20_000)
    k, n = sim["measured"]
    lo, hi = R.wilson(k, n)
    assert not (lo <= R.measured_rate(V1, MIX, KW) <= hi)


def test_parsers() -> None:
    v = R.parse_version("0.006, 0.12, 0.90")
    assert v["refuse"] == {"benign": 0.006, "sensitive": 0.12, "harmful": 0.90}
    for bad in ("0.1, 0.2", "a, 0.2, 0.3", "1.5, 0.2, 0.3"):
        with pytest.raises(ValueError):
            R.parse_version(bad)
    m = R.parse_mix("80, 15, 5")
    assert math.isclose(m["harmful"], 0.05)
    for bad in ("1, 2", "x, 1, 1", "1, 1, 0", "-1, 1, 1"):
        with pytest.raises(ValueError):
            R.parse_mix(bad)
