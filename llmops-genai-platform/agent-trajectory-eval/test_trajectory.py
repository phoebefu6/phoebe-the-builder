"""Engine tests. Each scorer is checked on hand-built paths, and the enumeration on its own arithmetic."""

from __future__ import annotations

import trajectory as T

OID, W = T.ORDER, T.WRONG
REF = [("lookup_order", OID), ("check_policy", None), ("issue_refund", OID)]


def test_reference_path_is_valid_and_passes_everything() -> None:
    assert T.violations(REF, True) == []
    assert all(T.score(REF, True, True).values())
    assert T.violations(REF[:2], False) == [] and all(T.score(REF[:2], False, True).values())


def test_other_order_is_valid_but_strict_fails_it() -> None:
    p = [REF[1], REF[0], REF[2]]
    assert T.violations(p, True) == []
    s = T.score(p, True, True)
    assert not s["strict"] and s["unordered"] and s["superset"]


def test_each_invariant_fires_on_its_own_defect() -> None:
    assert T.violations([REF[0], REF[2]], True) == ["unsupported decision", "acted before reading"]
    assert "acted before reading" in T.violations([("issue_refund", OID), REF[0], REF[1]], True)
    assert T.violations(REF + [REF[2]], True) == ["duplicate action"]
    assert T.violations(REF[:2] + [("issue_refund", W)], True) == ["wrong argument"]
    assert T.violations(REF, False) == ["wrong action"]


def test_name_matchers_cannot_see_a_wrong_argument() -> None:
    p = REF[:2] + [("issue_refund", W)]
    assert all(T.score(p, True, True).values()) and T.violations(p, True)


def test_superset_passes_a_duplicate_refund() -> None:
    assert T.score(REF + [REF[2]], True, True)["superset"]


def test_no_lookup_means_an_invented_order_id() -> None:
    steps, _ = T.build(True, "no reads", False, False, True, False, False, False)
    assert steps == [("issue_refund", W)]


def test_enumeration_is_a_distribution() -> None:
    for a in T.AGENTS.values():
        runs = T.enumerate_runs(a)
        assert len(runs) == 512 and abs(sum(r["p"] for r in runs) - 1) < 1e-12


def test_outcome_never_fails_a_valid_run_and_spec_errors_are_consistent() -> None:
    for s in T.evaluate().values():
        assert s["scorers"]["outcome"]["false_fail"] == 0
        assert s["p_good"] <= s["p_correct"]


def test_headline_numbers() -> None:
    ev = T.evaluate()
    v1, v2 = ev["v1 careful"], ev["v2 fewer calls"]
    assert round(v2["p_good"] - v1["p_good"], 3) == -0.312
    assert round(v2["scorers"]["strict"]["reported"] - v1["scorers"]["strict"]["reported"], 3) == -0.011
    assert round(v2["scorers"]["outcome"]["bad_among_passes"], 3) == 0.351


def test_simulation_check_can_fail() -> None:
    """A calibration that cannot fail proves nothing: a different agent's MC must miss this agent's exact value."""
    ex = T.summarise(T.enumerate_runs(T.AGENTS["v1 careful"]))["p_good"]
    mc = T.simulate(T.AGENTS["v2 fewer calls"], 20_000)["good"]
    lo, hi = T.wilson(mc, 20_000)
    assert not lo <= ex <= hi


def test_audit_counts_and_parser() -> None:
    assert T.parse_steps("search_kb; lookup_order:A17 ;check_policy") == [
        ("search_kb", None), ("lookup_order", "A17"), ("check_policy", None)]
    res = T.audit([(True, True, "lookup_order:A17;check_policy;issue_refund:A71"),
                   (True, True, "check_policy;lookup_order:A17;issue_refund:A17")])
    assert res["good"] == 0.5
    assert res["scorers"]["outcome"]["passed_bad"] == 1 and res["scorers"]["strict"]["failed_good"] == 1
    assert res["hidden"]["wrong argument"] == 1
