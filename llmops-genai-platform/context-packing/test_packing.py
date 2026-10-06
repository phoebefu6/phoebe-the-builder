"""Engine tests - the reading model, the packer, the exact rates against raw simulation, and the README's claims."""

from __future__ import annotations

import math

import packing as P
import pytest

LOG = P.make_log()


def test_curve_passes_through_the_three_declared_points() -> None:
    a, b, c = P.LIU_POINTS
    assert math.isclose(P.curve(0.0), a) and math.isclose(P.curve(0.5), b) and math.isclose(P.curve(1.0), c)
    assert P.curve(0.5) < P.curve(0.0) and P.curve(0.5) < P.curve(1.0)


def test_read_sinks_with_length_and_is_clipped() -> None:
    assert P.read(0.0, 2_000) > P.read(0.0, 4_000) > P.read(0.0, 12_000)
    assert P.read(0.5, 10 ** 7) == P.READ_FLOOR and P.read(0.0, -10 ** 7) == P.READ_CEIL


def test_comply_falls_with_distance_and_is_clipped() -> None:
    assert P.comply(0) == P.COMPLY_AT_ZERO
    assert P.comply(4_000) < P.comply(1_000) < P.comply(0)
    assert P.comply(10 ** 6) == P.COMPLY_FLOOR


def test_select_respects_the_budget_and_the_fixed_costs() -> None:
    lens = [300] * 30
    for instr in ("top", "both"):
        kept = P.select(lens, 4_000, instr)
        fixed = P.INSTRUCTION_TOKENS * (2 if instr == "both" else 1) + P.QUESTION_TOKENS
        assert sum(lens[r - 1] for r in kept) + fixed <= 4_000
        assert kept == list(range(1, len(kept) + 1))
    assert len(P.select(lens, 4_000, "both")) < len(P.select(lens, 4_000, "top"))


def test_arrange_orders() -> None:
    k = [1, 2, 3, 4, 5]
    assert P.arrange(k, "rank") == [1, 2, 3, 4, 5]
    assert P.arrange(k, "reverse") == [5, 4, 3, 2, 1]
    assert P.arrange(k, "reorder") == [1, 3, 5, 4, 2]   # best at both edges, worst in the middle
    with pytest.raises(ValueError):
        P.arrange(k, "random")


def test_window_positions_run_first_zero_to_last_one() -> None:
    w = P.window([300] * 30, {"budget": 4_000, "order": "rank", "instruction": "top"})
    pos = [w["pos"][r] for r in w["layout"]]
    assert pos[0] == 0.0 and math.isclose(pos[-1], 1.0) and pos == sorted(pos)
    lone = P.window([500] * 30, {"budget": P.INSTRUCTION_TOKENS + P.QUESTION_TOKENS + 500, "order": "rank", "instruction": "top"})
    assert lone["layout"] == [1] and lone["pos"][1] == 0.0
    assert w["distance"] == w["total"] - P.INSTRUCTION_TOKENS - P.QUESTION_TOKENS
    both = P.window([300] * 30, {"budget": 4_000, "order": "rank", "instruction": "both"})
    assert both["distance"] == 0


def test_failure_classes_sum_to_one_minus_answer() -> None:
    for pol in P.POLICIES.values():
        for e in LOG[:200]:
            s = P.score_query(e, pol)
            assert math.isclose(s["fail_missing"] + s["fail_unread"] + s["fail_ignored"], 1 - s["answer"], abs_tol=1e-12)


def test_simulation_agrees_with_exact() -> None:
    cal = P.calibrate(LOG, reps=20)
    assert sum(r["inside_99"] for r in cal) >= len(cal) - 1


def test_simulation_check_can_fail() -> None:
    """v2's simulated answer rate must land outside v1's exact value, or the check proves nothing."""
    sim = P.simulate(P.POLICIES["v2 rank, 12k"], LOG, reps=20)
    k, n = sim["answer_rate"]
    lo, hi = P.wilson(k, n)
    assert not (lo <= P.evaluate(P.POLICIES["v1 rank, 4k"], LOG)["answer_rate"] <= hi)


def test_ranking_flips_between_the_dashboard_and_the_user() -> None:
    v1, v2 = P.evaluate(P.POLICIES["v1 rank, 4k"], LOG), P.evaluate(P.POLICIES["v2 rank, 12k"], LOG)
    assert v2["context_recall"] > v1["context_recall"] + 0.05
    assert v2["answer_rate"] < v1["answer_rate"] - 0.05
    assert v2["fail_unread"] > v1["fail_unread"] and v2["fail_ignored"] > v1["fail_ignored"]


def test_present_but_unread_is_the_largest_failure_class_at_v1() -> None:
    v1 = P.evaluate(P.POLICIES["v1 rank, 4k"], LOG)
    assert v1["fail_unread"] > v1["fail_missing"] > v1["fail_ignored"]


def test_repeating_the_instruction_costs_the_dashboard_and_pays_the_user() -> None:
    v3, v4 = P.evaluate(P.POLICIES["v3 reorder, 4k"], LOG), P.evaluate(P.POLICIES["v4 reorder+repeat, 4k"], LOG)
    assert v4["context_recall"] < v3["context_recall"]
    assert v4["answer_rate"] > v3["answer_rate"] + 0.02
    assert v4["compliance"] > v3["compliance"]


def test_same_set_different_order_is_invisible_to_the_log() -> None:
    sd = P.same_set_different_order(LOG, P.POLICIES["v1 rank, 4k"], P.POLICIES["v3 reorder, 4k"])
    assert sd["identical_sets"] and sd["context_recall_gap"] == 0.0 and sd["tokens_gap"] == 0.0
    assert 0 < sd["answer_rate_gap"] < 0.02          # the published mitigation is nearly a wash here
    assert sd["gold_in_middle_b"] < sd["gold_in_middle_a"] / 1.5


def test_reorder_sends_rank_two_to_the_far_end() -> None:
    v1 = {r["rank"]: r for r in P.by_rank(P.POLICIES["v1 rank, 4k"], LOG, 5)}
    v3 = {r["rank"]: r for r in P.by_rank(P.POLICIES["v3 reorder, 4k"], LOG, 5)}
    assert v1[1]["read"] == v3[1]["read"]
    assert v3[2]["read"] < v1[2]["read"]
    assert v3[3]["read"] > v1[3]["read"] and v3[5]["read"] > v1[5]["read"]


def test_sweep_context_recall_monotone_answer_rate_peaks_inside() -> None:
    sw = P.sweep([1_000, 2_000, 4_000, 6_000, 8_000, 12_000], LOG)
    cr = [r["context_recall"] for r in sw]
    assert cr == sorted(cr)
    ar = [r["answer_rate"] for r in sw]
    assert ar.index(max(ar)) not in (0, len(ar) - 1)


def test_log_is_seeded_and_ranks_are_distinct() -> None:
    a, b = P.make_log(50, 7), P.make_log(50, 7)
    assert a == b and a != P.make_log(50, 8)
    for e in LOG:
        got = [r for r in e["gold"] if r is not None]
        assert len(got) == len(set(got)) and all(1 <= r <= P.N_CAND for r in got)
        assert len(e["gold"]) in P.FACTS_NEEDED


def test_parse_policy_forms() -> None:
    assert P.parse_policy("4000, reorder, both") == {"budget": 4000, "order": "reorder", "instruction": "both"}
    assert P.parse_policy("12k, Rank, TOP") == {"budget": 12000, "order": "rank", "instruction": "top"}
    for bad in ("4000, rank", "x, rank, top", "300, rank, top", "4000, shuffle, top", "4000, rank, middle"):
        with pytest.raises(ValueError):
            P.parse_policy(bad)
