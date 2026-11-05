"""Engine checks on a hand-built hierarchy, plus the study's invariants."""

from __future__ import annotations

import json

import explore as E

TINY = {
    "parent": {"North": "TOTAL", "South": "TOTAL", "A": "North", "B": "South", "X": "North", "L1": "B", "L2": "A",
               "K": "X", "NEW": "A"},
    "extract": [
        {"child": "North", "parent": "TOTAL", "load": 0}, {"child": "South", "parent": "TOTAL", "load": 0},
        {"child": "A", "parent": "North", "load": 0}, {"child": "B", "parent": "South", "load": 0},
        {"child": "L1", "parent": "A", "load": 1},   # old row, never end-dated
        {"child": "L1", "parent": "B", "load": 9},   # L1 moved to B
        {"child": "L2", "parent": "Z99", "load": 0},  # orphan
        {"child": "X", "parent": "K", "load": 0}, {"child": "K", "parent": "X", "load": 0},  # cycle
    ],
    "posts": [("L1", 10), ("L2", 5), ("K", 7), ("NEW", 3), ("A", 2)],
}


def test_naive_walk_counts_moved_leaf_twice_and_drops_the_rest() -> None:
    rep = E.rollup(TINY, "naive_walk")
    assert rep["North"] == 10 and rep["South"] == 10 and rep["TOTAL"] == 20  # L1 under both; A's own post dropped


def test_one_parent_rules() -> None:
    assert E.one_parent(TINY["extract"], "first")["L1"] == "A"
    assert E.one_parent(TINY["extract"], "latest")["L1"] == "B"


def test_validated_ties_and_names_the_unplaced() -> None:
    rep = E.rollup(TINY, "validated")
    assert rep["TOTAL"] == 27 and rep[E.UNASSIGNED] == 15 and rep["South"] == 10 and rep["North"] == 2


def test_defects_found() -> None:
    d = E.defects(TINY)
    assert d["multi_parent"] == ["L1"] and d["orphan"] == ["L2"] and d["cycle"] == ["K", "X"]
    assert d["unknown_code"] == ["NEW"]


def test_breakdown_splits_every_dollar() -> None:
    bd = E.breakdown(TINY, "naive_walk")
    assert bd["right"] + bd["misfiled"] + bd["dropped"] == 27 and bd["double"] == 10


def test_study_invariants() -> None:
    r = json.load(open("results.json"))
    m = r["methods"]
    assert all(v["sums_to_total"] for v in m.values())  # the check everyone runs never fires
    assert m["plug"]["total_is_ledger"] and m["plug"]["wrong"] > 2 * r["plug_net"]
    assert m["latest_parent"]["misfiled"] == 0 < m["first_parent"]["misfiled"]
    assert m["validated"]["total_is_ledger"] and m["validated"]["regions_right"] < 5


def test_results_match_a_fresh_run() -> None:
    r = json.load(open("results.json"))
    r.pop("seeds")
    assert json.loads(json.dumps(E.study())) == r
