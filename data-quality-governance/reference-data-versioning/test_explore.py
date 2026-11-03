"""Engine checks on a hand-built territory list, plus the study's invariants."""

from __future__ import annotations

import json

import explore as E

TINY = {
    "init": {"AAA": {"entity": "T0", "region": "North"}, "BBB": {"entity": "T1", "region": "South"}},
    "events": [
        {"kind": "move", "eff": 2, "rec": 4, "code": "AAA", "region": "East"},  # backdated by two months
        {"kind": "absorb", "eff": 3, "rec": 3, "code": "BBB", "into": "AAA"},
        {"kind": "create", "eff": 5, "rec": 5, "code": "BBB", "entity": "R0", "region": "West"},  # code reused
    ],
    "facts": [],
}


def test_backdated_move_is_invisible_until_recorded() -> None:
    assert [r["region"] for r in E.lookup(E.rows_known(TINY, 3), "AAA", 2)] == ["North"]
    assert [r["region"] for r in E.lookup(E.rows_known(TINY, 4), "AAA", 2)] == ["East"]


def test_reused_code_joins_by_time() -> None:
    today = E.rows_known(TINY)
    assert [r["region"] for r in E.lookup(today, "BBB", 1)] == ["South"]
    assert [r["region"] for r in E.lookup(today, "BBB", 6)] == ["West"]
    assert sorted(r["region"] for r in E.lookup(today, "BBB", None)) == ["South", "West"]  # code-only fans out


def test_lineage_follows_absorb() -> None:
    assert E.successors(TINY) == {"T1": ["T0"]}


def test_study_invariants() -> None:
    r = json.load(open("results.json"))
    a = r["asw"]
    assert a["bitemporal"]["months"] == 36 and a["bitemporal"]["reported"] == 1.0
    assert a["code_only"]["reported"] > 1 and a["code_only"]["lost"] == 0  # fan-out, never a drop
    assert a["overwrite"]["lost"] > 0
    assert a["scd2_asof"]["reported"] == 1.0 and a["scd2_asof"]["months"] < 36
    assert r["asis"]["even"]["cells"] < 180  # no allocation rule reaches the truth


def test_results_match_a_fresh_run() -> None:
    assert json.loads(json.dumps(E.study())) == json.load(open("results.json"))
