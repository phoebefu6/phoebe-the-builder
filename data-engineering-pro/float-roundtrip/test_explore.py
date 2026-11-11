"""Engine claims the README makes, each checked against a fresh run."""

from __future__ import annotations

import json

import explore as E

R = E.study()
W = R["writers"]


def test_seven_writers_exact_everywhere() -> None:
    clean = [k for k in E.WRITERS if all(d["changed"] + d["lost"] == 0 for d in W[k].values())]
    assert clean == ["repr", "%.17g", "pickle", "np.savetxt", "json", "csv module", "pandas round_trip"]


def test_typed_amounts_pass_all_but_float32() -> None:
    bad = [k for k in E.WRITERS if W[k]["entered"]["exact"] < E.N]
    assert bad == ["float32"]


def test_15g_changes_two_thirds_of_computed_and_total_does_not_move() -> None:
    assert W["%.15g"]["computed"]["changed"] == 1318
    assert R["ledger"]["%.15g"]["drift"] == 0.0
    assert R["ledger"]["%.15g"]["join_miss"] > 0.6


def test_sqlite_text_matches_15g_on_data_and_loses_nan_and_negative_zero() -> None:
    assert W["sqlite TEXT"]["computed"] == W["%.15g"]["computed"]
    assert R["edge"]["sqlite REAL"][1] == "0.0" and R["edge"]["sqlite REAL"][2] == "lost"


def test_pandas_file_is_exact_reader_is_not() -> None:
    assert W["pandas round_trip"]["measured"]["changed"] == 0
    assert W["pandas csv"]["measured"]["max_ulp"] > 1000


def test_strict_json_refuses_nan_and_inf() -> None:
    assert R["edge"]["json strict"][2:5] == ["lost", "lost", "lost"]


def test_results_json_matches_fresh_run() -> None:
    saved = json.load(open("results.json"))["study"]["writers"]
    assert saved == json.loads(json.dumps(W))
