"""Engine claims the README makes, each checked against a fresh run."""

from __future__ import annotations

import json

import explore as E

R = E.study()
RD, FX = R["readers"], R["fixes"]
BIG = ("boundary", "snowflake")


def test_three_readers_exact_everywhere() -> None:
    ok = [k for k in E.READERS if all(not RD[k][c][g]["wrong"] and not RD[k][c][g]["raised"]
                                      for c in E.CONTEXTS for g in E.GROUPS)]
    assert ok == ["json", "ujson", "pydantic int"]


def test_small_ids_never_wrong() -> None:
    assert not any(RD[k][c]["small"]["wrong"] for k in E.READERS for c in E.CONTEXTS)


def test_one_null_breaks_pandas_but_not_arrow_or_duckdb() -> None:
    assert RD["pandas read_json"]["one null"]["snowflake"]["dtype"] == "float64"
    assert RD["pandas read_json"]["one null"]["snowflake"]["wrong"] == 1960
    assert not RD["pyarrow"]["one null"]["snowflake"]["wrong"] and not RD["duckdb"]["one null"]["snowflake"]["wrong"]


def test_pandas_dtype_says_int64_while_values_are_wrong() -> None:
    d = RD["pandas read_json"]["one 123.0"]["snowflake"]
    assert d["dtype"] == "int64" and d["wrong"] == 1960


def test_boundary_merges_499_ids() -> None:
    d = RD["JSON.parse (double)"]["clean"]["boundary"]
    assert d["wrong"] == 500 and d["merged"] == 499 and d["max_off"] == 1


def test_first_failures() -> None:
    assert R["first"]["JSON.parse (double)"] == 2**53 + 1
    assert R["first"]["pyarrow"] == 2**63 + 1 and R["first"]["duckdb"] == 2**64 + 1


def test_pandas_fixes_fail_and_duckdb_hugeint_holds() -> None:
    assert FX["pandas dtype=Int64"]["one null"]["wrong"] == 1960
    assert FX["pandas dtype=str"]["one null"]["wrong"] == 2000
    assert all(not d["wrong"] and not d["raised"] for d in FX["duckdb columns=HUGEINT"].values())
    assert FX["pyarrow explicit int64"]["one 123.0"]["raised"]


def test_results_json_matches_fresh_run() -> None:
    saved = json.load(open("results.json"))["study"]
    assert saved == json.loads(json.dumps(R))
