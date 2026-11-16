"""The findings the README states, asserted against the engine."""

from __future__ import annotations

import explore as E
from evidence import pairs


def test_stdlib_last_strict_refuses_polars_first() -> None:
    assert E.classify("json", "top level") == "last"
    assert E.classify("json + strict hook", "escaped key") == "refuse"
    assert E.classify("polars read_ndjson", "top level") == "first"
    assert E.classify("duckdb json_extract", "nested") == "first"


def test_duckdb_case_collision_reads_wrong_column() -> None:
    assert E.classify("duckdb read_json", "case only (not a dup)") == "WRONG (1)"
    assert E.classify("json", "case only (not a dup)") == "ok"


def test_feed_first_wins_misses_every_refund_and_strict_refuses_file() -> None:
    f = E.make_feed()
    r = E.feed_study(f)
    assert r["_truth"]["refunds"] == 46
    assert r["json"]["refunds"] == 46
    assert r["polars read_ndjson"]["refunds"] == 0 and not r["polars read_ndjson"]["raised"]
    assert r["pyarrow read_json"]["raised"] and r["pyarrow read_json"]["rows"] == 0


def test_regex_misses_escaped_keys_hook_does_not() -> None:
    d = E.detect(E.make_feed()["lines"])
    assert d == {"regex": 30, "strict hook": 46}


def test_pair_disagreement() -> None:
    p = pairs(E.matrix())
    assert p["disagree"] > p["pairs"] // 2
