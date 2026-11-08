"""Engine tests: each pins a claim the README makes, and the check definitions against hand-built joins."""

from __future__ import annotations

import explore as E
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def world() -> dict:
    return E.make_world()


@pytest.fixture(scope="module")
def res(world: dict) -> dict:
    return E.run(world)


def test_clean_join_is_silent_and_exact(res: dict) -> None:
    v = res["clean"]
    assert v["rows"] == 0 and v["revenue"] == 0 and not any(v["fires"].values())


def test_pk_preserved_is_the_only_perfect_check(res: dict) -> None:
    broken = [k for k in E.SCENARIOS if k not in ("clean", "orphan_dups")]
    perfect = [c for c in E.CHECKS if all(res[k]["fires"][c] for k in broken)
               and not res["orphan_dups"]["fires"][c] and not res["clean"]["fires"][c]]
    assert perfect == ["pk_preserved"]


def test_declared_scd2_test_passes_on_the_fanout(res: dict) -> None:
    assert not res["scd2_no_filter"]["fires"]["declared_unique"] and res["scd2_no_filter"]["revenue"] > 0.1


def test_offsets_fool_exactly_one_total_check(res: dict) -> None:
    assert res["rows_offset"]["rows"] == 0 and not res["rows_offset"]["fires"]["row_count"]
    assert abs(res["revenue_offset"]["revenue"]) <= E.TOL and not res["revenue_offset"]["fires"]["revenue_total"]
    assert abs(res["rows_offset"]["worst"]) > 0.1 and abs(res["revenue_offset"]["worst"]) > 0.1


def test_orphan_duplicates_are_harmless_but_alarm_the_table_tests(res: dict) -> None:
    v = res["orphan_dups"]
    assert v["revenue"] == 0 and v["fires"]["join_key_unique"] and not v["fires"]["referenced_key_unique"]


def test_upgraders_are_worth_more_than_their_row_share(world: dict) -> None:
    s = E.study()
    assert s["upgraded_revenue_share"] > 1.4 * s["upgraded_order_share"]


def test_pk_check_on_a_hand_built_fanout() -> None:
    """The edge case: a dimension duplicated only for one key still breaks the grain."""
    o = pd.DataFrame({"order_id": [1, 2, 3], "k": [10, 20, 30], "amount": [5.0, 5.0, 5.0]})
    d = pd.DataFrame({"k": [10, 20, 20, 30], "x": list("abcd")})
    j = o.merge(d, on="k", how="left")
    assert len(j) == 4 and not j.order_id.is_unique
    with pytest.raises(pd.errors.MergeError):
        o.merge(d, on="k", validate="many_to_one")
