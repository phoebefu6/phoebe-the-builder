"""Engine checks: each fingerprint's known blind spot, pinned by hand-built tables, plus the study's headline."""

from __future__ import annotations

import json
import random

import explore as E

ROWS = [
    {"event_id": 1, "customer": "Ada Lim", "note": "gift", "amount": 12.5, "discount": None},
    {"event_id": 2, "customer": "Ben Okafor", "note": None, "amount": 7.0, "discount": 0.0},
    {"event_id": 3, "customer": "Chloe Tan", "note": "", "amount": 0.1 + 0.2, "discount": 5.0},
]
T = (list(E.COLS), ROWS)


def test_shuffle_breaks_only_the_dump() -> None:
    u = (T[0], list(reversed(ROWS)))
    assert E.csv_md5(u) != E.csv_md5(T)
    for k in ("count_sums", "xor_rows", "sum_rows", "canonical"):
        assert E.SCHEMES[k](u) == E.SCHEMES[k](T)


def test_xor_cancels_a_pair() -> None:
    doubled = (T[0], ROWS + [dict(ROWS[0]), dict(ROWS[0])])
    assert E.xor_rows(doubled) == E.xor_rows(T)
    assert E.sum_rows(doubled) != E.sum_rows(T)


def test_delimiter_collision_fools_naive_join_not_canonical() -> None:
    a = (T[0], [dict(ROWS[0], customer="Ada Lim|", note="gift")])
    b = (T[0], [dict(ROWS[0], customer="Ada Lim", note="|gift")])
    assert E.sum_rows(a) == E.sum_rows(b)
    assert E.canonical(a) != E.canonical(b)


def test_canonical_folds_types_and_negative_zero() -> None:
    u = (list(reversed(E.COLS)), [dict(r) for r in ROWS])
    u[1][1]["discount"] = -0.0
    u[1][2]["discount"] = 5
    u[1][2]["amount"] = "0.30"
    u[1][0]["discount"] = E.NULL
    assert E.canonical(u) == E.canonical(T)


def test_null_and_empty_are_different_to_canonical_same_to_dump() -> None:
    u = (T[0], [dict(r) for r in ROWS])
    u[1][1]["note"] = ""
    assert E.csv_md5(u) == E.csv_md5(T)
    assert E.canonical(u) != E.canonical(T)


def test_swap_fools_column_sums() -> None:
    u = (T[0], [dict(r) for r in ROWS])
    u[1][0]["amount"], u[1][1]["amount"] = u[1][1]["amount"], u[1][0]["amount"]
    assert E.count_sums(u) == E.count_sums(T)
    assert E.canonical(u) != E.canonical(T)


def test_half_cent_straddle() -> None:
    assert E.money(821.225) != E.money(821.2250000000001)


def test_results_match_a_fresh_run() -> None:
    r = json.load(open("results.json"))
    t, items = E.make_table()
    assert json.loads(json.dumps(E.float_detail(t, items))) == r["float"]
    assert r["tally"]["canonical"] == {"false_alarms": ["re-add line items"], "misses": []}
    assert r["tally"]["xor_rows"]["misses"] == ["dup pair dropped, another doubled", "'|' moved across a column"]


def test_mutations_are_real_changes() -> None:
    """Every mutation must actually change the multiset of canonical rows, or a 'miss' would be a fair answer."""
    t, items = E.make_table()
    base = sorted(E.canonical_row(r) for r in t[1])
    for name, fn in E.MUTATIONS.items():
        u = fn(t, items, random.Random(1))
        assert sorted(E.canonical_row(r) for r in u[1]) != base, name
