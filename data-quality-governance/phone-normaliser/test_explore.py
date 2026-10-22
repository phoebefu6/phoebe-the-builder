"""Engine tests: the truth column, the ceilings, and the findings the README states."""

from __future__ import annotations

import explore as E
import phonenumbers as pn
import pytest


@pytest.mark.parametrize("s,country,truth,fmt", E.CORPUS)
def test_every_truth_is_a_valid_number(s: str, country: str, truth: str, fmt: str) -> None:
    assert pn.is_valid_number(pn.parse(truth, None)), truth


def test_plus_rows_read_one_way_only() -> None:
    for s, _, t, _ in E.CORPUS:
        if s.startswith("+"):
            assert list(E.readings(s)) == [t], s


@pytest.fixture(scope="module")
def r() -> dict:
    return E.study(sweep_n=0)


def test_headline_numbers(r: dict) -> None:
    per = r["per"]
    assert r["n"] == 40 and r["true_customers"] == 15
    assert per["phonenumbers, region=country field"]["right"] == 33
    assert len(per["last 9 digits"]["false_merges"]) == 11
    assert len(per["digits only"]["false_splits"]) == 37
    nr = per["phonenumbers, no region (+ only)"]
    assert nr["wrong"] == 0 and not nr["false_merges"] and not nr["false_splits"] and nr["refused"] == 28


def test_ceilings(r: dict) -> None:
    assert r["ceiling"] == {"by_string": 13, "by_string_and_field": 39, "two_truths": ["9123 4567 [SG]"]}


def test_double_zero_is_not_universal() -> None:
    """Singapore's international prefix is not 00, so 0033... parses as a Hungarian number under region=SG."""
    assert E._lib("0033612345678", "SG") == "+3612345678"
    assert E._lib("0033612345678", "FR") == "+33612345678"


def test_spanish_digits_are_many_numbers() -> None:
    assert len(E.readings("612 345 678")) == 30


def test_sweep_is_seeded_and_never_unambiguous() -> None:
    a, b = E.sweep(40), E.sweep(40)
    assert a == b
    assert all(v["valid_3plus"] == 1.0 for v in a.values())


def test_garbage_does_not_crash() -> None:
    for f in E.NORMALISERS.values():
        f("call me", "", E.HQ)
        f("", "GB", E.HQ)
