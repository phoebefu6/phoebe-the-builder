"""Engine tests: the truth column, the ceilings, and the findings the README states."""

from __future__ import annotations

import explore as E
import pytest


@pytest.mark.parametrize("s,group,given,family", E.CORPUS)
def test_truth_words_come_from_the_string(s: str, group: str, given: str, family: str) -> None:
    """Every true given/family word is a word of the string, so `roles` covers the whole label."""
    r = E.roles(s, given, family)
    assert r.count("G") == len(given.split()) and r.count("F") == len(family.split()), (s, r)


def test_every_share_group_is_in_the_corpus() -> None:
    groups = {g for _, g, _, _ in E.CORPUS}
    assert groups == set(E.SHARE)
    assert abs(sum(E.SHARE.values()) - 1.0) < 1e-9


def test_only_one_string_has_two_truths() -> None:
    c = E.ceiling()
    assert c["ambiguous_strings"] == ["Chen Lin"] and c["by_string"] == 50


def test_shape_ceiling_and_three_word_split() -> None:
    c = E.ceiling()
    assert c["by_shape"] == 41 and c["WWW"] == {"G-F": 4, "GFF": 5}


def test_headline_numbers() -> None:
    r = E.study()
    per = r["per"]
    assert r["n"] == 51 and r["best"] == "nameparser 2.5"
    assert per["nameparser 2.5"]["both_right"] == 34 and per["+ East Asian surname list"]["both_right"] == 34
    assert round(per["nameparser 2.5"]["greeted_wrong"], 4) == 0.0915
    assert round(per['rsplit(" ", 1)']["greeted_wrong"], 4) == 0.3850
    assert ("van den Broek, Pieter", "van") in per['split(" ", 1)']["greeted_as_affix"]
    assert len(r["wrong_by_all"]) == 6


def test_surname_list_trades_one_group_for_another() -> None:
    bg = E.study()["by_group"]
    assert bg["+ East Asian surname list"]["family first"] > bg["particle-aware rules"]["family first"]
    assert bg["+ East Asian surname list"]["Western-ordered East Asian"] < 1.0


def test_mononym_does_not_crash_any_splitter() -> None:
    for f in E.SPLITTERS.values():
        assert f("Wiratmo")[0] == "Wiratmo"
