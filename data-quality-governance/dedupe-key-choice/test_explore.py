"""Engine tests: the corpus, the findings the README states, and the invariants behind them."""

from __future__ import annotations

from itertools import combinations

import explore as E
import pytest


def test_corpus_shape() -> None:
    assert len(E.CORPUS) == 45
    assert len({t for _, t, _ in E.CORPUS}) == 23
    ids = sorted({t for _, t, _ in E.CORPUS})
    assert ids == [f"P{i:02d}" for i in range(1, 24)]


@pytest.fixture(scope="module")
def r() -> dict:
    return E.study()


def test_headline_numbers(r: dict) -> None:
    per = r["per"]
    assert r["ceiling"]["true_pairs"] == 37
    assert len(per["exact"]["false_splits"]) == 37 and len(per["exact"]["false_merges"]) == 1
    assert len(per["normalised"]["false_splits"]) == 27 and per["normalised"]["customers"] == 36
    assert len(per["soundex"]["false_merges"]) == 13 and len(per["metaphone"]["false_merges"]) == 11
    jw, ts = per["jaro-winkler >= 0.9"], per["token-set ratio >= 85"]
    assert (len(jw["false_splits"]), len(jw["false_merges"]), jw["chained_pairs"]) == (14, 19, 3)
    assert (len(ts["false_splits"]), len(ts["false_merges"]), ts["chained_pairs"]) == (9, 18, 5)


def test_exact_merges_only_the_namesakes(r: dict) -> None:
    assert r["per"]["exact"]["false_merges"] == [("John Smith", "John Smith")]
    assert r["ceiling"]["two_truths"] == ["John Smith"]


def test_every_key_merges_a_superset_of_exact(r: dict) -> None:
    """Exact is the finest key: nothing it joins is kept apart by any other key."""
    assert all(r["disagreement"]["exact"][b] == 0 for b in E.NAMES)
    assert r["disagreement"]["token-set ratio >= 85"]["exact"] == 45


def test_no_zero_error_threshold(r: dict) -> None:
    for b in r["ceiling"]["best_frontier"].values():
        assert not b["zero_exists"]
        assert b["false_splits"] + b["false_merges"] >= 23


def test_clustering_trades_splits_for_merges(r: dict) -> None:
    for n in E.PAIR_KEYS:
        d, p = r["direct"][n], r["per"][n]
        assert len(p["false_splits"]) <= d["false_splits"]
        assert len(p["false_merges"]) >= d["false_merges"]


def test_chained_pair_is_below_threshold() -> None:
    sim, t = E.PAIR_KEYS["token-set ratio >= 85"]
    assert sim("Anna Schmidt", "Anna Müller") < t
    lab = E.clusters("token-set ratio >= 85")
    i = [s for s, _, _ in E.CORPUS].index("Anna Schmidt")
    j = [s for s, _, _ in E.CORPUS].index("Anna Müller")
    assert lab[i] == lab[j]


def test_floor_shared_by_every_key(r: dict) -> None:
    assert len(r["never_merged"]) == 7
    assert ("William Jones", "Bill Jones") in r["never_merged"]


def test_normalised_handles_order_and_accents_but_not_apostrophes() -> None:
    assert E._normalised("Smith, Jonathan") == E._normalised("Jonathan Smith")
    assert E._normalised("José Núñez") == E._normalised("JOSE NUNEZ")
    assert E._normalised("Mary O'Brien") != E._normalised("Mary OBrien")


def test_components_is_transitive() -> None:
    assert E._components(4, [(0, 1), (1, 2)]) in ([0, 0, 0, 3], [1, 1, 1, 3], [2, 2, 2, 3])


def test_garbage_does_not_crash() -> None:
    for f in E.ROW_KEYS.values():
        f("")
        f("   ")
        f("Dr.")
    for sim, _ in E.PAIR_KEYS.values():
        sim("", "x")


def test_true_pairs_count_matches_combinations() -> None:
    n = sum(1 for i, j in combinations(range(len(E.CORPUS)), 2) if E.CORPUS[i][1] == E.CORPUS[j][1])
    assert n == 37
