"""Engine tests: the town, the findings the README states, and the invariants behind them."""

from __future__ import annotations

from collections import Counter

import explore as E
import pytest


@pytest.fixture(scope="module")
def pop() -> list:
    return E.population()


@pytest.fixture(scope="module")
def r(pop: list) -> dict:
    return E.study(pop)


def test_town_shape(pop: list) -> None:
    assert len(pop) == 60_000
    codes = Counter(p[1] for p in pop)
    assert len(codes) == 600
    assert all(len(c) == 6 and c.isdigit() for c in codes)
    # the hierarchy is real: four digits group three postcodes, two digits group thirty
    assert len({c[:4] for c in codes}) == 200 and len({c[:2] for c in codes}) == 20
    assert min(codes.values()) == 15 and max(codes.values()) == 4438
    assert all(p[2] <= E.TODAY for p in pop)
    assert not any(p[4] == "pregnancy" and p[3] == "M" for p in pop)


def test_population_is_seeded_and_the_export_is_a_subset(pop: list) -> None:
    assert E.population() == pop
    ex = E.sample(pop, E.EXPORT_FRAC)
    assert len(ex) == 3000 and len({e[0] for e in ex}) == 3000
    ids = {p[0] for p in pop}
    assert all(e[0] in ids for e in ex)


def test_how_few_columns(r: dict) -> None:
    by = {tuple(p["columns"]): p for p in r["subsets"]}
    assert by[("postcode",)]["unique_pop"] == 0
    assert by[("sex",)]["unique_export"] == 0
    assert by[("date of birth",)]["unique_pop"] == 498
    assert by[("postcode", "date of birth")]["unique_pop"] == 2932
    assert by[("postcode", "date of birth", "sex")] == {
        **by[("postcode", "date of birth", "sex")],
        "unique_export": 3000,
        "unique_pop": 2964,
    }


def test_ladder_headlines(r: dict) -> None:
    by = {tuple(p["levels"]): p for p in r["ladder"]}
    assert len(by) == 40
    y6 = by[("6 digits", "year", "kept")]
    assert (y6["unique_export"], y6["unique_pop"]) == (2487, 1097)
    d6 = by[("6 digits", "decade", "kept")]
    assert d6["unique_pop"] == 111
    sy = by[("sector (2 digits)", "year", "kept")]
    assert (sy["unique_export"], sy["unique_pop"], sy["below_k"], sy["below_k_pop"], sy["k_min"]) == (1203, 1, 2740, 52, 1)
    assert len(r["k_anonymous"]) == 6
    assert tuple(r["best"]["levels"]) == ("sector (2 digits)", "dropped", "kept") and r["best"]["classes"] == 40
    rel = r["release"]
    assert tuple(rel["levels"]) == ("sector (2 digits)", "decade", "dropped")
    assert (rel["classes"], rel["below_k"], rel["released"]) == (199, 51, 2949)


def test_k_anonymous_means_nothing_below_k(r: dict) -> None:
    ok = {tuple(p["levels"]) for p in r["k_anonymous"]}
    assert ok == {tuple(p["levels"]) for p in r["ladder"] if p["below_k"] == 0}
    assert all(p["k_min"] >= E.K for p in r["k_anonymous"])
    assert all(p["below_k"] <= 0.05 * p["n"] for p in r["with_suppression"])


def test_town_uniqueness_never_exceeds_export_uniqueness(r: dict) -> None:
    """A row unique in the town is unique in any subset of it; the converse fails, and that gap is the finding."""
    for p in r["ladder"] + r["subsets"]:
        assert p["unique_pop"] <= p["unique_export"]
        assert p["below_k_pop"] <= p["below_k"]
    for s in r["sweep"]:
        assert s["unique_pop_pct"] <= s["unique_export_pct"] + 1e-9


def test_generalising_never_adds_classes(r: dict) -> None:
    by = {tuple(p["levels"]): p for p in r["ladder"]}
    P, D = list(E.POSTCODE_LEVELS), list(E.DOB_LEVELS)
    for sl in E.SEX_LEVELS:
        for dl in D:
            for a, b in zip(P, P[1:]):
                assert by[(b, dl, sl)]["classes"] <= by[(a, dl, sl)]["classes"]
        for pl in P:
            for a, b in zip(D, D[1:]):
                assert by[(pl, b, sl)]["classes"] <= by[(pl, a, sl)]["classes"]


def test_sweep(r: dict) -> None:
    at5 = {tuple(s["levels"]): s for s in r["sweep"] if s["frac"] == 0.05}
    sy = at5[("sector (2 digits)", "year", "kept")]
    assert round(sy["unique_export_pct"], 1) == 40.1 and round(sy["unique_pop_pct"], 2) == 0.03
    full = at5[E.FULL]
    assert full["unique_export_pct"] == 100.0 and round(full["unique_pop_pct"], 1) == 98.8
    whole = [s for s in r["sweep"] if s["frac"] == 1.0]
    assert all(s["unique_export_pct"] == s["unique_pop_pct"] for s in whole), "at 100% the export IS the town"


def test_linkage(r: dict) -> None:
    by = {(tuple(k["levels"]), k["coverage"]): k for k in r["linkage"]}
    full = by[(E.FULL, 1.0)]
    assert (full["right"], full["wrong"], full["ambiguous"], full["none"]) == (2964, 0, 36, 0)
    assert full["right"] == r["subsets"][-1]["unique_pop"], "a complete register re-identifies exactly the town-unique rows"
    assert all(k["wrong"] == 0 for k in r["linkage"] if k["coverage"] == 1.0), "a complete register is never wrong"
    y6 = by[(("6 digits", "year", "kept"), 0.8)]
    assert (y6["right"], y6["wrong"]) == (977, 101)
    sy = by[(("sector (2 digits)", "year", "kept"), 0.8)]
    assert sy["wrong"] > sy["right"]
    for k in r["linkage"]:
        assert k["right"] + k["wrong"] + k["ambiguous"] + k["none"] == r["export_n"]


def test_k_does_not_protect_the_value(r: dict) -> None:
    assert r["sensitive_base"] == 352
    rel = r["release"]
    assert (rel["homog_classes"], rel["near_homog_rows"], rel["sensitive_confident_rows"]) == (0, 31, 0)
    hot = [p for p in r["ladder"] if p["sensitive_confident_rows"] > 0]
    assert len(hot) == 10
    assert max(p["sensitive_confident_rows"] for p in hot) == 13


def test_ceiling(r: dict) -> None:
    assert r["smallest_postcodes"][0] == ("040800", 15)
    assert r["postcodes"] == {"postcodes": 600, "min": 15, "median": 46, "max": 4438}
