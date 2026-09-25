"""Tests for the engine and for the claims the README makes from results.json."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import posthoc as P
import pytest
from scipy import stats

R = json.load(open("results.json")) if os.path.exists("results.json") else None
needs_results = pytest.mark.skipif(R is None, reason="evidence.py has not been run")


# ------------------------------------------------------------------ engine
def test_tukey_matches_scipy() -> None:
    cal = P.calibrate_against_scipy(trials=60)
    assert cal["tukey_decision_mismatches"] == 0
    assert cal["max_F_p_gap"] < 1e-12


def test_holm_matches_reference_step_down() -> None:
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 0.06, (500, 6))
    got = P.holm_reject(p)
    for row, g in zip(p, got):
        order = np.argsort(row)
        ref = np.zeros(6, bool)
        for r, i in enumerate(order):
            if row[i] <= P.ALPHA / (6 - r):
                ref[i] = True
            else:
                break
        assert (ref == g).all()


def test_identities_hold_on_every_replicate() -> None:
    ids = P.identity_checks(reps=2000)
    assert ids["holm_misses_bonferroni_pair"] == 0
    assert ids["holm_vs_bonferroni_any_differs_under_null"] == 0
    assert ids["lsd_without_F"] == 0


def test_wilson_verdicts_both_directions_reachable() -> None:
    assert P.verdict(1200, 20000) == "INFLATED"
    assert P.verdict(700, 20000) == "CONSERVATIVE"
    assert P.verdict(1000, 20000) == "ok"


def test_analyse_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        P.analyse([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        P.analyse([[1, 2], [3, 4], [5]])


def test_analyse_unequal_n_uses_tukey_kramer() -> None:
    g = [[1.0, 1.2, 0.9, 1.1], [2.0, 2.3, 1.8], [1.1, 0.8, 1.0, 1.3, 0.9, 1.2]]
    res = P.analyse(g)
    ref = stats.tukey_hsd(*g).pvalue
    assert res["tukey_p"][0] == pytest.approx(ref[0, 1])


def test_seeds_are_indices_not_hashes() -> None:
    """Day 178: hash()-derived seeds changed every process."""
    code = "import posthoc as P; print(P.run_design(4, 'null', 0.0, 1001, reps=500)['none_fwer'])"
    a = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
    b = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
    assert a == b and a.strip()


# ------------------------------------------------------------------ the study's claims
@needs_results
def test_tukey_is_the_calibration_cell() -> None:
    assert all(r["tukey_fwer_verdict"] == "ok" for r in R["null"])


@needs_results
def test_unadjusted_is_inflated_and_grows_with_k() -> None:
    f = [r["none_fwer"] for r in R["null"]]
    assert all(r["none_fwer_verdict"] == "INFLATED" for r in R["null"])
    assert f == sorted(f) and f[-1] > 0.6


@needs_results
def test_lsd_protected_only_under_complete_null_and_at_k3() -> None:
    assert all(r["lsd_fwer_verdict"] == "ok" for r in R["null"])
    part = R["partial"]
    assert part[0]["k"] == 3 and part[0]["lsd_fwer_verdict"] == "ok"
    assert all(r["lsd_fwer_verdict"] == "INFLATED" for r in part[1:])
    assert all(r["lsd_fwer"] == r["none_fwer"] for r in part), "protection should be fully spent"


@needs_results
def test_bonferroni_equals_holm_under_complete_null() -> None:
    assert all(r["bonferroni_fwer"] == r["holm_fwer"] for r in R["null"])


@needs_results
def test_tukey_vs_holm_crossover_in_k() -> None:
    for r in R["power"]:
        assert r["tukey_vs_holm"] == ("holm" if r["k"] == 3 else "tukey"), r
    assert R["tukey_vs_holm_tally"] == {"tukey": 5, "holm": 2, "tie": 0}


@needs_results
def test_inflated_procedures_are_not_compared() -> None:
    for r in R["power"]:
        if r["k"] > 3:
            assert not r["lsd_comparable"] and not r["none_comparable"]
        assert r["tukey_comparable"] and r["holm_comparable"]


@needs_results
def test_f_without_pair_grows_on_the_ladder() -> None:
    s = [r["tukey_no_pair_given_F"] for r in R["power"] if r["shape"] == "spread"]
    assert s == sorted(s) and s[-1] > 0.1


@needs_results
def test_exemplar_is_what_it_claims() -> None:
    ex = R["exemplar"]
    assert ex["F_p"] < 0.05 and not ex["significant"]["tukey"] and ex["significant"]["none"]
    res = P.analyse(ex["groups"])
    assert res["F_p"] == pytest.approx(ex["F_p"])


@needs_results
def test_results_are_reproducible_across_processes() -> None:
    """A small design rerun in a fresh process must equal the stored row exactly."""
    r = R["null"][0]
    again = P.run_design(r["k"], r["shape"], r["effect"], r["seed"], r["reps"])
    assert again["none_fwer"] == r["none_fwer"] and again["tukey_fwer"] == r["tukey_fwer"]
