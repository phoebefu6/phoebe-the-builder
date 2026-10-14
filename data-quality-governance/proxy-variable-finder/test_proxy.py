"""Tests for the proxy engine. Fast properties on small samples, then the study's headlines from results.json."""

from __future__ import annotations

import json
import os

import numpy as np
import proxy as P
import pytest

D = P.generate(n=30_000, seed=7)


def test_skill_is_identical_across_groups_and_history_is_not() -> None:
    g = D["g"]
    assert abs(D["skill"][g == 0].mean() - D["skill"][g == 1].mean()) < 0.05
    assert abs(D["y_true"][g == 0].mean() - D["y_true"][g == 1].mean()) < 0.02
    assert D["y_hist"][g == 0].mean() - D["y_hist"][g == 1].mean() > 0.15


def test_pair_is_invisible_alone_and_visible_together() -> None:
    sc = P.screen(D)
    assert sc["single"]["commute"] < 0.53 and sc["single"]["shift"] < 0.53
    assert sc["pairs"]["commute+shift"] > 0.85
    assert sc["joint"] > sc["pairs"]["commute+shift"]


def test_pair_link_zero_removes_the_joint_proxy() -> None:
    """The screen must report no pair when the world has none - a detector that always fires is not a detector."""
    d = P.generate(n=30_000, seed=7, world=P.parse_world("pair_link=0"))
    assert P.screen(d)["pairs"]["commute+shift"] < 0.53


def test_repair_removes_the_group_mean_and_nothing_else() -> None:
    r = P._repair(D, ["exp"])
    g = D["g"]
    assert abs(r["exp"][g == 0].mean()) < 1e-9 and abs(r["exp"][g == 1].mean()) < 1e-9
    assert np.allclose(np.diff(r["exp"][g == 1]), np.diff(D["exp"][g == 1]))
    assert r["cert"] is D["cert"]


def test_linear_repair_cannot_see_the_pair() -> None:
    """commute and shift have the same mean in both groups, so subtracting group means leaves them untouched."""
    r = P._repair(D, ["commute", "shift"])
    for c in ("commute", "shift"):
        assert np.corrcoef(r[c], D[c])[0, 1] > 0.999


def test_no_bias_world_selects_fairly_even_when_aware() -> None:
    d = P.generate(n=30_000, seed=7, world=P.parse_world("label_bias=0, zip_gap=0, exp_gap=0, pair_link=0"))
    r = P.evaluate(d, P.MODELS["aware (group as a feature)"])
    assert 0.9 < r["ratio"] < 1.1


def test_flagged_reports_the_hidden_pair() -> None:
    fake = {"single": {f: {"mean": v} for f, v in {"zip": 0.84, "exp": 0.64, "cert": 0.5, "commute": 0.5,
                                                    "shift": 0.5}.items()},
            "pairs": {"commute+shift": {"mean": 0.92}, "zip+exp": {"mean": 0.85}, "cert+shift": {"mean": 0.5}}}
    fl = P.flagged(fake)
    assert fl["single"] == ["zip", "exp"] and fl["pairs_hidden_from_single"] == ["commute+shift"]


@pytest.mark.parametrize("bad", ["nope=1", "share_b", "share_b=x", "share_b=1.5", "pair_link=2"])
def test_parse_world_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        P.parse_world(bad)


needs_results = pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")


@needs_results
def test_headlines_from_the_study() -> None:
    r = json.load(open("results.json"))
    m = r["models"]
    aware, unaware = m["aware (group as a feature)"], m["unaware (group removed)"]
    best = m["drop zip + pair, repair exp"]
    assert aware["ratio"]["mean"] < 0.25 and unaware["ratio"]["mean"] < 0.45, "removing the group is not enough"
    assert best["ratio"]["lo"] >= 0.95 and best["group_from_score"]["mean"] < 0.52
    assert best["auc_true"]["mean"] > aware["auc_true"]["mean"] + 0.03
    assert best["auc_hist"]["mean"] < aware["auc_hist"]["mean"] - 0.03, "the watched number falls"
    assert m["linear repair of every feature"]["ratio"]["mean"] < 0.7, "linear repair leaves the pair in"
    assert r["flagged"]["pairs_hidden_from_single"] == ["commute+shift"]


@needs_results
def test_drop_one_ranks_the_obvious_proxy_low() -> None:
    """Redundancy: the pair can stand in for zip, so removing zip alone barely changes the joint AUC."""
    d1 = json.load(open("results.json"))["screen"]
    single, drop = d1["single"], d1["drop_one"]
    assert max(single, key=lambda f: single[f]["mean"]) == "zip"
    assert drop["zip"]["mean"] < drop["commute"]["mean"] / 2


@needs_results
def test_seed_spread_is_reported_and_small_where_claimed() -> None:
    m = json.load(open("results.json"))["models"]
    for k in ("aware (group as a feature)", "drop zip + the commute/shift pair", "drop zip + pair, repair exp"):
        assert m[k]["ratio"]["hi"] - m[k]["ratio"]["lo"] < 0.05, k
    assert m["drop zip"]["ratio"]["hi"] - m["drop zip"]["ratio"]["lo"] > 0.1, "drop-zip is unstable; README says so"
