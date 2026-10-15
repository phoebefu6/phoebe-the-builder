"""Tests for the frontier engine. Fast properties on small samples, then the study's headlines from results.json."""

from __future__ import annotations

import json
import os

import frontier as F
import numpy as np
import pytest

D = F.generate(n=30_000, seed=7)


def test_world_has_a_real_gap_and_the_history_adds_bias_only() -> None:
    g = D["g"]
    assert 0.15 < D["skill"][g == 0].mean() - D["skill"][g == 1].mean() < 0.35
    # same noise draw: for group A the two labels are identical; for B history is never positive when truth is not
    assert (D["y_hist"][g == 0] == D["y_true"][g == 0]).all()
    assert (D["y_hist"][g == 1] <= D["y_true"][g == 1]).all()


def test_select_at_ratio_hits_the_total_and_the_ratio() -> None:
    rng = np.random.default_rng(0)
    s, g = rng.random(10_000), (rng.random(10_000) < 0.3).astype(int)
    for r in (0.2, 0.8, 1.0, 1.25):
        sel = F.select_at_ratio(s, g, r)
        assert sel.sum() == 3_000
        assert abs(sel[g == 1].mean() / sel[g == 0].mean() - r) < 0.01


def test_select_at_ratio_takes_the_top_within_each_group() -> None:
    rng = np.random.default_rng(1)
    s, g = rng.random(5_000), (rng.random(5_000) < 0.3).astype(int)
    sel = F.select_at_ratio(s, g, 0.6)
    for gv in (0, 1):
        m = g == gv
        assert s[m & sel].min() >= s[m & ~sel].max()


def test_reweighing_makes_group_and_label_independent() -> None:
    n = len(D["g"])
    g, y = D["g"][: n // 2], D["y_hist"][: n // 2]
    wt = np.empty(len(y))
    for gv in (0, 1):
        for yv in (0, 1):
            m = (g == gv) & (y == yv)
            wt[m] = (g == gv).mean() * (y == yv).mean() / m.mean()
    rate = lambda gv: (wt * y * (g == gv)).sum() / (wt * (g == gv)).sum()  # noqa: E731
    assert abs(rate(0) - rate(1)) < 1e-9


def test_honest_history_makes_the_rulers_agree() -> None:
    """The yardstick decomposition must vanish when there is no yardstick error - a detector that always fires is
    not a detector."""
    r = F.one_seed(7, 30_000, F.parse_world("label_bias=0"), with_audit=False)
    p = r["prices"]
    assert p["parity_hit_hist"] == pytest.approx(p["parity_hit_true"])


def test_no_gap_no_bias_puts_the_free_choice_near_parity() -> None:
    r = F.one_seed(7, 30_000, F.parse_world("label_bias=0, skill_gap=0, exp_gap=0"), with_audit=False)
    assert 0.85 < r["start"] < 1.15
    assert abs(r["prices"]["parity_hit_true"]) < 0.6


def test_parse_world_rejects_bad_input() -> None:
    for bad in ("nope=1", "skill_gap=x", "share_b=1.5", "skill_gap"):
        with pytest.raises(ValueError):
            F.parse_world(bad)


R = json.load(open("results.json")) if os.path.exists("results.json") else None
needs = pytest.mark.skipif(R is None, reason="evidence.py has not been run")


@needs
def test_headline_parity_costs_on_history_and_pays_on_truth() -> None:
    p = R["prices"]
    assert p["parity_hit_hist"]["lo"] > 3.5 and p["parity_hit_true"]["hi"] < -1.0


@needs
def test_control_price_is_small_and_ruler_part_grows_with_bias() -> None:
    sw = R["sweep"]
    assert abs(sw[0]["parity_hit_true"]["mean"]) < 1.0
    ruler = [row["parity_hit_hist"]["mean"] - row["parity_hit_true"]["mean"] for row in sw]
    assert ruler[0] == pytest.approx(0) and all(b > a for a, b in zip(ruler, ruler[1:]))


@needs
def test_single_cutoff_models_sit_near_the_frontier() -> None:
    assert all(abs(v["mean"]) < 1.0 for v in R["gap_below"].values())


@needs
def test_small_audit_is_wider_than_large() -> None:
    a = R["audit"]
    assert a["2000"]["p95"]["mean"] - a["2000"]["p05"]["mean"] > 3 * (a["20000"]["p95"]["mean"] - a["20000"]["p05"]["mean"])
