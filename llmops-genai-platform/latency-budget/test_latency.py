"""Tests for the exact latency engine."""

from __future__ import annotations

import latency as L
import numpy as np
import pytest


def test_every_pmf_sums_to_one_and_is_nonnegative() -> None:
    for prm in [L.BASE] + [{**L.BASE, **v} for v in L.FIXES.values()]:
        pm = L.hop_pmfs(prm)
        for h, p in pm.items():
            assert abs(p.sum() - 1) < 1e-9, h
            assert (p >= -1e-15).all(), h
        assert abs(L.chain(pm).sum() - 1) < 1e-9


def test_grid_tail_is_negligible() -> None:
    """Mass folded into the last bin must be tiny, or every percentile near the edge is wrong."""
    assert L.chain(L.hop_pmfs(L.BASE))[-1] < 1e-9


def test_lognormal_median_lands_on_the_median() -> None:
    assert L.percentile(L.lognormal_pmf(900.0, 0.12), 0.5) == 900


def test_convolution_mean_is_sum_of_means() -> None:
    pm = L.hop_pmfs(L.BASE)
    assert abs(L.mean(L.chain(pm)) - sum(L.mean(pm[h]) for h in L.HOPS)) < 1e-6


def test_fan_out_is_max_of_shards() -> None:
    one = L.shard_pmf(L.BASE)
    s = L.hop_pmfs(L.BASE)["search"]
    assert L.percentile(s, 0.5) > L.percentile(one, 0.5)
    assert abs(np.cumsum(s)[100] - np.cumsum(one)[100] ** L.SHARDS) < 1e-12


def test_hedging_never_makes_a_shard_slower() -> None:
    a = np.cumsum(L.shard_pmf(L.BASE))
    b = np.cumsum(L.shard_pmf({**L.BASE, "hedge_after_ms": 80.0}))
    assert (b >= a - 1e-12).all()


def test_retry_mass_is_hang_probability_when_body_never_times_out() -> None:
    r = L.hop_pmfs(L.BASE)["rerank"]
    assert abs(r[400:].sum() - (0.03 + 0.97 * L.lognormal_pmf(60, 0.3)[401:].sum())) < 1e-9


def test_percentiles_do_not_add() -> None:
    s = L.study()
    assert s["sum_of_hop_p99"] - s["e2e"]["p99"] > 500
    assert s["sum_of_hop_p99"] > L.SLO_MS + 700 and 0 < s["e2e"]["p99"] - L.SLO_MS < 100


def test_tail_attribution_sums_to_conditional_excess() -> None:
    pm = L.hop_pmfs(L.BASE)
    total = L.chain(pm)
    t = L.percentile(total, 0.99)
    ks = np.arange(len(total))
    tail = total[t + 1:]
    cond = float(np.dot(ks[t + 1:], tail) / tail.sum()) - L.mean(total)
    assert abs(sum(L.tail_attribution(pm, t).values()) - cond) < 1e-6


def test_blame_flips_between_mean_and_tail() -> None:
    s = L.study()
    assert max(L.HOPS, key=lambda h: s["mean_share"][h]) == "llm"
    assert max(L.HOPS, key=lambda h: s["tail_share"][h]) == "rerank"
    assert max(L.HOPS, key=lambda h: s["hops"][h]["p99"] / s["hops"][h]["p50"]) == "embed"


def test_every_shard_healthy_but_step_is_not() -> None:
    s = L.study()
    assert s["shard"]["p99"] < 100 and s["hops"]["search"]["p95"] > 250


def test_ranking_flips_between_mean_and_p99() -> None:
    rows = {r["fix"]: r for r in L.fix_table()}
    llm, cfg = rows["smaller LLM (-15% median)"], rows["all three config fixes"]
    assert llm["d_mean"] < 5 * cfg["d_mean"]
    assert cfg["d_p99"] < llm["d_p99"] and cfg["p_over_slo"] < llm["p_over_slo"]


def test_negative_result_no_single_config_fix_beats_llm_on_p99() -> None:
    rows = {r["fix"]: r for r in L.fix_table()}
    for k in ("warm embedding pool", "hedge shard calls at 80 ms", "rerank timeout 400 -> 120 ms"):
        assert rows[k]["d_p99"] > rows["smaller LLM (-15% median)"]["d_p99"]


def test_exact_matches_simulation() -> None:
    cal = L.calibrate(n=50_000)
    assert sum(r["inside"] for r in cal) >= len(cal) - 1


def test_parse_params_refuses_bad_input() -> None:
    assert L.parse_params("llm_median=700")["llm_median"] == 700
    for bad in ("nope=1", "llm_median", "embed_cold_p=1.5", "llm_sigma=-1"):
        with pytest.raises(ValueError):
            L.parse_params(bad)
