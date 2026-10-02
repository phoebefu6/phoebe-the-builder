"""Engine tests. Each checker is also shown able to FAIL (a verifier must be tested against passing)."""

from __future__ import annotations

import migrate as M
import numpy as np
import pytest
from scipy import stats


def test_calibration_is_clean() -> None:
    cal = M.calibrate(6000)
    assert cal["conv_vs_binom_gap"] < 1e-12 and cal["fixed_mass_off_zero"] == 0
    assert sum(not r["inside_99"] for r in cal["rates"]) <= 1


def test_simulation_check_can_fail() -> None:
    """Simulating the WRONG migration must land outside the exact interval of the no-change rate."""
    pa, intent, kind = M.eval_set()
    pb, _ = M.scenario("net zero, spread", pa, intent, kind, 6)
    hits = int(M.decide(*M.simulate(pa, pb, 4000), intent)["churn vs rerun"].sum())
    lo, hi = M.wilson(hits, 4000)
    assert not lo <= M.gate_probs(pa, pa, intent)["churn vs rerun"] <= hi


def test_joint_pmf_sums_to_one_and_has_the_right_means() -> None:
    pa, pb = np.array([0.9, 0.2, 0.5]), np.array([0.1, 0.8, 0.5])
    pmf = M.joint_pmf(pa, pb)
    assert pmf.sum() == pytest.approx(1)
    b, f = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    assert (pmf * b).sum() == pytest.approx((pa * (1 - pb)).sum())
    assert (pmf * f).sum() == pytest.approx(((1 - pa) * pb).sum())


def test_mcnemar_table_matches_scipy() -> None:
    p = M.mcnemar_p(30)
    assert p[10, 2] == pytest.approx(stats.binomtest(2, 12, 0.5).pvalue)
    assert p[0, 0] == 1.0


def test_net_zero_migrations_keep_the_score_and_change_40_cases() -> None:
    pa, intent, kind = M.eval_set()
    for sc in ("net zero, concentrated", "net zero, spread"):
        pb, ch = M.scenario(sc, pa, intent, kind)
        assert pb.mean() == pytest.approx(pa.mean()) and ch.sum() == 2 * M.N_SWAP


def test_headline_findings() -> None:
    ev = {r["scenario"]: r["gates"] for r in M.evaluate()["scenarios"]}
    for sc in ("net zero, concentrated", "net zero, spread"):
        assert ev[sc]["mcnemar"] < ev["no change"]["mcnemar"]  # quieter than on no change at all
        assert ev[sc]["churn vs rerun"] > 0.99
    assert ev["no change"]["churn vs rerun"] <= M.ALPHA
    assert ev["net zero, concentrated"]["per-intent"] > 0.99 and ev["net zero, spread"]["per-intent"] < 0.01
    assert ev["plain regression"]["mcnemar"] > 0.9 and ev["plain regression"]["per-intent"] < 0.01


def test_audit_counts_and_guards() -> None:
    rows = [("a", 1, 0), ("a", 0, 1), ("b", 1, 1), ("b", 1, 0)]
    res = M.audit(rows, [1, 0, 1, 1])
    assert (res["broke"], res["fixed"], res["churn"]) == (2, 1, 3)
    assert res["odd_b"] == 3 and res["odd_r"] == 0 and res["noise_churn"] == 0
    for bad, rr in (([("a", 2, 1)], ()), ([], ()), ([("a", 1, 1)], [1, 1])):
        with pytest.raises(ValueError):
            M.audit(bad, rr)
