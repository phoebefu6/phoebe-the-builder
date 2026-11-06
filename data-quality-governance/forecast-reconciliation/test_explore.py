"""Engine tests: the algebra of reconciliation, and the findings the README states."""

from __future__ import annotations

import explore as E
import numpy as np


def test_summing_matrix_shape_and_rows() -> None:
    S = E.summing_matrix()
    assert S.shape == (17, 12)
    assert S[0].sum() == 12 and all(S[i].sum() == 3 for i in range(1, 5))


def test_every_reconciled_method_adds_up() -> None:
    w = E.make_world()
    yhat, res = E.base_forecasts(w)
    for m in E.METHODS[1:]:
        assert E.coherence_gap(w, E.reconcile(w, yhat, res, m)) < 1e-9, m
    assert E.coherence_gap(w, yhat) > 0.01


def test_raw_scale_linear_model_is_coherent_by_construction() -> None:
    w = E.make_world()
    assert E.coherence_gap(w, E.base_forecasts(w, log=False)[0]) < 1e-9


def test_projection_leaves_coherent_forecasts_unchanged() -> None:
    """Least-squares reconciliation is a projection: a forecast that already adds up must come back unchanged."""
    w = E.make_world()
    yhat, res = E.base_forecasts(w)
    coherent = w["S"] @ yhat[5:]
    for m in ("ols", "wls", "mint_shrink"):
        assert np.allclose(E.reconcile(w, coherent, res, m), coherent), m


def test_shrunk_covariance_is_positive_definite() -> None:
    w = E.make_world()
    _, res = E.base_forecasts(w)
    assert np.linalg.eigvalsh(E.shrink_cov(res)).min() > 0


def test_top_down_hist_is_worst_below_the_top() -> None:
    s = E.seeds(n=5)
    worst = max(E.METHODS, key=lambda m: s["mean"][m]["region"])
    assert worst == "top_down_hist"


def test_shock_is_missed_by_every_method() -> None:
    sh = E.shock_study(n=5)
    assert all(v > 0.08 for v in sh["median_bias"].values())


def test_shock_region_reported() -> None:
    assert E.make_world(shock=0.2)["shock_region"] in range(4)
    assert E.make_world()["shock_region"] is None
