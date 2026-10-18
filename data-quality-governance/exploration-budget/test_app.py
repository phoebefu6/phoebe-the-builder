"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")


def _run(**vals) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    assert not at.exception
    for k, v in vals.items():
        (at.slider if k == "target" else at.number_input)(key=k).set_value(v)
    if vals:
        at.run()
        assert not at.exception
    return at


def test_default_names_both_designs_and_the_fairness_caveat() -> None:
    at = _run()
    msg = at.warning[0].value
    assert "Targeted 2.0%" in msg and "uniform needs 2.0%" in msg and "TPR" in msg
    assert len(at.table[0].value) == 12


def test_profitable_economics_show_the_success_branch() -> None:
    at = _run(gain=2_000.0, loss=500.0)
    assert "pays for itself" in at.success[0].value


def test_higher_confidence_moves_both_designs_up_the_grid() -> None:
    """The info branch is a guard only: both designs reach 100% at the top of this grid, so it cannot fire."""
    at = _run(target=0.99)
    msg = at.warning[0].value
    assert "Targeted 5.0%" in msg and "uniform needs 5.0%" in msg
