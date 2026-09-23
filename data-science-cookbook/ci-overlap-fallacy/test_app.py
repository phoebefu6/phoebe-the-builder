"""Headless Streamlit checks, widgets addressed BY KEY.

An index-addressed widget silently moves when the layout changes, and `AppTest` accepts an
out-of-range set_value by doing nothing - so every branch below is also checked for being
REACHABLE, not just for not crashing.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"),
                                reason="evidence.py has not been run")

TIMEOUT = 120


def _run() -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    return at


def test_default_chart_is_the_dead_zone() -> None:
    """Opens on 30/30, d = 0.55: significant, bars overlap. The error must show untouched."""
    at = _run()
    assert len(at.error) == 1
    assert "IS significant" in at.error[0].value


def test_agreeing_branch_is_reachable() -> None:
    at = _run()
    at.slider(key="diff").set_value(1.5).run()
    assert not at.error
    assert len(at.success) == 1


def test_matching_level_closes_the_dead_zone() -> None:
    at = _run()
    at.slider(key="diff").set_value(0.60).run()
    at.select_slider(key="level").set_value(0.834).run()
    assert not at.error


def test_paired_branch_explains_the_slack() -> None:
    at = _run()
    at.slider(key="rho").set_value(0.9).run()
    assert not at.exception
    assert any("further apart" in i.value for i in at.info)
    assert any(m.value == "no such level" for m in at.metric)


def test_wald_filter_shortens_the_table() -> None:
    at = _run()
    full = len(at.dataframe[2].value)
    at.checkbox(key="only_bad").set_value(True).run()
    assert 0 < len(at.dataframe[2].value) < full
