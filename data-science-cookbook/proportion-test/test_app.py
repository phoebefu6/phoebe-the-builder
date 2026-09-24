"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

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


def test_default_table_is_the_disagreement() -> None:
    """Opens on 0/20 vs 4/20 - z and Barnard say moved, Fisher and Yates do not."""
    at = _run()
    assert len(at.error) == 1
    assert "DISAGREE" in at.error[0].value


def test_agreeing_branch_is_reachable() -> None:
    at = _run()
    at.number_input(key="x2").set_value(12).run()
    assert not at.error
    assert "agree" in at.success[0].value


def test_degenerate_table_warns() -> None:
    at = _run()
    at.number_input(key="x2").set_value(0).run()
    assert any("every test returns p = 1" in w.value for w in at.warning)


def test_rule_filter_shortens_the_table() -> None:
    at = _run()
    full = len(at.dataframe[-1].value)
    at.checkbox(key="only_broken").set_value(True).run()
    assert 0 < len(at.dataframe[-1].value) < full
