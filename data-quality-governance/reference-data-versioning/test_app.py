"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _run() -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    assert not at.exception
    return at


def test_default_month_hits_a_backdated_move() -> None:
    at = _run()
    assert "overwrite" in at.warning[0].value and "scd2_asof" in at.warning[0].value
    assert "recorded after" in at.error[0].value


def test_early_month_reaches_success() -> None:
    at = _run()
    at.slider(key="month").set_value(0).run()
    assert not at.exception
    assert "scd2_asof" not in at.warning[0].value
    assert "reproduces this month" in at.success[0].value


def test_table_has_published_plus_five_rows() -> None:
    at = _run()
    assert len(at.table[0].value) == 6
