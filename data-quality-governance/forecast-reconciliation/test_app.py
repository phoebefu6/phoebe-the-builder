"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _run(method: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    if method:
        at.selectbox(key="method").set_value(method).run()
    assert not at.exception
    return at


def test_default_base_does_not_add_up() -> None:
    at = _run()
    assert "do not add up" in at.error[0].value
    assert any(r == "sum of regions" for r in at.table[0].value["region"])


def test_top_down_hist_reaches_warning() -> None:
    at = _run("top_down_hist")
    assert not at.error and "tests arithmetic, not accuracy" in at.warning[0].value


def test_mint_reaches_success() -> None:
    at = _run("mint_shrink")
    assert not at.error and not at.warning and "adds up" in at.success[0].value
