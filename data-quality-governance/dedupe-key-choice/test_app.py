"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _run() -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=90)
    at.run()
    assert not at.exception
    return at


def test_default_rows_show_disagreement() -> None:
    at = _run()
    assert "anywhere from" in at.warning[0].value


def test_agreeing_rows_reach_the_success_branch() -> None:
    at = _run()
    at.text_area(key="rows").set_value("Elena Petrova\nPriya Raman").run()
    assert not at.exception
    assert "the keys agree" in at.success[0].value


def test_empty_input_reaches_the_info_branch() -> None:
    at = _run()
    at.text_area(key="rows").set_value("  ").run()
    assert "at least one name" in at.info[0].value
