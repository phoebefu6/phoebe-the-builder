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


def test_default_rows_show_ambiguity() -> None:
    at = _run()
    assert "3 of 4 numbers are valid in more than one country" in at.warning[0].value


def test_plus_numbers_reach_the_success_branch() -> None:
    at = _run()
    at.text_area(key="rows").set_value("+44 20 7946 0958, GB\n+1 202 555 0123,").run()
    assert not at.exception
    assert "carries its own country code" in at.success[0].value


def test_empty_input_reaches_the_info_branch() -> None:
    at = _run()
    at.text_area(key="rows").set_value("  ").run()
    assert "at least one phone" in at.info[0].value


def test_unknown_region_reaches_the_error_branch() -> None:
    at = _run()
    at.text_input(key="hq").set_value("XX").run()
    assert "not one of the 245" in at.error[0].value


def test_other_region_renders_table() -> None:
    at = _run()
    at.text_input(key="hq").set_value("gb").run()
    assert not at.exception and not at.error
