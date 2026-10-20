"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _run() -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_default_names_show_disagreement() -> None:
    at = _run()
    assert "disagree on 4 of 4" in at.warning[0].value
    assert 'rsplit(" ", 1), greeting 38.50%' in at.warning[1].value


def test_plain_name_reaches_the_success_branch() -> None:
    at = _run()
    at.text_area(key="names").set_value("Grace Holloway").run()
    assert not at.exception
    assert "All six splitters agree" in at.success[0].value


def test_empty_input_reaches_the_info_branch() -> None:
    at = _run()
    at.text_area(key="names").set_value("  ").run()
    assert "Paste at least one" in at.info[0].value


def test_zero_mix_reaches_the_error_branch() -> None:
    at = _run()
    for g in __import__("explore").SHARE:
        at.number_input(key=f"share_{g}").set_value(0.0)
    at.run()
    assert "more than zero" in at.error[0].value
