"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _run(scenario: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    if scenario:
        at.selectbox(key="scenario").set_value(scenario).run()
    assert not at.exception
    return at


def test_default_scd2_names_the_missed_check() -> None:
    at = _run()
    assert "declared_unique" in at.error[0].value and "report is wrong" in at.error[0].value


def test_orphan_dups_reaches_warning() -> None:
    at = _run("orphan_dups")
    assert not at.error and "false alarm" in at.warning[0].value


def test_clean_reaches_success() -> None:
    at = _run("clean")
    assert not at.error and not at.warning and "correct" in at.success[0].value
