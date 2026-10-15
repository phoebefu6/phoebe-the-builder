"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("evidence.txt"), reason="evidence.py has not been run")


def _run(overrides: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    assert not at.exception
    if overrides:
        at.text_input(key="overrides").set_value(overrides)
        at.run()
        assert not at.exception
    return at


def test_default_world_says_the_ruler_carries_the_price() -> None:
    at = _run()
    assert "biased ruler" in at.warning[0].value
    assert len(at.table[0].value) == 4


def test_honest_history_shows_the_agree_branch() -> None:
    at = _run("label_bias=0")
    assert "rulers agree" in at.success[0].value
    assert not at.warning


def test_bad_override_warns_not_crashes() -> None:
    at = _run("nope=1")
    assert "unknown parameter" in at.warning[0].value
