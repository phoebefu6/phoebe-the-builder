"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("evidence.txt"), reason="evidence.py has not been run")


def _run(shares: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    assert not at.exception
    if shares:
        at.text_input(key="shares").set_value(shares)
        at.run()
        assert not at.exception
    return at


def test_default_quota_says_marginals_pass_and_cells_fail() -> None:
    at = _run()
    assert "one-attribute audit passes" in at.warning[0].value
    assert len(at.table[0].value) == 6


def test_proportional_shares_show_the_ok_branch() -> None:
    at = _run("A-urban=0.42, A-rural=0.28, B-urban=0.18, B-rural=0.12")
    assert "inside [0.80, 1.25]" in at.success[0].value
    assert not at.warning


def test_marginal_miss_shows_the_plain_flag_branch() -> None:
    at = _run("A-urban=0.52, A-rural=0.38, B-urban=0.06, B-rural=0.04")
    assert at.warning[0].value.startswith("Flagged:") and "group B" in at.warning[0].value


def test_bad_shares_warn_not_crash() -> None:
    at = _run("B-rural=0.5")
    assert "sum to 1" in at.warning[0].value
