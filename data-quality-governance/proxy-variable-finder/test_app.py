"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")


def _run(overrides: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    assert not at.exception
    if overrides:
        at.text_input(key="overrides").set_value(overrides)
        at.run()
        assert not at.exception
    return at


def test_default_flags_the_joint_proxy_and_the_unaware_gap() -> None:
    at = _run()
    assert "commute+shift" in at.error[0].value
    assert "Removing the group leaves" in at.warning[0].value
    assert len(at.table[0].value) == 7


def test_unbiased_world_with_no_pair_shows_both_clean_branches() -> None:
    at = _run("pair_link=0, label_bias=0, zip_gap=0, exp_gap=0")
    assert "No pair predicts" in at.success[0].value
    assert not at.warning


def test_bad_override_warns_not_crashes() -> None:
    at = _run("nope=1")
    assert "unknown parameter" in at.warning[0].value
