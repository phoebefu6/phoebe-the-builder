"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")


def _run(overrides: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=240)
    at.run()
    assert not at.exception
    if overrides:
        at.text_input(key="overrides").set_value(overrides)
        at.run()
        assert not at.exception
    return at


def test_default_names_the_best_policy_and_both_traps() -> None:
    at = _run()
    assert "551 tickets/day" in at.success[0].value and "refund below 0.875" in at.success[0].value
    assert "35,153" in at.success[0].value and "\\$" in at.success[0].value
    assert "accuracy rule" in at.error[0].value
    assert "Floods a 700/day team" in at.warning[0].value


def test_a_huge_team_removes_the_flood_warning() -> None:
    """With no capacity limit nothing floods, but 'fill the team' now escalates every ticket and loses to doing nothing."""
    at = _run("capacity=20000")
    assert not at.warning, "nothing should overflow a 20,000/day team"
    assert at.error[0].value == "Costs more than never escalating: staffing rule: fill capacity."


def test_bad_override_warns_not_crashes() -> None:
    at = _run("nope=3")
    assert "unknown parameter" in at.warning[0].value
    assert not at.success


def test_table_has_seven_policies() -> None:
    at = _run()
    assert len(at.table[0].value) == 7
