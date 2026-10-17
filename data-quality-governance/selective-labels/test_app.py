"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("evidence.txt"), reason="evidence.py has not been run")


def _run(world: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    assert not at.exception
    if world:
        at.text_input(key="world").set_value(world)
        at.run()
        assert not at.exception
    return at


def test_default_world_shows_the_ranking_flip() -> None:
    at = _run()
    assert "rank the worse model first" in at.warning[0].value
    assert len(at.table[0].value) == 5


def test_control_world_shows_ipw_recovering() -> None:
    at = _run("sel_u=0")
    assert "IPW recovers" in at.success[0].value


def test_unrecorded_selection_without_a_bar_shows_the_info_branch() -> None:
    at = _run("bar_b=0")
    assert at.info and "IPW is off" in at.info[0].value


def test_bad_input_warns_not_crash() -> None:
    at = _run("nope=1")
    assert "unknown setting" in at.warning[0].value
