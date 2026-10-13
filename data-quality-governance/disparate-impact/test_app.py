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


def test_default_fails_four_fifths_and_warns_impossibility() -> None:
    at = _run()
    assert "four-fifths ratio 0.657" in at.error[0].value and "Fails" in at.error[0].value
    assert "cannot all hold" in at.info[0].value
    assert len(at.table[0].value) == 5


def test_identical_groups_pass_and_drop_the_impossibility_note() -> None:
    at = _run("B.base=0.40, B.mu_q=1.2, B.mu_u=-0.4")
    assert "Passes" in at.success[0].value
    assert not at.info


def test_bad_override_warns_not_crashes() -> None:
    at = _run("C.base=0.3")
    assert "unknown parameter" in at.warning[0].value
