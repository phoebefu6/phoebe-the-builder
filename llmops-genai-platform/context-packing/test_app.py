"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")
TIMEOUT = 180


def _run(policies=None) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    if policies is not None:
        for i, p in enumerate(policies):
            at.text_input(key=f"policy{i}").set_value(p)
        at.run()
        assert not at.exception
    return at


def test_opens_on_the_study_and_flags_the_flip() -> None:
    at = _run()
    assert len(at.error) == 1 and "RANKING FLIPS" in at.error[0].value and "policy 2" in at.error[0].value


def test_single_default_policy_warns_present_but_unread() -> None:
    at = _run(["4000, rank, top", "", ""])
    assert at.warning and "PRESENT BUT UNREAD" in at.warning[0].value


def test_consistent_branch() -> None:
    at = _run(["1500, rank, top", "", ""])
    assert at.success and "CONSISTENT" in at.success[0].value


def test_bad_input_warns_instead_of_crashing() -> None:
    at = _run(["4000, rank", "", ""])
    assert any("3 comma-separated fields" in w.value for w in at.warning)
    at = _run(["4000, shuffle, top", "", ""])
    assert any("order must be" in w.value for w in at.warning)
    at = _run(["", "", ""])
    assert any("at least one policy" in w.value for w in at.warning)


def test_table_shows_every_policy() -> None:
    at = _run()
    body = "".join(str(t.value) for t in at.table)
    assert "policy 1: 4000, rank, top" in body and "policy 3: 4000, reorder, both" in body
