"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")
TIMEOUT = 120


def _run(classes: str = "", policies=None) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    if classes or policies:
        if classes:
            at.text_area(key="classes").set_value(classes)
        for i, p in enumerate(policies or []):
            at.text_input(key=f"policy{i}").set_value(p)
        at.run()
        assert not at.exception
    return at


def test_opens_on_the_study_and_flags_the_flip() -> None:
    at = _run()
    assert len(at.error) == 1 and "RANKING FLIPS" in at.error[0].value and "policy 3" in at.error[0].value


def test_single_policy_hot_corpus_warns_answers_staler_than_index() -> None:
    at = _run(policies=["30", "", ""])
    assert at.warning and "ANSWERS STALER THAN THE INDEX" in at.warning[0].value


def test_consistent_branch() -> None:
    at = _run("a, 100, 30, 0.5, 1\nb, 100, 30, 0.5, 1", ["30", "7", ""])
    assert at.success and "CONSISTENT" in at.success[0].value


def test_bad_input_warns_instead_of_crashing() -> None:
    at = _run("a, 10, 14", ["30", "", ""])
    assert any("5 comma-separated fields" in w.value for w in at.warning)
    at = _run(policies=["hot=1", "", ""])
    assert any("no period for" in w.value for w in at.warning)
    at = _run(policies=["", "", ""])
    assert any("at least one policy" in w.value for w in at.warning)


def test_table_shows_every_policy() -> None:
    at = _run()
    body = "".join(str(t.value) for t in at.table)
    assert "policy 1: 30" in body and "policy 3: hot=1, warm=7, cold=90" in body
