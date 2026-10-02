"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")
TIMEOUT = 120


def _run(text: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    if text:
        at.text_area(key="rows").set_value(text).run()
        assert not at.exception
    return at


def test_opens_on_the_concentrated_migration_and_names_refunds() -> None:
    at = _run()
    assert len(at.error) == 1 and "HIDDEN REGRESSION" in at.error[0].value and "refunds" in at.error[0].value


def test_score_moved_branch() -> None:
    at = _run("\n".join(["x,1,0"] * 12 + ["y,1,1"] * 30))
    assert any("SCORE MOVED" in e.value or "HIDDEN REGRESSION" in e.value for e in at.error)


def test_churn_branch() -> None:
    # 2 intents x 15 swaps each way, net zero, rerun of A identical: churn 30 vs noise 0
    rows = ["p,1,0,1"] * 8 + ["p,0,1,0"] * 7 + ["q,1,0,1"] * 7 + ["q,0,1,0"] * 8 + ["q,1,1,1"] * 40
    at = _run("\n".join(rows))
    assert any("BEHAVIOUR CHANGED" in w.value for w in at.warning)


def test_no_rerun_branch() -> None:
    at = _run("a,1,0\na,0,1\na,1,1")
    assert at.info and "NO RERUN GIVEN" in at.info[0].value


def test_clean_branch() -> None:
    at = _run("a,1,1,1\na,0,0,0\na,1,0,0\na,0,1,1")
    assert at.success and "NO DETECTABLE CHANGE" in at.success[0].value


def test_bad_input_warns_instead_of_crashing() -> None:
    for text, msg in (("a,1,2", "0 or 1"), ("a,1,1\nb,1,1,1", "same shape"), ("a,1", "same shape")):
        at = _run(text)
        assert any(msg in w.value for w in at.warning), msg
