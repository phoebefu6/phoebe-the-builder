"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")
TIMEOUT = 120


def _run() -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    return at


def _summary(at: AppTest, n: int, p: float) -> None:
    at.number_input(key="n").set_value(n)
    at.number_input(key="p").set_value(p).run()
    assert not at.exception


def test_opens_on_the_exemplar_paradox() -> None:
    at = _run()
    assert len(at.error) == 1 and "LINDLEY'S PARADOX" in at.error[0].value


def test_agree_branch() -> None:
    at = _run()
    _summary(at, 100, 0.0001)
    assert at.success and "BOTH ANALYSTS AGREE" in at.success[0].value


def test_evidence_of_absence_branch() -> None:
    at = _run()
    _summary(at, 5000, 0.6)
    assert at.info and "EVIDENCE OF ABSENCE" in at.info[0].value


def test_inconclusive_branch() -> None:
    at = _run()
    _summary(at, 30, 0.04)
    assert any("INCONCLUSIVE" in w.value for w in at.warning)


def test_raw_mode_runs() -> None:
    at = _run()
    at.radio(key="mode").set_value("Raw values").run()
    assert not at.exception
    assert at.text_area(key="values").value
    assert at.metric[0].value == "10"


def test_bad_input_warns_instead_of_crashing() -> None:
    at = _run()
    at.radio(key="mode").set_value("Raw values").run()
    at.text_area(key="values").set_value("1, 2").run()
    assert not at.exception
    assert any("at least 3" in w.value for w in at.warning)
