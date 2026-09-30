"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")
TIMEOUT = 120


def _run(a: str = "", b: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    if a:
        at.text_area(key="base").set_value(a)
        at.text_area(key="cand").set_value(b).run()
        assert not at.exception
    return at


def test_opens_on_the_hard_break_and_calls_it_real() -> None:
    at = _run()
    assert len(at.error) == 1 and "LIKELY REAL BREAK" in at.error[0].value


def test_noise_branch() -> None:
    at = _run("111\n101\n111", "111\n011\n111")
    assert any("DIFF IS NOISE" in w.value for w in at.warning)


def test_single_run_branch() -> None:
    at = _run("1\n1\n0", "0\n1\n0")
    assert at.info and "ONE RUN PER CASE" in at.info[0].value


def test_clean_branch() -> None:
    at = _run("111\n000", "111\n000")
    assert at.success and "CLEAN" in at.success[0].value


def test_bad_input_warns_instead_of_crashing() -> None:
    for a, b, msg in (("12\n11", "11\n11", "0s and 1s"), ("111\n11", "111\n11", "same number of runs"),
                      ("111\n111", "111", "same cases")):
        at = _run(a, b)
        assert any(msg in w.value for w in at.warning), msg
