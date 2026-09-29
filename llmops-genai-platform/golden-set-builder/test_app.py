"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")
TIMEOUT = 120
HEAD = "intent,cases,passes,traffic\n"


def _run(csv: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    if csv:
        at.text_area(key="table").set_value(HEAD + csv).run()
        assert not at.exception
    return at


def test_opens_on_the_month_12_set_with_a_coverage_gap() -> None:
    at = _run()
    assert len(at.error) == 1 and "COVERAGE GAP" in at.error[0].value
    assert "voice agent" in at.error[0].value


def test_stale_mix_branch() -> None:
    at = _run("a,100,95,100\nb,100,60,900")
    assert any("STALE MIX" in w.value for w in at.warning)


def test_thin_branch() -> None:
    at = _run("a,100,90,900\nb,5,4,100")
    assert at.info and "THIN" in at.info[0].value


def test_representative_branch() -> None:
    at = _run("a,50,45,500\nb,50,45,500")
    assert at.success and "REPRESENTATIVE" in at.success[0].value


def test_bad_input_warns_instead_of_crashing() -> None:
    for bad in ("a,10,11,5", "a,x,1,5"):
        at = _run(bad)
        assert any("Cannot audit" in w.value for w in at.warning)
    at = _run()
    at.text_area(key="table").set_value("name,n\na,1").run()
    assert not at.exception and any("header must be" in w.value for w in at.warning)
