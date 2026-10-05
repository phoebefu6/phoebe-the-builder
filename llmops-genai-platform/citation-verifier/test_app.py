"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")
TIMEOUT = 120
P = "[1] Pro plan includes 5 seats.\n[2] Basic plan includes 2 seats."


def _run(passages: str = "", claims: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    if passages or claims:
        if passages:
            at.text_area(key="passages").set_value(passages)
        if claims:
            at.text_area(key="claims").set_value(claims)
        at.run()
        assert not at.exception
    return at


def test_opens_on_v2_sample_and_flags_wrong_citations() -> None:
    at = _run()
    assert len(at.error) == 1 and "CITATION DOES NOT SAY THAT" in at.error[0].value


def test_clean_branch_says_what_it_did_not_check() -> None:
    at = _run(P, "Pro plan comes with 5 seats. [1]\nBasic plan includes 2 seats. [2]")
    assert at.success and "LEXICALLY CONSISTENT" in at.success[0].value and "NOT checked" in at.success[0].value


def test_uncited_branch() -> None:
    at = _run(P, "Pro plan includes 5 seats.")
    assert any("UNCITED" in w.value for w in at.warning)


def test_wrong_number_and_polarity_are_named() -> None:
    at = _run(P, "Pro plan includes 2 seats. [1]\nPro plan does not include 5 seats. [1]")
    assert at.error
    body = "".join(str(t.value) for t in at.table)
    assert "number the cited passage does not hold" in body and "polarity differs" in body


def test_bad_input_warns_instead_of_crashing() -> None:
    at = _run(P, "Pro plan includes 5 seats. [9]")
    assert any("names a passage that is not there" in w.value for w in at.warning)
    at = _run("   ", "x [1]")
    assert any("at least one passage" in w.value for w in at.warning)
