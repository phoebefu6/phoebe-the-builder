"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")
TIMEOUT = 120
REF = "lookup_order:A17;check_policy;issue_refund:A17"


def _run(text: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    if text:
        at.text_area(key="runs").set_value(text).run()
        assert not at.exception
    return at


def test_opens_on_v2_sample_and_flags_outcome() -> None:
    at = _run()
    assert len(at.error) == 1 and "OUTCOME SCORE OVERSTATES" in at.error[0].value


def test_strict_branch() -> None:
    at = _run(f"1,1,{REF}\n1,1,check_policy;lookup_order:A17;issue_refund:A17")
    assert any("STRICT MATCH UNDERSTATES" in w.value for w in at.warning)


def test_clean_branch() -> None:
    at = _run(f"1,1,{REF}\n0,1,lookup_order:A17;check_policy")
    assert at.success and "PATHS CLEAN" in at.success[0].value


def test_bad_input_warns_instead_of_crashing() -> None:
    for text in ("2,1,lookup_order", "1,1", "yes,no,x"):
        at = _run(text)
        assert any("0 or 1" in w.value for w in at.warning), text
