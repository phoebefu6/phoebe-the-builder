"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")
TIMEOUT = 180


def _run(a: str = "", b: str = "", mix: str = "", new_phrasing=None, detector: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    if a or b or mix or new_phrasing is not None or detector:
        if a:
            at.text_input(key="version_a").set_value(a)
        if b:
            at.text_input(key="version_b").set_value(b)
        if mix:
            at.text_input(key="mix").set_value(mix)
        if new_phrasing is not None:
            at.checkbox(key="new_phrasing").set_value(new_phrasing)
        if detector:
            at.selectbox(key="detector").set_value(detector)
        at.run()
        assert not at.exception
    return at


def test_opens_on_the_study_and_flags_the_wrong_sign() -> None:
    at = _run()
    assert len(at.error) == 1 and "MEASURED SIGN IS WRONG" in at.error[0].value


def test_llm_judge_fixes_the_sign_and_names_the_over_refusal() -> None:
    at = _run(detector="LLM judge")
    assert not at.error and at.warning and "RATE ROSE, OVER-REFUSAL ROSE" in at.warning[0].value


def test_flat_twin_warns() -> None:
    at = _run(b="0.012, 0.24, 0.444", new_phrasing=False)
    assert at.warning and "FLAT AGGREGATE HIDES A SWAP" in at.warning[0].value


def test_tuned_down_warns_under_refusal() -> None:
    at = _run(b="0.003, 0.06, 0.70", new_phrasing=False)
    assert at.warning and "RATE FELL, UNDER-REFUSAL ROSE" in at.warning[0].value


def test_consistent_branch() -> None:
    at = _run(b="0.003, 0.06, 0.95", new_phrasing=False)
    assert at.success and "CONSISTENT" in at.success[0].value


def test_bad_input_warns_instead_of_crashing() -> None:
    at = _run(a="0.1, 0.2")
    assert any("3 comma-separated refusal rates" in w.value for w in at.warning)
    at = _run(mix="1, 1, 0")
    assert any("harmful > 0" in w.value for w in at.warning)


def test_table_shows_both_versions() -> None:
    at = _run()
    body = "".join(str(t.value) for t in at.table)
    assert "A" in body and "B" in body and "%" in body
