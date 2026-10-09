"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")


def _run(overrides: str = "", slo: int = 0) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    assert not at.exception
    if overrides:
        at.text_input(key="overrides").set_value(overrides)
    if slo:
        at.number_input(key="slo").set_value(slo)
    at.run()
    assert not at.exception
    return at


def test_default_is_over_slo_and_flags_the_blame_flip() -> None:
    at = _run()
    assert "Over SLO by 51 ms" in at.error[0].value
    assert "rerank" in at.warning[0].value and "llm" in at.warning[0].value


def test_percentiles_do_not_add_branch() -> None:
    at = _run(slo=1600)
    assert "percentiles do not add" in at.success[0].value


def test_both_within_branch() -> None:
    at = _run(slo=3000)
    assert "on both readings" in at.success[0].value


def test_bad_override_warns_not_crashes() -> None:
    at = _run(overrides="nope=3")
    assert "unknown parameter" in at.warning[0].value
