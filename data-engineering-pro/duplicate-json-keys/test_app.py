"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _run(doc: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    if doc:
        at.text_area(key="doc").set_value(doc).run()
    assert not at.exception
    return at


def test_default_role_escalation_disagrees() -> None:
    at = _run()
    assert '"admin"' in at.error[0].value and '"viewer"' in at.error[0].value


def test_no_duplicate_reaches_success() -> None:
    at = _run('{"a":1,"b":2}')
    assert not at.error and "No top-level key repeats" in at.success[0].value


def test_same_value_twice_reaches_info() -> None:
    at = _run('{"a":1,"a":1}')
    assert not at.error and "same value" in at.info[0].value


def test_bad_json_and_array_reach_warning() -> None:
    assert "Not valid JSON" in _run("{nope").warning[0].value
    assert "JSON object" in _run("[1,2]").warning[0].value
