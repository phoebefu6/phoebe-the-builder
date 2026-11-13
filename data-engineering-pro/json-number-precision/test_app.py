"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _run(value: str = "", context: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    if value:
        at.text_input(key="value").set_value(value).run()
    if context:
        at.selectbox(key="context").set_value(context).run()
    assert not at.exception
    return at


def test_default_snowflake_with_null_is_wrong_in_pandas() -> None:
    at = _run()
    assert "pandas read_json" in at.error[0].value and "JSON.parse" in at.error[0].value


def test_small_id_reaches_success() -> None:
    at = _run("12345", "clean")
    assert not at.error and "exactly" in at.success[0].value


def test_bad_input_reaches_warning() -> None:
    at = _run("abc")
    assert not at.error and "not an integer" in at.warning[0].value
