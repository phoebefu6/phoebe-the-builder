"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _run(value: str = "", writer: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    if value:
        at.text_input(key="value").set_value(value).run()
    if writer:
        at.selectbox(key="writer").set_value(writer).run()
    assert not at.exception
    return at


def test_default_15g_changes_point_three() -> None:
    at = _run()
    assert "changed" in at.error[0].value and "0.3" in at.error[0].value


def test_repr_reaches_success() -> None:
    at = _run(writer="repr")
    assert not at.error and "bit for bit" in at.success[0].value


def test_nan_in_sqlite_is_lost() -> None:
    at = _run("nan", "sqlite REAL")
    assert "lost" in at.error[0].value


def test_bad_input_reaches_warning() -> None:
    at = _run("twelve")
    assert not at.error and "Could not read" in at.warning[0].value
