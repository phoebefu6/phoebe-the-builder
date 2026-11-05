"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _run(method: str = "") -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    if method:
        at.selectbox(key="method").set_value(method).run()
    assert not at.exception
    return at


def test_default_naive_walk_misses_the_ledger() -> None:
    at = _run()
    assert "naive_walk" in at.error[0].value and "counted twice" in at.error[0].value
    assert "yes" in at.info[0].value


def test_plug_reaches_warning() -> None:
    at = _run("plug")
    assert not at.error and "plug of" in at.warning[0].value


def test_validated_reaches_success() -> None:
    at = _run("validated")
    assert not at.error and not at.warning and "Unassigned" in at.success[0].value
    assert any(r == "Unassigned" for r in at.table[0].value["bucket"])
