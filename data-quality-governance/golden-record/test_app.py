"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _run() -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    assert not at.exception
    return at


def test_default_cluster_shows_contested_fields() -> None:
    at = _run()
    msg = at.warning[0].value
    for f in ("name", "phone", "address"):
        assert f in msg
    assert "dob" not in msg.split("Choosing")[0]


def test_identical_records_reach_success() -> None:
    at = _run()
    row = "Ada Lim,1990-02-03,ada@x.test,+44 7700 900002,\"1 Elm Row, A1 1XY\""
    at.text_area(key="cluster").set_value(f"source,updated_at,name,dob,email,phone,address\ncrm,1,{row}\nweb,2,{row}\n").run()
    assert not at.exception
    assert "same golden record" in at.success[0].value


def test_unknown_source_errors() -> None:
    at = _run()
    at.text_area(key="cluster").set_value("source,updated_at,name\nerp,1,Ada\n").run()
    assert "unknown source" in at.error[0].value


def test_single_record_reaches_info() -> None:
    at = _run()
    at.text_area(key="cluster").set_value("source,updated_at,name\ncrm,1,Ada\n").run()
    assert "at least two records" in at.info[0].value
