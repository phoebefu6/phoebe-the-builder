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


def test_default_rows_are_unique_and_warn() -> None:
    at = _run()
    assert "are unique on this key" in at.warning[0].value


def test_sector_and_sex_still_leave_a_unique_row() -> None:
    """300 rows, 40 classes: even the study's finest k-anonymous key is not k-anonymous on a smaller export."""
    at = _run()
    at.selectbox(key="pl").set_value("sector (2 digits)")
    at.selectbox(key="dl").set_value("dropped")
    at.slider(key="k").set_value(2).run()
    assert not at.exception
    assert "1 of 300 rows are unique" in at.warning[0].value


def test_coarse_key_reaches_the_success_branch() -> None:
    at = _run()
    at.selectbox(key="pl").set_value("sector (2 digits)")
    at.selectbox(key="dl").set_value("dropped")
    at.selectbox(key="sl").set_value("dropped")
    at.slider(key="k").set_value(2).run()
    assert not at.exception
    assert "k >= 2" in at.success[0].value


def test_small_classes_without_uniques_warn_differently() -> None:
    at = _run()
    at.text_area(key="rows").set_value(
        "postcode,dob,sex\n" + "\n".join(["010101,1980-01-01,F"] * 3 + ["020202,1990-05-05,M"] * 3)
    ).run()
    assert not at.exception
    assert "sit in classes smaller than" in at.warning[0].value


def test_empty_input_reaches_the_info_branch() -> None:
    at = _run()
    at.text_area(key="rows").set_value("postcode,dob,sex\n").run()
    assert "at least one row" in at.info[0].value
