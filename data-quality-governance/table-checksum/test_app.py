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


def test_default_tables_are_the_same_data() -> None:
    """Shuffled rows, reordered columns, 12.50 vs 12.5, 7.00 vs 7, 0.30 vs 0.1+0.2: canonical says match."""
    at = _run()
    assert "Same data" in at.success[0].value
    assert "csv_md5" in at.success[0].value


def test_one_cent_reaches_the_warning_branch() -> None:
    at = _run()
    at.text_area(key="b").set_value("event_id,customer,note,amount\n1,Ada Lim,gift,12.51\n2,Ben Okafor,\\N,7\n"
                                     "3,Chloe Tan,,0.3\n").run()
    assert not at.exception
    assert "differ" in at.warning[0].value


def test_null_vs_empty_is_a_difference() -> None:
    at = _run()
    at.text_area(key="b").set_value("event_id,customer,note,amount\n1,Ada Lim,gift,12.5\n2,Ben Okafor,,7\n"
                                     "3,Chloe Tan,,0.3\n").run()
    assert at.warning[0].value.endswith("count_sums.")  # column sums never look at a text column


def test_column_mismatch_errors() -> None:
    at = _run()
    at.text_area(key="b").set_value("event_id,customer\n1,Ada Lim\n").run()
    assert "Different columns" in at.error[0].value


def test_empty_input_reaches_the_info_branch() -> None:
    at = _run()
    at.text_area(key="a").set_value("event_id,customer\n").run()
    assert "at least one row" in at.info[0].value
