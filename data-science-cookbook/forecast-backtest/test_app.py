"""The Streamlit app has to actually run. A scheduled build cannot open a browser, so this
drives it headless through Streamlit's own AppTest harness.

Widgets are addressed by KEY, never by index: AppTest returns main-body widgets before sidebar
ones regardless of source order, and an out-of-range set_value is silently ignored rather than
raising - between them, an index-based test can pass while exercising nothing (Day 171).
"""

from __future__ import annotations

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def fresh():
    at = AppTest.from_file("app.py", default_timeout=240)
    at.run()
    assert not at.exception
    return at


def test_app_runs_without_exception():
    fresh()


def test_the_key_addressing_this_file_relies_on_actually_works():
    """Guard the guard: if set_value stopped taking, every test below would pass vacuously."""
    at = fresh()
    at.slider(key="h").set_value(6).run()
    assert at.slider(key="h").value == 6


def test_app_runs_in_every_world():
    at = fresh()
    for w in at.radio(key="world").options:
        at.radio(key="world").set_value(w).run()
        assert not at.exception, f"world {w} raised"


def test_app_runs_for_every_forecaster():
    at = fresh()
    for m in at.selectbox(key="model").options:
        at.selectbox(key="model").set_value(m).run()
        assert not at.exception, f"model {m} raised"


def test_app_runs_under_a_sliding_window():
    at = fresh()
    at.radio(key="window").set_value("sliding").run()
    at.slider(key="train_len").set_value(40).run()
    assert not at.exception


def test_app_survives_non_overlapping_origins():
    at = fresh()
    at.slider(key="step").set_value(12).run()
    assert at.slider(key="step").value == 12
    assert not at.exception


def test_app_survives_a_single_origin():
    at = fresh()
    at.slider(key="k").set_value(1).run()
    assert not at.exception


def test_app_reports_the_impossible_protocol_instead_of_crashing():
    """A 160-point sliding window cannot be backtested on a 120-point series."""
    at = fresh()
    at.select_slider(key="n").set_value(120).run()
    at.radio(key="window").set_value("sliding").run()
    at.slider(key="train_len").set_value(160).run()
    assert not at.exception
    assert any("NO ORIGINS" in str(e.value) for e in at.error)


def test_app_survives_two_candidates_in_the_selection_panel():
    at = fresh()
    at.slider(key="M").set_value(2).run()
    assert not at.exception
