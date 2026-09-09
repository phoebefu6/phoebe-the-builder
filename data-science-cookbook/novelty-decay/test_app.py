"""The Streamlit app has to actually run. A scheduled build cannot open a browser, so this
drives it headless through Streamlit's own AppTest harness.

Widgets are addressed by KEY, never by index: AppTest returns main-body widgets before
sidebar ones regardless of source order, and an out-of-range set_value is silently ignored
rather than raising - between them, an index-based test can pass while exercising nothing.
"""

from __future__ import annotations

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def fresh():
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    assert not at.exception
    return at


def test_app_runs_without_exception():
    fresh()


def test_the_key_addressing_this_file_relies_on_actually_works():
    """Guard the guard: if set_value stopped taking, every test below would pass vacuously."""
    at = fresh()
    at.slider(key="weeks").set_value(9).run()
    assert at.slider(key="weeks").value == 9


def test_app_runs_in_every_world():
    at = fresh()
    for w in at.radio(key="world").options:
        at.radio(key="world").set_value(w).run()
        assert not at.exception, f"world {w} raised"


def test_app_runs_under_both_enrollment_processes():
    at = fresh()
    for e in at.radio(key="enroll").options:
        at.radio(key="enroll").set_value(e).run()
        assert not at.exception, f"enrollment {e} raised"


def test_app_reports_the_degenerate_calendar_instead_of_crashing():
    at = fresh()
    at.slider(key="gap").set_value(0).run()
    assert at.slider(key="gap").value == 0
    assert not at.exception
    assert any("NOT IDENTIFIED" in str(e.value) for e in at.error)


def test_app_survives_a_zero_long_run_effect():
    """tau_inf = 0 divides by zero in several ratios if they are not guarded."""
    at = fresh()
    at.slider(key="tau_inf").set_value(0.0).run()
    assert at.slider(key="tau_inf").value == 0.0
    assert not at.exception


def test_app_survives_the_shortest_window():
    at = fresh()
    at.slider(key="weeks").set_value(1).run()
    assert not at.exception
