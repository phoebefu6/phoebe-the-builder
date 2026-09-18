"""Smoke test for the Streamlit front end.

It is imported, not run - importing it executes the whole script top to bottom under
AppTest, which is the only way a Streamlit page gets exercised without a browser.
"""

from __future__ import annotations

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest


def _run(timeout: float = 120.0):
    at = AppTest.from_file("app.py", default_timeout=timeout)
    at.run()
    return at


def test_app_loads_without_exception() -> None:
    at = _run()
    assert not at.exception


def test_default_design_is_the_headline_failure() -> None:
    """The page opens on n=50/10 with a 3x gap, and must be shouting about it."""
    at = _run()
    assert at.error, "the default design inflates Student's; the page should say so"
    assert "not a 5% test" in at.error[0].value


def test_three_metrics_are_shown_one_per_test() -> None:
    at = _run()
    labels = [m.label for m in at.metric]
    assert labels == ["student", "welch", "z-shortcut"], labels


def test_balancing_the_design_clears_the_alarm() -> None:
    """Addressed by KEY. An index drifts the moment a widget is added, and drifts silently."""
    at = AppTest.from_file("app.py", default_timeout=120.0)
    at.run()
    at.number_input(key="n2").set_value(50)  # -> balanced 50/50
    at.run()
    assert not at.exception
    assert not at.error, "a balanced design with a 3x gap is fine for Student's"
    assert at.success


def test_setting_a_true_difference_switches_to_power_language() -> None:
    at = AppTest.from_file("app.py", default_timeout=120.0)
    at.run()
    at.slider(key="shift").set_value(0.8)
    at.run()
    assert not at.exception
    assert at.info, "with a real effect the page must stop calling the rate a Type I error"
    assert "POWER" in at.info[0].value
