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
    at = AppTest.from_file("app.py", default_timeout=300)
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


def test_app_runs_for_every_interval_method():
    at = fresh()
    for m in at.selectbox(key="method").options:
        at.selectbox(key="method").set_value(m).run()
        assert not at.exception, f"method {m} raised"


def test_app_runs_at_every_nominal_level():
    at = fresh()
    for c in at.select_slider(key="conf").options:
        at.select_slider(key="conf").set_value(c).run()
        assert not at.exception, f"level {c} raised"


def test_app_survives_a_one_step_horizon():
    """h=1 makes path coverage equal pointwise and k_eff degenerate."""
    at = fresh()
    at.slider(key="h").set_value(1).run()
    assert not at.exception


def test_app_warns_when_a_calibrated_method_has_too_little_history():
    at = fresh()
    at.selectbox(key="method").set_value("split conformal").run()
    at.select_slider(key="n_train").set_value(36).run()
    assert not at.exception
    assert any("too few origins" in str(w.value) for w in at.warning)


def test_app_reports_conditional_coverage_only_where_it_exists():
    at = fresh()
    at.radio(key="world").set_value("constant").run()
    assert any("one noise level" in str(m.value) for m in at.markdown)
    at.radio(key="world").set_value("two volatility regimes").run()
    assert any("volatile half" in str(m.value) for m in at.markdown)
