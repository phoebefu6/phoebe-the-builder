"""The Streamlit app, verified headless.

Two rules carried from earlier builds, both of which produced green runs that tested nothing:

- Address widgets BY KEY. Setting a slider by list index, or `set_value` with a value outside the
  slider's range, passes vacuously - so every assertion here first checks the value actually took.
- Every branch of a verdict must be shown to be REACHABLE. An app that can only ever print one
  message is not a measurement, and a test that only ever sees that message is not a test.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest


def run(d: float, n: int, reps: int) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    at.session_state["d"] = d
    at.session_state["n"] = n
    at.session_state["reps"] = reps
    at.run()
    # The guard: prove the widget actually holds the value before reading anything downstream.
    assert at.session_state["d"] == d, "the effect-size widget did not take the value"
    assert at.session_state["n"] == n, "the sample-size widget did not take the value"
    assert at.session_state["reps"] == reps
    assert not at.exception, at.exception
    return at


def test_it_renders_at_all():
    at = run(0.35, 30, 10000)
    assert len(at.metric) >= 4
    assert at.title[0].value.startswith("A p-value is a random variable")


def test_null_design_explains_itself_and_does_not_warn():
    at = run(0.0, 50, 10000)
    assert len(at.info) >= 1
    assert "false positive" in at.info[0].value
    assert len(at.warning) == 0, "a true null must not trip the straddle warning"
    assert len(at.error) == 0


def test_underpowered_design_reports_the_straddle():
    at = run(0.35, 30, 20000)
    assert len(at.warning) == 1
    assert "straddles" in at.warning[0].value
    assert len(at.success) == 0


def test_high_power_design_reports_the_dance_has_stopped():
    at = run(0.8, 200, 10000)
    assert len(at.success) == 1
    assert "stopped" in at.success[0].value
    assert len(at.warning) == 0


def test_calibration_error_never_fires_on_a_good_design():
    """The app's own alarm must be quiet when the study is sound - otherwise it is noise."""
    for d, n in [(0.0, 50), (0.5, 50), (0.8, 100)]:
        at = run(d, n, 20000)
        assert len(at.error) == 0, f"calibration alarm fired on a valid design d={d}, n={n}"


@pytest.mark.parametrize("d,n", [(0.2, 20), (0.5, 50)])
def test_winners_curse_panel_reports_an_exaggeration_above_one(d, n):
    at = run(d, n, 20000)
    labels = {m.label: m.value for m in at.metric}
    assert "Exaggeration (type M)" in labels
    assert float(labels["Exaggeration (type M)"].rstrip("x")) >= 1.0
