"""The Streamlit app, verified headless.

Widgets are addressed BY KEY. Setting a slider by list index, or `set_value` with a value outside
its range, passes vacuously - so every run first checks the value actually took before reading
anything downstream. And every branch of a verdict has to be shown REACHABLE: an app that can only
print one message is not a measurement.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest


def run(shape: str, d: float, n: int, reps: int = 2000) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    at.session_state["shape"] = shape
    at.session_state["d"] = d
    at.session_state["n"] = n
    at.session_state["reps"] = reps
    at.run()
    assert at.session_state["shape"] == shape, "the shape widget did not take the value"
    assert at.session_state["d"] == d, "the effect-size widget did not take the value"
    assert at.session_state["n"] == n, "the sample-size widget did not take the value"
    assert not at.exception, at.exception
    return at


def test_it_renders_at_all():
    at = run("normal", 0.5, 50)
    assert at.title[0].value.startswith("Cohen's d is not the effect")
    assert len(at.metric) >= 6


def test_a_null_effect_explains_itself_and_does_not_warn():
    at = run("normal", 0.0, 50)
    assert len(at.info) >= 1
    assert "null value" in at.info[0].value
    assert len(at.warning) == 0


def test_normal_data_agrees_with_the_textbook_gloss():
    at = run("normal", 0.5, 50)
    assert len(at.success) == 1, "normal data should match the normal gloss"
    assert len(at.warning) == 0


def test_skewed_data_contradicts_the_textbook_gloss():
    at = run("lognormal", 0.5, 50)
    assert len(at.warning) == 1
    assert "Same d, different answer" in at.warning[0].value
    assert len(at.success) == 0


@pytest.mark.parametrize("shape", ["normal", "lognormal", "contaminated"])
def test_the_metric_table_is_populated_for_every_shape(shape):
    at = run(shape, 0.5, 50)
    assert len(at.dataframe) == 1
    df = at.dataframe[0].value
    assert len(df) == 7, "all seven metrics should appear"
    assert df["mean estimate"].notna().all()


def test_the_smallest_detectable_effect_falls_as_n_rises():
    small_n = run("normal", 0.5, 20)
    big_n = run("normal", 0.5, 500)

    def smallest(at):
        return float(next(m for m in at.metric if m.label == "Smallest d this n can see").value)

    assert smallest(big_n) < smallest(small_n)
