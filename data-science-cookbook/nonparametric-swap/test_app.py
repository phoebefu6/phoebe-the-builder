"""Headless Streamlit checks.

Widgets are addressed BY KEY. An index-addressed widget silently moves when the layout changes,
and `AppTest` accepts an out-of-range index by doing nothing - so an index-addressed test can
pass while testing nothing at all.
"""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"),
                                reason="evidence.py has not been run")

TIMEOUT = 120


def _run() -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    return at


def test_app_starts() -> None:
    at = _run()
    assert any("Mann-Whitney" in m.value for m in at.markdown)


def test_default_design_is_flagged_as_a_disagreement() -> None:
    """The app opens on the study's own design, which disagrees in sign - so the error banner
    must be showing before the user touches anything."""
    at = _run()
    assert len(at.error) == 1
    assert "disagree in SIGN" in at.error[0].value


def test_the_agreeing_branch_is_reachable() -> None:
    """Both branches have to be reachable, or the banner is a constant wearing a test. Pushing
    the minority component down to +0.5 makes the mean negative AND P(B>A) below a half, so the
    two targets agree again."""
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    at.slider(key="mu_high").set_value(0.5).run()
    assert not at.error
    assert len(at.success) == 1
    assert "agree in sign" in at.success[0].value


def test_running_the_live_design_produces_a_rate() -> None:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    at.select_slider(key="reps").set_value(500).run()
    at.button(key="run_disagreement").click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert any("opposite directions" in lb for lb in labels)


def test_power_tab_switches_population() -> None:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    at.selectbox(key="power_shape").set_value("lognormal").run()
    assert not at.exception
    assert len(at.dataframe) >= 3


def test_broken_only_filter_shortens_the_table() -> None:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    full = len(at.dataframe[3].value)
    at.checkbox(key="only_broken").set_value(True).run()
    filtered = len(at.dataframe[3].value)
    assert 0 < filtered < full


def test_app_reads_stored_verdicts_rather_than_recomputing() -> None:
    """Day 175: the chart and the notebook each grew their own copy of the comparison and
    disagreed with the evidence file. The app must quote results.json."""
    src = open("app.py").read()
    assert 'json.load(open("results.json"))' in src
    assert "N.verdict(" not in src
    r = json.load(open("results.json"))
    at = _run()
    mw_bad = sum(1 for row in r["unequal"] if row["mw_verdict"] != "ok")
    assert any(f"{mw_bad} of {len(r['unequal'])}" == m.value for m in at.metric)
