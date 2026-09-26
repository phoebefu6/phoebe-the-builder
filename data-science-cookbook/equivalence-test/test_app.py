"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")
TIMEOUT = 120


def _run() -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    return at


def _vals(xs) -> str:
    return ", ".join(f"{v:.3f}" for v in xs)


def test_opens_on_the_exemplar_inconclusive() -> None:
    at = _run()
    assert len(at.error) == 1 and "INCONCLUSIVE" in at.error[0].value


def test_equivalent_branch() -> None:
    import numpy as np

    at = _run()
    base = np.linspace(80, 120, 300)
    at.text_area(key="x").set_value(_vals(base)).run()
    at.text_area(key="y").set_value(_vals(base + 0.2)).run()
    assert at.success and "EQUIVALENT within" in at.success[0].value


def test_both_branch() -> None:
    import numpy as np

    at = _run()
    base = np.linspace(80, 120, 2000)
    at.text_area(key="x").set_value(_vals(base)).run()
    at.text_area(key="y").set_value(_vals(base + 3.0)).run()
    assert at.info and "AND DIFFERENT" in at.info[0].value


def test_different_branch() -> None:
    import numpy as np

    at = _run()
    base = np.linspace(80, 120, 200)
    at.text_area(key="x").set_value(_vals(base)).run()
    at.text_area(key="y").set_value(_vals(base + 25)).run()
    assert any("DIFFERENT, and not" in w.value for w in at.warning)


def test_bad_input_warns_instead_of_crashing() -> None:
    at = _run()
    at.text_area(key="x").set_value("5").run()
    assert not at.exception
    assert any("at least 2" in w.value for w in at.warning)


def test_margin_widget_is_live() -> None:
    at = _run()
    at.number_input(key="margin").set_value(40.0).run()
    assert not at.error  # a +-$40 margin makes the exemplar equivalent
    assert at.success
