"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

import os

import numpy as np
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


def _set(at: AppTest, b, a) -> None:
    at.text_area(key="before").set_value(", ".join(f"{v:.3f}" for v in b))
    at.text_area(key="after").set_value(", ".join(f"{v:.3f}" for v in a)).run()
    assert not at.exception


def test_opens_on_the_exemplar_where_pairing_is_the_finding() -> None:
    at = _run()
    assert len(at.success) == 1 and "PAIRING IS THE FINDING" in at.success[0].value


def test_negative_correlation_branch() -> None:
    at = _run()
    rng = np.random.default_rng(1)
    b = rng.normal(10, 1, 30)
    _set(at, b, 20 - b + rng.normal(0, 0.3, 30))
    assert at.error and "NEGATIVE CORRELATION" in at.error[0].value


def test_weak_pairing_branch() -> None:
    at = _run()
    for seed in range(200):  # first seed with a weak positive link, below break-even at n = 400
        rng = np.random.default_rng(seed)
        b, a = rng.normal(10, 1, 400), rng.normal(10.2, 1, 400)
        if 0 <= np.corrcoef(b, a)[0, 1] < 0.012:
            break
    _set(at, b, a)
    assert at.info and "WEAK PAIRING" in at.info[0].value


def test_agree_branch() -> None:
    at = _run()
    b = np.linspace(10, 20, 40)
    _set(at, b, b + 5 + np.sin(np.arange(40)))
    assert any("Both analyses agree" in w.value for w in at.warning)


def test_bad_input_warns_instead_of_crashing() -> None:
    at = _run()
    at.text_area(key="before").set_value("1, 2, 3, 4").run()
    assert not at.exception
    assert any("pairs must match" in w.value for w in at.warning)
