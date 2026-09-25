"""Headless Streamlit checks, widgets addressed BY KEY, every banner branch shown reachable."""

from __future__ import annotations

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


def test_opens_on_the_exemplar_f_significant_no_tukey_pair() -> None:
    at = _run()
    assert len(at.error) == 1 and "names NO pair" in at.error[0].value


def test_agreeing_branch_is_reachable() -> None:
    at = _run()
    far = "\n".join([", ".join(["0", "0.1", "-0.1", "0.2", "-0.2"])] * 2 + ["5, 5.1, 4.9, 5.2, 4.8"])
    at.text_area(key="groups").set_value(far).run()
    assert not at.error
    assert "agree" in at.success[0].value


def test_bad_input_warns_instead_of_crashing() -> None:
    at = _run()
    at.text_area(key="groups").set_value("1, 2, 3\n4").run()
    assert not at.exception
    assert any("at least" in w.value for w in at.warning)


def test_disagreement_branch_is_reachable() -> None:
    """Found by search, not hand-built: a k=3 dataset where the family-wise procedures split on
    whether ANY pair differs, and which does not trip the higher-priority F-without-Tukey banner."""
    import posthoc as P

    for seed in range(5000):
        data = P.simulate(3, "spread", 0.9, seed, reps=1)
        groups = [[round(v, 2) for v in data[0, g]] for g in range(3)]
        res = P.analyse(groups)
        named = {p: bool(res["significant"][p]) for p in ("bonferroni", "holm", "tukey")}
        error_branch = res["F_p"] < P.ALPHA and not named["tukey"]
        if not error_branch and len(set(named.values())) > 1:
            break
    else:
        pytest.fail("no disagreement dataset found")
    at = _run()
    at.text_area(key="groups").set_value("\n".join(", ".join(map(str, g)) for g in groups)).run()
    assert any("disagree" in w.value for w in at.warning)
