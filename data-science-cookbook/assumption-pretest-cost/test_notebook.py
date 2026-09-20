"""The notebook re-derives the engine inline so it runs standalone in Colab.

Two copies of one formula is exactly the thing that drifts, so this suite executes the notebook's
own engine cells and checks its functions against pretest.py on fixed inputs. It also checks the
notebook uses the LIBRARY'S SEEDS, so its rows are the study's cells rather than a second sample
of them - Day 175 shipped a notebook that disagreed with its own evidence file on a borderline
cell for exactly that reason.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import pretest as P
import pytest


@pytest.fixture(scope="module")
def nb_ns() -> Dict[str, Any]:
    """Execute the notebook's engine cells (the first two) in a fresh namespace."""
    nb = json.load(open("demo.ipynb"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ns: Dict[str, Any] = {}
    for cell in code_cells[:2]:
        exec("".join(cell["source"]), ns)
    return ns


def _src() -> str:
    nb = json.load(open("demo.ipynb"))
    return "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_defines_the_engine(nb_ns: Dict[str, Any]) -> None:
    for name in ("draw", "student_p", "welch_p", "pretest_p", "measure", "wilson", "is_broken",
                 "ALPHA", "BAND_LO", "BAND_HI", "DESIGNS", "grid_seed"):
        assert name in nb_ns, f"notebook lost {name}"


def test_notebook_constants_match(nb_ns: Dict[str, Any]) -> None:
    assert nb_ns["ALPHA"] == P.ALPHA
    assert (nb_ns["BAND_LO"], nb_ns["BAND_HI"]) == (P.BAND_LO, P.BAND_HI)


def test_notebook_wilson_and_verdict_match(nb_ns: Dict[str, Any]) -> None:
    """One broken-verdict rule per build, not one per artifact."""
    for rate, reps in ((0.05, 4_000), (0.0433, 1_500), (0.10, 20_000), (0.0, 1_000)):
        assert nb_ns["wilson"](rate, reps) == pytest.approx(P.wilson(rate, reps), abs=1e-15)
        assert nb_ns["is_broken"](rate, reps) == P.is_broken(rate, reps)


@pytest.mark.parametrize("dist", P.DISTRIBUTIONS)
def test_notebook_draws_match_the_library(nb_ns: Dict[str, Any], dist: str) -> None:
    """Same seed, same stream: bit-identical, not merely close.

    A reordered rng call would change every number in the notebook while leaving both versions
    individually defensible.
    """
    a1, a2 = nb_ns["draw"](dist, 12, 9, 1.0, 2.0, 50, 0.0, np.random.default_rng(77))
    b1, b2 = P.draw(dist, 12, 9, 1.0, 2.0, 50, 0.0, np.random.default_rng(77))
    np.testing.assert_allclose(a1, b1, rtol=0, atol=0)
    np.testing.assert_allclose(a2, b2, rtol=0, atol=0)


def test_notebook_t_tests_match_the_library(nb_ns: Dict[str, Any]) -> None:
    x, y = P.draw("normal", 13, 8, 1.0, 2.0, 120, 0.0, np.random.default_rng(808))
    np.testing.assert_allclose(nb_ns["student_p"](x, y), P.student_p(x, y), rtol=0, atol=1e-14)
    np.testing.assert_allclose(nb_ns["welch_p"](x, y), P.welch_p(x, y), rtol=0, atol=1e-14)


@pytest.mark.parametrize("name", P.PRETESTS)
def test_notebook_pretests_match_the_library(nb_ns: Dict[str, Any], name: str) -> None:
    x, y = P.draw("lognormal", 14, 11, 1.0, 1.5, 80, 0.0, np.random.default_rng(909))
    np.testing.assert_allclose(nb_ns["pretest_p"](name, x, y), P.pretest_p(name, x, y),
                               rtol=0, atol=1e-14)


def test_notebook_measure_reproduces_a_library_cell_exactly(nb_ns: Dict[str, Any]) -> None:
    """The notebook's headline rows must BE the study's cells, not a re-run of the same design."""
    d = P.Design(50, 10, 2.0)
    seed = 17 * list(P.DESIGNS).index(d)
    got = nb_ns["measure"](50, 10, 2.0, reps=1_500, seed=seed)
    want = P.run_cell(d, "brown-forsythe", 1_500, seed=seed)
    assert got["student"] == want.student_error
    assert got["welch"] == want.welch_error
    assert got["gated"] == want.gated_error
    assert got["pretest_fires"] == want.pretest_reject_rate
    assert got["student_given_pass"] == want.student_error_given_pass


def test_notebook_design_list_matches_the_library_in_ORDER() -> None:
    """The seed formula is an index into this list, so order is load-bearing, not cosmetic.

    Reordering DESIGNS in pretest.py without reordering the notebook's copy would silently
    re-seed every row - the numbers would all still be valid, and all different.
    """
    ns: Dict[str, Any] = {}
    nb = json.load(open("demo.ipynb"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for cell in code_cells[:2]:
        exec("".join(cell["source"]), ns)
    lib = [(d.n1, d.n2, d.sd_ratio) for d in P.DESIGNS]
    assert ns["DESIGNS"] == lib, "notebook design list drifted from pretest.DESIGNS"
    for i, d in enumerate(P.DESIGNS):
        assert ns["grid_seed"](d.n1, d.n2, d.sd_ratio) == 17 * i, f"seed formula drifted at {d.label}"


def test_notebook_ladder_uses_the_evidence_seed() -> None:
    """evidence.py walks the variance ratio at seed 31; the notebook must use the same one."""
    src = _src()
    assert "measure(50, 10, r, reps=20_000, seed=31)" in src, \
        "notebook ladder re-seeded away from evidence.py's"


def test_notebook_is_prerendered() -> None:
    """GitHub shows outputs only if they were committed. An unexecuted notebook is a blank page."""
    nb = json.load(open("demo.ipynb"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "no code cells"
    unrun = [i for i, c in enumerate(code_cells) if not c.get("outputs")]
    assert not unrun, f"code cells with no committed output: {unrun}"
    assert any(o.get("data", {}).get("image/png") for c in code_cells for o in c["outputs"]), \
        "no rendered figure in the notebook"
