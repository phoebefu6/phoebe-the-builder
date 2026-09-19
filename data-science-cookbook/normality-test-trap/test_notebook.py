"""The notebook re-derives the engine inline so it runs standalone in Colab.

Two copies of one formula is exactly the thing that drifts, so this suite executes the notebook's
own engine cells and checks its functions against normality.py on fixed inputs. Day 171 shipped a
notebook whose inline copy had silently dropped a parameter; the guard is standard since Day 172.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import normality as N
import numpy as np
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


def test_notebook_defines_the_engine(nb_ns: Dict[str, Any]) -> None:
    for name in ("draw", "one_sample_t_p", "shapiro_reject", "wilcoxon_p", "measure",
                 "wilson", "is_broken", "ALPHA", "SHAPE"):
        assert name in nb_ns, f"notebook lost {name}"


def test_notebook_alpha_and_shape_match(nb_ns: Dict[str, Any]) -> None:
    assert nb_ns["ALPHA"] == N.ALPHA
    assert nb_ns["SHAPE"] == N.SHAPE
    assert (nb_ns["BAND_LO"], nb_ns["BAND_HI"]) == (N.BAND_LO, N.BAND_HI)


def test_notebook_wilson_matches_the_library(nb_ns: Dict[str, Any]) -> None:
    """One broken-verdict rule, not two. The notebook used to compare the point estimate."""
    for rate, reps in ((0.05, 4000), (0.0433, 1500), (0.10, 20000), (0.0, 1000)):
        assert nb_ns["wilson"](rate, reps) == pytest.approx(N.wilson(rate, reps), abs=1e-15)
        cell = N.CellResult("x", 1, reps, 0.05, 0.5, rate, rate / 2, rate / 2, 0.05, rate, rate, rate)
        assert nb_ns["is_broken"](rate, reps) == cell.t_is_broken


@pytest.mark.parametrize("dist", N.DISTRIBUTIONS)
def test_notebook_draws_match_the_library(nb_ns: Dict[str, Any], dist: str) -> None:
    """Same seed, same stream: the two draw() implementations must be bit-identical.

    Bit-identical, not merely close - a reordered rng call would change every number in the
    notebook while leaving both versions individually defensible.
    """
    a = nb_ns["draw"](dist, 25, 120, np.random.default_rng(77))
    b = N.draw(dist, 25, 120, np.random.default_rng(77))
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_notebook_t_matches_the_library(nb_ns: Dict[str, Any]) -> None:
    rng = np.random.default_rng(808)
    x = rng.standard_normal((80, 17))
    ta, pa = nb_ns["one_sample_t_p"](x)
    tb, pb = N.one_sample_t_p(x)
    np.testing.assert_allclose(ta, tb, rtol=0, atol=1e-14)
    np.testing.assert_allclose(pa, pb, rtol=0, atol=1e-14)


def test_notebook_shapiro_and_wilcoxon_match(nb_ns: Dict[str, Any]) -> None:
    rng = np.random.default_rng(909)
    x = N.draw("lognormal-mild", 30, 40, rng)
    np.testing.assert_array_equal(nb_ns["shapiro_reject"](x), N.shapiro_reject(x))
    np.testing.assert_allclose(nb_ns["wilcoxon_p"](x), N.wilcoxon_p(x), rtol=0, atol=1e-14)


def test_notebook_measure_matches_run_cell(nb_ns: Dict[str, Any]) -> None:
    """The notebook's headline numbers must be the library's numbers, not a near-miss."""
    got = nb_ns["measure"]("lognormal-mild", 30, reps=800, seed=31)
    want = N.run_cell("lognormal-mild", 30, 800, seed=31)
    assert got["shapiro_rejects"] == pytest.approx(want.shapiro_reject_rate, abs=0)
    assert got["t_error"] == pytest.approx(want.t_error, abs=0)
    assert got["wilcoxon_error"] == pytest.approx(want.wilcoxon_error, abs=0)
    assert got["gated_error"] == pytest.approx(want.conditional_error, abs=0)
    assert got["t_low"] == pytest.approx(want.t_error_lower_tail, abs=0)
    assert got["t_high"] == pytest.approx(want.t_error_upper_tail, abs=0)


def test_notebook_is_prerendered() -> None:
    """GitHub shows outputs only if they were committed. An unexecuted notebook is a blank page."""
    nb = json.load(open("demo.ipynb"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "no code cells"
    unrun = [i for i, c in enumerate(code_cells) if not c.get("outputs")]
    assert not unrun, f"code cells with no committed output: {unrun}"
    assert any(o.get("data", {}).get("image/png") for c in code_cells for o in c["outputs"]), \
        "no rendered figure in the notebook"


def test_notebook_headline_table_uses_the_study_replicate_count() -> None:
    """The notebook's verdicts must be decided at the same resolution as evidence.txt.

    At 4,000 replicates the notebook labelled lognormal-mild cells "gate fires on a test that
    works" that the 20,000-replicate study flags as broken. Neither artifact was wrong about its
    own numbers; they were answering at different resolutions, which is indistinguishable from
    one of them being wrong.
    """
    nb = json.load(open("demo.ipynb"))
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "NB_REPS = 20_000" in src, "notebook headline table dropped below the study's reps"
    for n in (10, 20, 30, 50, 100, 200, 500):
        assert N.reps_for(n) == 20_000, f"library grid moved off 20,000 at n={n}; notebook is now stale"


def test_notebook_reproduces_the_library_grid_cell_exactly(nb_ns: Dict[str, Any]) -> None:
    """The notebook's headline rows must BE the study's cells, not a re-run of the same design.

    A borderline cell (lognormal-mild at n=50 sits within one interval-width of the threshold)
    lands either side of "broken" on different seeds, which reads to a reader as the notebook and
    evidence.txt contradicting each other.
    """
    seed = 1000 * list(N.DISTRIBUTIONS).index("lognormal-mild") + list(N.GRID_N).index(30)
    got = nb_ns["measure"]("lognormal-mild", 30, reps=2_000, seed=seed)
    want = N.run_cell("lognormal-mild", 30, 2_000, seed=seed)
    assert got["t_error"] == want.t_error
    assert got["shapiro_rejects"] == want.shapiro_reject_rate


def test_notebook_seed_formula_matches_run_grid() -> None:
    """If run_grid's seeding changes, the notebook's copy of the formula must fail loudly."""
    nb = json.load(open("demo.ipynb"))
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "1000 * GRID_ORDER.index(dist) + GRID_N_ORDER.index(n)" in src
    assert list(N.DISTRIBUTIONS) == ["normal", "uniform", "t5", "contaminated", "lognormal-mild",
                                     "exponential", "lognormal-strong"], "grid order moved"
    assert list(N.GRID_N) == [10, 20, 30, 50, 100, 200, 500, 1000, 5000], "grid n order moved"
