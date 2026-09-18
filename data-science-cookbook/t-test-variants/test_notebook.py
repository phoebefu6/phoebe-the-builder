"""The notebook re-derives the engine inline so it runs standalone in Colab.

Two copies of one formula is exactly the thing that drifts, so this suite executes the
notebook's own code cells and checks its functions against ttests.py on fixed inputs. Day 171
shipped a notebook whose inline copy had silently dropped a parameter; the guard is standard.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import pytest
import ttests as tt


@pytest.fixture(scope="module")
def nb_ns() -> Dict[str, Any]:
    """Execute the notebook's engine cells (1 and 2) in a fresh namespace."""
    nb = json.load(open("demo.ipynb"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ns: Dict[str, Any] = {}
    for cell in code_cells[:2]:  # the two cells that define the engine
        exec("".join(cell["source"]), ns)
    return ns


def test_notebook_defines_the_engine(nb_ns: Dict[str, Any]) -> None:
    for name in ("student_p_vec", "welch_p_vec", "zscore_p_vec", "draw", "simulate", "ALPHA"):
        assert name in nb_ns, f"notebook lost {name}"


@pytest.mark.parametrize("fn", ["student_p_vec", "welch_p_vec", "zscore_p_vec"])
def test_notebook_statistics_match_the_library(nb_ns: Dict[str, Any], fn: str) -> None:
    rng = np.random.default_rng(808)
    X, Y = rng.standard_normal((60, 14)), rng.standard_normal((60, 6)) * 2.5
    np.testing.assert_allclose(nb_ns[fn](X, Y), getattr(tt, fn)(X, Y), rtol=0, atol=1e-14)


@pytest.mark.parametrize("dist", ["normal", "lognormal", "t3", "uniform"])
def test_notebook_draws_match_the_library(nb_ns: Dict[str, Any], dist: str) -> None:
    """Same seed, same stream: the two draw() implementations must be bit-identical."""
    a = nb_ns["draw"](np.random.default_rng(77), 200, 30, 1.7, dist, 0.4)
    b = tt.draw(np.random.default_rng(77), 200, 30, 1.7, dist, 0.4)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_notebook_simulate_matches_the_library(nb_ns: Dict[str, Any]) -> None:
    a = nb_ns["simulate"](50, 10, sd2=3.0, reps=5_000, seed=31)
    b = tt.simulate(tt.Cell(n1=50, n2=10, sd2=3.0), reps=5_000, seed=31)
    assert a == b


def test_notebook_alpha_matches(nb_ns: Dict[str, Any]) -> None:
    assert nb_ns["ALPHA"] == tt.ALPHA


def test_notebook_is_prerendered() -> None:
    """GitHub shows outputs only if they were committed. An unexecuted notebook is a blank page."""
    nb = json.load(open("demo.ipynb"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "no code cells"
    unrun = [i for i, c in enumerate(code_cells) if not c.get("outputs")]
    assert not unrun, f"code cells with no committed output: {unrun}"
    assert any(o.get("data", {}).get("image/png") for c in code_cells for o in c["outputs"]), \
        "no rendered figure in the notebook"
