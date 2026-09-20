"""The notebook must be the same study as the library, not a lookalike of it.

demo.ipynb embeds the engine by AST extraction from pvalue.py at build time (see
build_notebook.py), so the copy cannot go stale. That is a guarantee, and a guarantee nobody
checks is one that quietly stops holding - so it is asserted here, alongside the seed-identity
rule carried from Day 175: the notebook must use the LIBRARY'S seeds, because a borderline cell
lands either side of a threshold on a different seed and reads as two artifacts contradicting
each other.
"""

from __future__ import annotations

import json
import os
import re

import pytest
from build_notebook import ENGINE, EXPORTS, extract
from pvalue import DESIGNS, REPS

NB = "demo.ipynb"
pytestmark = pytest.mark.skipif(not os.path.exists(NB), reason="run build_notebook.py first")


@pytest.fixture(scope="module")
def nb():
    return json.load(open(NB))


def src_of(cell) -> str:
    return "".join(cell["source"])


def all_src(nb, kind: str) -> str:
    return "\n".join(src_of(c) for c in nb["cells"] if c["cell_type"] == kind)


# ---------------------------------------------------------------------------
# Structure - the things that fail silently until nbconvert runs
# ---------------------------------------------------------------------------


def test_every_cell_has_an_id_and_keeps_its_line_breaks(nb):
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids)) == len(nb["cells"])
    for c in nb["cells"]:
        body = c["source"]
        assert isinstance(body, list) and body
        # Every line but the last must still end in a newline, or nbformat concatenates the
        # whole cell into one unparseable line.
        for line in body[:-1]:
            assert line.endswith("\n"), f"cell {c['id']} lost a line break"


def test_first_code_cell_opens_with_the_future_import(nb):
    first = next(c for c in nb["cells"] if c["cell_type"] == "code")
    assert src_of(first).lstrip().startswith("from __future__ import annotations")


def test_notebook_is_prerendered_and_nothing_errored(nb):
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    errors = [c["id"] for c in code_cells
              if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    assert not errors, f"cells raised: {errors}"
    # The "Try your own" cell is deliberately all comments and produces nothing.
    produced = [c for c in code_cells if c.get("outputs")]
    assert len(produced) >= len(code_cells) - 1, "notebook was not executed before commit"


def test_badges_point_at_this_build(nb):
    head = src_of(nb["cells"][0])
    assert "colab.research.google.com/github/phoebefu6/phoebe-the-builder" in head
    assert "mybinder.org" in head
    assert "data-science-cookbook/p-value-dance" in head


def test_it_has_a_chart_and_a_try_your_own_section(nb):
    code = all_src(nb, "code")
    assert "notebook_figure.png" in code and "matplotlib" in code
    assert "Try your own" in all_src(nb, "markdown")


# ---------------------------------------------------------------------------
# Identity with the library
# ---------------------------------------------------------------------------


def test_the_embedded_engine_is_the_librarys_own_source(nb):
    """Not 'equivalent to' - character-identical, because it was extracted from it."""
    assert ENGINE == extract("pvalue.py", EXPORTS)
    code = all_src(nb, "code")
    for chunk in ENGINE.split("\n\n\n"):
        chunk = chunk.strip()
        if chunk:
            assert chunk in code, f"notebook is missing engine block starting: {chunk[:60]!r}"


def test_importing_the_generator_does_not_overwrite_the_notebook():
    """Regression: build_notebook.py used to emit demo.ipynb at import time, so importing it here
    replaced the executed notebook with a blank one - and no later test could see it, because the
    test suite was the thing that had done it."""
    before = os.path.getmtime(NB)
    import importlib

    import build_notebook
    importlib.reload(build_notebook)
    assert os.path.getmtime(NB) == before, "importing build_notebook rewrote demo.ipynb"


def test_the_extraction_actually_pulled_everything_it_names():
    for name in EXPORTS:
        assert re.search(rf"^(def |class |{name} *=)|^def {name}\b", ENGINE, re.M) or \
               re.search(rf"^{re.escape(name)}\b", ENGINE, re.M), f"{name} was not extracted"


def test_notebook_replays_the_librarys_designs_in_order(nb):
    """The seed IS the index into DESIGNS, so the order of that list is load-bearing."""
    code = all_src(nb, "code")
    assert "for i, (d, n) in enumerate(DESIGNS)" in code
    assert "simulate(d, n, REPS, seed=i)" in code
    assert "default_rng(900 + i)" in code and "default_rng(700 + i)" in code
    assert list(DESIGNS) == [(d, n) for d in (0.0, 0.2, 0.5, 0.8)
                             for n in (10, 20, 50, 100, 500)]


def test_notebook_output_matches_the_evidence_file(nb):
    """The whole point of sharing seeds: the printed rows must BE the evidence cells."""
    if not os.path.exists("results.json"):
        pytest.skip("run evidence.py first")
    results = json.load(open("results.json"))
    assert results["reps"] == REPS
    text = "\n".join("".join(o.get("text", []))
                     for c in nb["cells"] for o in c.get("outputs", [])
                     if o.get("output_type") == "stream")
    checked = 0
    for cell in results["cells"]:
        label = f"d={cell['d_true']:g}, n={cell['n']}"
        row = next((ln for ln in text.splitlines()
                    if ln.startswith(label) and f"{cell['power']:.4f}" in ln), None)
        assert row is not None, (
            f"no notebook row reports power {cell['power']:.4f} for {label} - the notebook and "
            f"evidence.txt are sampling different studies"
        )
        checked += 1
    assert checked == len(results["cells"])


def test_notebook_states_the_findings_it_measured(nb):
    md = all_src(nb, "markdown")
    for claim in ["winner's curse", "power < 0.95", "unconditional power",
                  "Uniform(0, 1)", "type M", "Type S"]:
        assert claim.lower() in md.lower(), f"the notebook never states: {claim}"
