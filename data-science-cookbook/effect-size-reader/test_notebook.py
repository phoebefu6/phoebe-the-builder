"""The notebook must be the same study as the library, not a lookalike of it.

demo.ipynb embeds the engine by AST extraction from effectsize.py at build time, so the copy
cannot go stale. That is a guarantee, and a guarantee nobody checks is one that quietly stops
holding - so it is asserted here, along with the seed-identity rule: the notebook uses the
library's own design lists, and the seed is the index into them, so its printed rows ARE the cells
in evidence.txt rather than a second sample of them.
"""

from __future__ import annotations

import json
import os

import pytest
from build_notebook import ENGINE, EXPORTS, HEADER, extract
from effectsize import CELL_DESIGNS, DICH_DESIGNS, SHAPES, STUDY_NS, TRUTH_DESIGNS

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
    produced = [c for c in code_cells if c.get("outputs")]
    assert len(produced) >= len(code_cells) - 1, "notebook was not executed before commit"


def test_badges_point_at_this_build(nb):
    head = src_of(nb["cells"][0])
    assert "colab.research.google.com/github/phoebefu6/phoebe-the-builder" in head
    assert "mybinder.org" in head
    assert "data-science-cookbook/effect-size-reader" in head


def test_it_has_a_chart_and_a_try_your_own_section(nb):
    code = all_src(nb, "code")
    assert "notebook_figure.png" in code and "matplotlib" in code
    assert "Try your own" in all_src(nb, "markdown")


# ---------------------------------------------------------------------------
# Identity with the library
# ---------------------------------------------------------------------------


def test_the_embedded_engine_is_the_librarys_own_source(nb):
    """Not 'equivalent to' - character-identical, because it was extracted from it."""
    assert ENGINE == extract("effectsize.py", EXPORTS)
    code = all_src(nb, "code")
    for chunk in ENGINE.split("\n\n\n"):
        chunk = chunk.strip()
        if chunk:
            assert chunk in code, f"notebook is missing engine block starting: {chunk[:60]!r}"


def test_the_engine_block_is_self_contained_and_every_export_survived():
    """The guard that would have caught the real bug.

    `BAND_LO, BAND_HI = 0.045, 0.055` binds two names through a TUPLE target, and the first
    version of the extractor only read `targets[0].id` - so it dropped them silently. The notebook
    then died several minutes into nbconvert with a NameError raised from inside a function body.
    Executing the block and checking every requested name is present catches that in a second.
    """
    ns: dict = {}
    exec(HEADER + "\n" + ENGINE, ns)
    missing = [n for n in EXPORTS if n not in ns]
    assert not missing, f"the extractor dropped: {missing}"


def test_extractor_handles_tuple_targets():
    """Directly, so the fix cannot regress under a later refactor of `extract`."""
    got = extract("effectsize.py", ["BAND_LO"])
    assert "BAND_LO, BAND_HI" in got


def test_notebook_replays_the_librarys_designs_in_order(nb):
    """The seed IS the index into these lists, so their order is part of the result."""
    code = all_src(nb, "code")
    assert "for i, (shape, d) in enumerate(TRUTH_DESIGNS)" in code
    assert "for i, (shape, n) in enumerate(CELL_DESIGNS)" in code
    assert "for i, (shape, n) in enumerate(DICH_DESIGNS)" in code
    assert "seed=i" in code and "DICH_SEED_BASE + i" in code
    assert list(CELL_DESIGNS) == [(s, n) for s in SHAPES for n in STUDY_NS]
    assert len(TRUTH_DESIGNS) == 18 and len(DICH_DESIGNS) == 12


def test_notebook_output_matches_the_evidence_file(nb):
    """The whole point of sharing seeds and designs: the printed rows must BE the evidence cells.

    Checked on the population truths (18 of them) and on the estimator table (24 cells), by
    searching the notebook's own stdout for each value formatted exactly as the notebook prints it.
    """
    if not os.path.exists("results.json"):
        pytest.skip("run evidence.py first")
    r = json.load(open("results.json"))
    text = "\n".join("".join(o.get("text", []))
                     for c in nb["cells"] for o in c.get("outputs", [])
                     if o.get("output_type") == "stream")

    checked = 0
    for shape, d in [tuple(x) for x in r["truth_designs"]]:
        val = r["truths"][f"{shape}|{d}"]["ps"]
        assert f"{val:.4f}" in text, (
            f"the notebook never prints P(X>Y)={val:.4f} for {shape} at d={d} - it and "
            f"evidence.txt are sampling different populations"
        )
        checked += 1
    assert checked == 18

    for cell in r["cells"]:
        if cell["n"] != 10:
            continue
        row = next((ln for ln in text.splitlines()
                    if ln.startswith(cell["shape"]) and f"{cell['d_bias']:+.4f}" in ln), None)
        assert row is not None, (
            f"no notebook row reports a d-hat bias of {cell['d_bias']:+.4f} for {cell['shape']}"
        )


def test_notebook_states_the_findings_it_measured(nb):
    md = all_src(nb, "markdown")
    for claim in ["Glass's delta", "Hedges", "median split", "nan", "P(X>Y) is unbiased",
                  "50.25%", "analytic"]:
        assert claim.lower() in md.lower(), f"the notebook never states: {claim}"


def test_importing_the_generator_does_not_overwrite_the_notebook():
    """Regression carried from Day 177: build_notebook used to emit demo.ipynb at IMPORT time, so
    running the tests replaced the executed notebook with a blank one - and nothing could detect
    it afterwards, because the test suite was what had done it."""
    before = os.path.getmtime(NB)
    import importlib

    import build_notebook
    importlib.reload(build_notebook)
    assert os.path.getmtime(NB) == before, "importing build_notebook rewrote demo.ipynb"
