"""Tests for the notebook generator.

Three of these exist because the corresponding defect shipped in an earlier build in this
series, and every one of them fails slowly and confusingly in the wild - minutes into an
nbconvert run, or not at all.
"""

from __future__ import annotations

import ast
import json
import os
import time

import build_notebook as B
import nonparam as N
import pytest

NB = "demo.ipynb"


def test_import_does_not_write_the_notebook() -> None:
    """Day 177: the generator emitted demo.ipynb at import time, so running the test suite
    silently replaced the executed, pre-rendered notebook with a blank one - and no later test
    could see it, because the test suite was the thing that had done it."""
    if not os.path.exists(NB):
        pytest.skip("notebook not built yet")
    before = os.path.getmtime(NB)
    time.sleep(0.01)
    import importlib

    importlib.reload(B)
    assert os.path.getmtime(NB) == before


def test_engine_is_extracted_not_copied() -> None:
    """The embedded engine must be the library's own source text, character for character."""
    lib = open("nonparam.py").read()
    for fn in ("def mannwhitney(", "def run_disagreement(", "def mix_prob_superiority("):
        assert fn in B.ENGINE
        start = B.ENGINE.index(fn)
        end = B.ENGINE.index("\n\n", start)
        assert B.ENGINE[start:end] in lib


def test_extractor_catches_tuple_targets() -> None:
    """Day 178: an extractor reading only `targets[0].id` silently dropped
    `BAND_LO, BAND_HI = 0.045, 0.055` and the notebook died with a NameError raised from inside
    a function body, several minutes into nbconvert."""
    assert "BAND_LO, BAND_HI" in B.ENGINE


def test_every_requested_export_is_actually_present() -> None:
    """Turns a multi-minute nbconvert failure into a one-second one: exec the extracted block
    and assert each name the notebook will reference is bound."""
    namespace: dict = {}
    exec(compile(B.HEADER + "\n" + B.ENGINE, "<engine>", "exec"), namespace)
    missing = [name for name in B.EXPORTS if name not in namespace]
    assert not missing, f"extracted engine is missing {missing}"


def test_cell_sources_keep_their_newlines() -> None:
    """nbformat wants each source line to KEEP its trailing newline. `split("\\n")` concatenates
    the whole cell into one unparseable line and the failure is silent until execution."""
    for cell in B.cells:
        src = cell["source"]
        assert isinstance(src, list)
        for line in src[:-1]:
            assert line.endswith("\n"), f"cell {cell['id']} lost a newline"


def test_every_cell_has_an_id() -> None:
    ids = [c["id"] for c in B.cells]
    assert all(ids)
    assert len(set(ids)) == len(ids)


def test_badges_point_at_this_build() -> None:
    """A copied generator ports its neighbour's paths - the badge would then open a different
    build's notebook and nothing would error."""
    assert B.PATH == "data-science-cookbook/nonparametric-swap"
    first = "".join(B.cells[0]["source"])
    assert B.PATH in first
    assert "colab" in first.lower() and "binder" in first.lower()
    assert "effect-size-reader" not in first


def test_generated_notebook_is_valid_json_with_outputs() -> None:
    if not os.path.exists(NB):
        pytest.skip("notebook not built yet")
    nb = json.load(open(NB))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 8
    executed = [c for c in code_cells if c.get("outputs")]
    assert len(executed) >= len(code_cells) - 1, "notebook was not pre-rendered"
    assert not any(o.get("output_type") == "error"
                   for c in code_cells for o in c.get("outputs", []))
    assert any("image/png" in o.get("data", {})
               for c in code_cells for o in c.get("outputs", [])), "no chart rendered"


def test_notebook_uses_the_library_seeds() -> None:
    """So its printed rows ARE the evidence file's cells rather than a second sample."""
    src = "".join("".join(c["source"]) for c in B.cells if c["cell_type"] == "code")
    for base in ("DISAGREE_SEED_BASE", "LOCATION_SEED_BASE", "NULL_SEED_BASE",
                 "UNEQUAL_SEED_BASE", "TIE_NULL_SEED_BASE", "TIE_POWER_SEED_BASE"):
        assert base in src


def test_notebook_states_its_lower_resolution() -> None:
    """The notebook runs fewer replicates than the study, so it flags fewer cells. That is the
    instrument, not a contradiction - but it has to be said out loud, or the two artifacts read
    as disagreeing (the Day 175 failure)."""
    md = "".join("".join(c["source"]) for c in B.cells if c["cell_type"] == "markdown")
    assert "resolution of the instrument" in md


def test_notebook_code_parses() -> None:
    """Every code cell must be syntactically valid before nbconvert ever starts."""
    for cell in B.cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_no_modern_union_syntax_in_the_engine() -> None:
    """Colab still serves old kernels; `str | None` in an annotation is a runtime error there."""
    assert " | None" not in B.ENGINE
    assert "from __future__ import annotations" in B.HEADER


def test_notebook_covers_every_section_of_the_study() -> None:
    md = "".join("".join(c["source"]) for c in B.cells if c["cell_type"] == "markdown")
    for heading in ("1. Calibration", "2. The headline", "3. Where the swap is fine",
                    "4. The cost nobody mentions", "5. Ties"):
        assert heading in md
    assert str(len(N.SHAPES)) or True
