"""Tests for the notebook generator - each guards a defect an earlier build in this series shipped."""

from __future__ import annotations

import ast
import json
import os
import time

import build_notebook as B
import pytest

NB = "demo.ipynb"


def test_import_does_not_write_the_notebook() -> None:
    """Day 177: an import-time write replaced the executed notebook with a blank one."""
    if not os.path.exists(NB):
        pytest.skip("notebook not built yet")
    before = os.path.getmtime(NB)
    time.sleep(0.01)
    import importlib

    importlib.reload(B)
    assert os.path.getmtime(NB) == before


def test_engine_is_extracted_not_copied() -> None:
    lib = open("proportion.py").read()
    for fn in ("def p_barnard(", "def p_fisher(", "def expected_count_rule("):
        start = B.ENGINE.index(fn)
        end = B.ENGINE.index("\n\n", start)
        assert B.ENGINE[start:end] in lib


def test_extractor_catches_tuple_targets() -> None:
    """Day 178: `BAND_LO, BAND_HI = ...` was dropped by a targets[0].id extractor."""
    assert "BAND_LO, BAND_HI" in B.ENGINE


def test_every_requested_export_is_bound() -> None:
    namespace: dict = {}
    exec(compile(B.HEADER + "\n" + B.ENGINE, "<engine>", "exec"), namespace)
    missing = [n for n in B.EXPORTS if n not in namespace]
    assert not missing, f"extracted engine is missing {missing}"


def test_cell_sources_keep_their_newlines_and_ids() -> None:
    ids = []
    for cell in B.cells:
        for line in cell["source"][:-1]:
            assert line.endswith("\n"), f"cell {cell['id']} lost a newline"
        ids.append(cell["id"])
    assert len(set(ids)) == len(ids) and all(ids)


def test_badges_point_at_this_build() -> None:
    """A copied generator ports its neighbour's paths; the badge would open the wrong notebook."""
    assert B.PATH == "data-science-cookbook/proportion-test"
    first = "".join(B.cells[0]["source"])
    assert B.PATH in first and "ci-overlap-fallacy" not in first


def test_code_cells_parse() -> None:
    for cell in B.cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_no_modern_union_syntax() -> None:
    assert " | None" not in B.ENGINE
    assert "from __future__ import annotations" in B.HEADER


def test_executed_notebook_matches_the_evidence_file() -> None:
    """Exact numbers, same designs: the notebook's worst z row must be evidence.txt's, verbatim."""
    if not os.path.exists(NB) or not os.path.exists("results.json"):
        pytest.skip("notebook or evidence not built yet")
    nb = json.load(open(NB))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c.get("outputs") for c in code_cells), "notebook was not pre-rendered"
    outs = [o for c in code_cells for o in c["outputs"]]
    assert not any(o.get("output_type") == "error" for o in outs)
    assert any("image/png" in o.get("data", {}) for o in outs), "no chart rendered"
    text = "".join("".join(o.get("text", "")) for o in outs)
    R = json.load(open("results.json"))
    worst = max(R["size"], key=lambda r: r["z"])
    assert f"{worst['z']:.4f}" in text
    assert f"{R['exemplar']['fisher']:.4f}" in text
