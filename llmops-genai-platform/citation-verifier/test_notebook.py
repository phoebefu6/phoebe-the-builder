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
    lib = open("citation.py").read()
    for fn in ("def build(", "def check(", "def audit("):
        start = B.ENGINE.index(fn)
        end = B.ENGINE.find("\n\n", start) % (len(B.ENGINE) + 1)
        assert B.ENGINE[start:end] in lib


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
    assert B.PATH == "llmops-genai-platform/citation-verifier"
    first = "".join(B.cells[0]["source"])
    assert B.PATH in first and "agent-trajectory-eval" not in first[:600]


def test_code_cells_parse() -> None:
    for cell in B.cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_no_modern_union_syntax() -> None:
    assert " | None" not in B.ENGINE
    assert "from __future__ import annotations" in B.HEADER


def test_resolution_note_exists() -> None:
    """Day 179: a notebook at fewer replicates flags fewer cells; that must be said."""
    text = "".join("".join(c["source"]) for c in B.cells)
    assert "Resolution of the instrument" in text and "evidence.txt" in text


def test_executed_notebook_matches_the_evidence_file() -> None:
    """Deterministic parts (calibration, exemplar) must print exactly what evidence.py stored."""
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
    for k, d in R["deltas"].items():
        assert f"{k:<32} {d * 100:+6.1f} pts" in text
    v2 = R["models"]["v2 always cite"]
    assert f"supported by its citation {v2['p_supported']:.1%}, claim true {v2['p_true']:.1%}" in text
    o = R["order"]
    assert f"flip scores above the paraphrase in {o['flip_above']:.1%} of pairs" in text
