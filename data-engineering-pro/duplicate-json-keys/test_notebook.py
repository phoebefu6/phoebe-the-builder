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


def test_every_requested_export_is_bound() -> None:
    namespace: dict = {}
    engine_cell = next(c for c in B.cells if c["cell_type"] == "code")
    exec(compile("".join(engine_cell["source"]), "<engine>", "exec"), namespace)
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
    assert B.PATH == "data-engineering-pro/duplicate-json-keys"
    first = "".join(B.cells[0]["source"])
    assert B.PATH in first and "json-number-precision/demo" not in first


def test_code_cells_parse_and_no_modern_unions() -> None:
    for cell in B.cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    assert " | None" not in B.ENGINE
    assert "__main__" not in B.EVIDENCE, "the notebook would rewrite evidence.txt"


def test_executed_notebook_matches_the_evidence_file() -> None:
    if not os.path.exists(NB) or not os.path.exists("evidence.txt"):
        pytest.skip("notebook or evidence not built yet")
    nb = json.load(open(NB))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c.get("outputs") for c in code_cells[1:-1]), "notebook was not pre-rendered"
    outs = [o for c in code_cells for o in c["outputs"]]
    assert not any(o.get("output_type") == "error" for o in outs)
    assert any("image/png" in o.get("data", {}) for o in outs), "no chart rendered"
    text = "".join("".join(o.get("text", "")) for o in outs)
    ev = open("evidence.txt").read()
    for n in (1, 2, 3, 4):
        for line in ev.split(f"\n{n}. ")[1].split("\n\n")[0].splitlines()[1:]:
            assert line in text, f"notebook does not reproduce section {n}: {line!r}"


def test_notebook_prose_numbers_match_the_study() -> None:
    res = json.load(open("results.json"))["study"]
    m, fd, d = res["matrix"], res["feed"], res["detect"]
    prose = "".join("".join(c["source"]) for c in B.cells if c["cell_type"] == "markdown")
    top = {k: v["top level"] for k, v in m.items()}
    claims = {
        "First-wins readers see none": all(fd[k]["refunds"] == 0 for k, v in top.items() if v == "first"),
        "refuse all 2,000 rows because of 46": fd["_truth"]["refunds"] == 46 and all(
            fd[k]["raised"] for k, v in top.items() if v == "refuse"),
        "catches every one": d["strict hook"] == 46 and d["regex"] < 46,
        "turns `1` then `\"x\"` into the string `'1'`": m["polars read_ndjson"]["type change"] == "other ('1')",
        "duckdb still returns the wrong one": m["duckdb read_json"]["case only (not a dup)"].startswith("WRONG"),
    }
    for text, ok in claims.items():
        assert text in prose, f"{text!r} missing from the notebook prose"
        assert ok, f"{text!r} no longer matches the study"
