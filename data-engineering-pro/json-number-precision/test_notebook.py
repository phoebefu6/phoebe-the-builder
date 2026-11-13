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
    assert B.PATH == "data-engineering-pro/json-number-precision"
    first = "".join(B.cells[0]["source"])
    assert B.PATH in first and "float-roundtrip/demo" not in first


def test_code_cells_parse_and_no_modern_unions() -> None:
    for cell in B.cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    assert " | None" not in B.ENGINE


def test_determinism_note_exists() -> None:
    text = "".join("".join(c["source"]) for c in B.cells)
    assert "seeded" in B.__doc__.lower() and "evidence.txt" in text


def test_executed_notebook_matches_the_evidence_file() -> None:
    """Seeded parts must print exactly what evidence.py stored."""
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
    section = lambda n: ev.split(f"\n{n}. ")[1].split("\n\n")[0].splitlines()[1:]  # noqa: E731
    for n in (1, 2, 3, 4, 5, 7):
        for line in section(n):
            assert line in text, f"notebook does not reproduce section {n}: {line!r}"


def test_notebook_prose_numbers_match_the_study() -> None:
    """Prose is hand-written; every headline number in it must be one the study computes."""
    import explore as E

    res = json.load(open("results.json"))["study"]
    rd, fx = res["readers"], res["fixes"]
    prose = "".join("".join(c["source"]) for c in B.cells if c["cell_type"] == "markdown")
    ok_all = [k for k in E.READERS if all(not rd[k][c][g]["wrong"] and not rd[k][c][g]["raised"]
                                          for c in E.CONTEXTS for g in E.GROUPS)]
    silent = sum(any(E.silent(rd[k][c][g]) for g in ("boundary", "snowflake")) for k in E.READERS for c in E.CONTEXTS)
    claims = {
        "Only 3 of 9 readers give every ID back": ok_all == ["json", "ujson", "pydantic int"],
        "14 of 45\nreader-contexts": silent == 14,
        "int64 while 1,960 of 2,000": rd["pandas read_json"]["one 123.0"]["snowflake"]["dtype"] == "int64"
        and rd["pandas read_json"]["one 123.0"]["snowflake"]["wrong"] == 1960,
        "499 IDs merge": rd["JSON.parse (double)"]["clean"]["boundary"]["merged"] == 499,
        "up\nto 63": rd["JSON.parse (double)"]["clean"]["snowflake"]["max_off"] == 63,
        "duckdb with\n`columns={'id': 'HUGEINT'}`": all(not d["wrong"] and not d["raised"]
                                                         for d in fx["duckdb columns=HUGEINT"].values()),
    }
    for text, ok in claims.items():
        assert text in prose, f"{text!r} missing from the notebook prose"
        assert ok, f"{text!r} no longer matches the study"
