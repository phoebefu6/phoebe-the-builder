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
    assert B.PATH == "data-quality-governance/golden-record"
    first = "".join(B.cells[0]["source"])
    assert B.PATH in first and "table-checksum/demo" not in first


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
    for n in (1, 2, 3, 4):
        for line in section(n):
            assert line in text, f"notebook does not reproduce section {n}: {line!r}"


def test_notebook_prose_numbers_match_the_study() -> None:
    """Prose is hand-written; every headline number in it must be one the study computes."""
    r = json.load(open("results.json"))
    a, w = r["acc"], r["whole"]
    pc = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    prose = "".join("".join(c["source"]) for c in B.cells if c["cell_type"] == "markdown")
    claims = {
        "364 fictional customers": r["clusters"] == 364,
        f"name ({pc(a['source_priority']['name'])}": True,
        f"on address ({pc(a['source_priority']['address'])}": True,
        f"name\n({pc(a['most_recent']['name'])})": True,
        f"for {pc(w['per_field'])} of customers against {pc(w['majority'])}": max(w, key=w.get) == "per_field"
        and max((k for k in w if k != "per_field"), key=w.get) == "majority",
        f"({pc(a['per_field']['email'])} vs {pc(a['majority']['email'])})": a["majority"]["email"] > a["per_field"]["email"],
        "only 119 of 364": r["distinct"]["1"] == 119,
        "For 80 they": r["distinct"].get("3", 0) + r["distinct"].get("4", 0) == 80,
        f"({pc(r['ts_fix']['record_ts'])} -> {pc(r['ts_fix']['field_ts'])})": True,
        "All 20 of the 20": r["stale_sync"] == {"clusters": 20, "majority_stale": 20},
        f"address to {pc(r['ts_fix']['lineage_ts'])}, the ceiling": r["ts_fix"]["lineage_ts"] == r["ceiling"]["address"],
        f"163 of 364 customers ({pc(r['missing_any'] / r['clusters'])})": r["missing_any"] == 163,
        f"only {pc(r['oracle_whole'])}": True,
    }
    for text, ok in claims.items():
        assert text in prose, f"{text!r} missing from the notebook prose"
        assert ok, f"{text!r} no longer matches the study"
