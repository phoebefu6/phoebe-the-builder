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
    assert B.PATH == "data-quality-governance/table-checksum"
    first = "".join(B.cells[0]["source"])
    assert B.PATH in first and "k-anonymity-check/demo" not in first


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
    f, ta = r["float"], r["tally"]
    prose = "".join("".join(c["source"]) for c in B.cells if c["cell_type"] == "markdown")
    claims = {
        "1,000-row keyless": r["rows"] == 1000,
        "20 exact duplicate\npairs": r["dup_pairs"] == 20,
        "fails all 7": ta["csv_md5"]["false_alarms"] == list(r["scores"]["benign"]),
        "miss 4 of 6": len(ta["count_sums"]["misses"]) == 4 and len(r["scores"]["mutations"]) == 6,
        "caught in 1 trial of 100": round(100 * (1 - r["scores"]["mutations"]["two offsetting cents"]["match"]["count_sums"])) == 1,
        "1 false alarm of 7, 0 misses of 6": (len(ta["canonical"]["false_alarms"]), len(ta["canonical"]["misses"])) == (1, 0),
        "at least 6 of 7": all(len(ta[k]["false_alarms"]) >= 6 for k in ("csv_md5", "xor_rows", "sum_rows")),
        "259 of 1,000": f["bits_differ"] == 259,
        "13 true amounts": f["half_cent_exact"] == 13,
        "event 10459": f["straddle"][0][0] == 10459 and f["cents_differ"] == 1,
        "7 false alarms of 7": len(ta["csv_md5"]["false_alarms"]) == 7,
        "12 of 13 right": 13 - len(ta["canonical"]["false_alarms"]) - len(ta["canonical"]["misses"]) == 12,
    }
    for text, ok in claims.items():
        assert text in prose, f"{text!r} missing from the notebook prose"
        assert ok, f"{text!r} no longer matches the study"
