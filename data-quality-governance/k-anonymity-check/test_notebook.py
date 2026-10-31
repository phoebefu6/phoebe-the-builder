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
    assert B.PATH == "data-quality-governance/k-anonymity-check"
    first = "".join(B.cells[0]["source"])
    assert B.PATH in first and "dedupe-key-choice/demo" not in first


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
    for n in (1, 2, 3, 4, 5, 6):
        for line in section(n):
            assert line in text, f"notebook does not reproduce section {n}: {line!r}"


def test_notebook_prose_numbers_match_the_study() -> None:
    """Prose is hand-written; every headline number in it must be one the study computes."""
    import explore as E

    r = E.study()
    by = {tuple(p["levels"]): p for p in r["ladder"]}
    sub = {tuple(p["columns"]): p for p in r["subsets"]}
    link = {(tuple(k["levels"]), k["coverage"]): k for k in r["linkage"]}
    sy = by[("sector (2 digits)", "year", "kept")]
    at5 = next(s for s in r["sweep"] if s["frac"] == 0.05 and tuple(s["levels"]) == ("sector (2 digits)", "year", "kept"))
    prose = "".join("".join(c["source"]) for c in B.cells if c["cell_type"] == "markdown")
    claims = {
        "town of 60,000": r["pop_n"] == 60_000,
        "3,000-row clinic export": r["export_n"] == 3000,
        "holds 15 people": r["postcodes"]["min"] == 15,
        "makes 498 of the 3,000": sub[("date of birth",)]["unique_pop"] == 498,
        "make 2,932 of 3,000": sub[("postcode", "date of birth")]["unique_pop"] == 2932,
        "that to\n2,964": sub[("postcode", "date of birth", "sex")]["unique_pop"] == 2964,
        "added 32 rows": 2964 - 2932 == 32,
        "the second added 2,434": 2932 - sub[("date of birth",)]["unique_pop"] == 2434,
        "leaves 2,487 export rows unique\nand 1,097": (by[("6 digits", "year", "kept")]["unique_export"], by[("6 digits", "year", "kept")]["unique_pop"]) == (2487, 1097),
        "leaves 111 town-unique": by[("6 digits", "decade", "kept")]["unique_pop"] == 111,
        "Only 6 of the 40": len(r["k_anonymous"]) == 6 and len(r["ladder"]) == 40,
        "sector + sex, 40 classes": tuple(r["best"]["levels"]) == ("sector (2 digits)", "dropped", "kept") and r["best"]["classes"] == 40,
        "199\nclasses, 51 rows suppressed": (r["release"]["classes"], r["release"]["below_k"]) == (199, 51),
        "40.1% of the 3,000": round(at5["unique_export_pct"], 1) == 40.1,
        "0.03% - one row": round(at5["unique_pop_pct"], 2) == 0.03 and sy["unique_pop"] == 1,
        "re-identifies 1,203 people": sy["unique_export"] == 1203,
        "same, 98.8%": round(next(s for s in r["sweep"] if s["frac"] == 0.05 and tuple(s["levels"]) == E.FULL)["unique_pop_pct"], 1) == 98.8,
        "rows - 2,964 - and is never wrong": link[(E.FULL, 1.0)] == {**link[(E.FULL, 1.0)], "right": 2964, "wrong": 0},
        "loses 599 rows": link[(E.FULL, 0.8)]["none"] == 599,
        "right 977 times and wrong 101 times": (link[(("6 digits", "year", "kept"), 0.8)]["right"], link[(("6 digits", "year", "kept"), 0.8)]["wrong"]) == (977, 101),
        "31 rows sit in classes": r["release"]["near_homog_rows"] == 31,
        "Ten of the 40": sum(1 for p in r["ladder"] if p["sensitive_confident_rows"] > 0) == 10,
        "baseline 11.7%": round(100 * r["sensitive_base"] / r["export_n"], 1) == 11.7,
        "2,740\nrows below k = 5 and the town has 52": (sy["below_k"], sy["below_k_pop"]) == (2740, 52),
        "97.7% of the exported": round(100 * 2932 / 3000, 1) == 97.7,
    }
    for text, ok in claims.items():
        assert text in prose, f"{text!r} missing from the notebook prose"
        assert ok, f"{text!r} no longer matches the study"
