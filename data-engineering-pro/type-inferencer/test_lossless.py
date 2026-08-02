"""The tool's own claim, as a test: nothing it proposes may lose information.

Run with `python test_lossless.py` (no pytest needed - it is what CI runs).
"""

from __future__ import annotations

from type_infer import (
    EXPECTED,
    Policy,
    demo_rows,
    emit_ddl,
    infer_column,
    infer_table,
    is_lossy,
    run_benchmark,
)

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


rows = demo_rows()
results = infer_table(rows)

print("core claim")
for r in results:
    raw = [row[r.name] for row in rows]
    check(f"{r.name} is not lossy as {r.type}", not is_lossy(r.type, raw))

print("\nanswer key")
for r in results:
    check(f"{r.name} == {EXPECTED[r.name]}", r.type.sql("postgres") == EXPECTED[r.name],
          f"got {r.type.sql('postgres')}")

print("\nthe traps stay untyped, and say why")
by_name = {r.name: r for r in results}
for col in ("zip_code", "ship_date", "legacy_amount"):
    check(f"{col} abstained with a finding", by_name[col].abstained)
check("flag_01 is not a BOOLEAN", by_name["flag_01"].type.kind == "SMALLINT")
check("weight_kg found the row-210 decimal", by_name["weight_kg"].type.scale == 1)
check("weight_kg is nullable", by_name["weight_kg"].type.nullable)

print("\nbenchmark")
bench = run_benchmark(rows)
mine = bench["lossless (this tool)"]
naive = bench["naive cast (200-row sample)"]
check("lossless corrupts nothing", not any(v == "lossy" for v in mine.values()))
check("lossless asserts nothing unproven", not any(v == "unsafe" for v in mine.values()))
check("the naive strawman is genuinely worse", sum(1 for v in naive.values() if v == "lossy") >= 4,
      f"{sum(1 for v in naive.values() if v == 'lossy')} lossy columns")

print("\nedge cases do not crash or over-claim")
edge = {
    "empty": ([""] * 10, lambda r: r.type.kind == "TEXT"),
    "all_null_tokens": (["N/A", "null", "-", ""], lambda r: r.type.kind == "TEXT"),
    "three_rows": (["1", "2", "3"], lambda r: r.type.nullable),        # too few rows for NOT NULL
    "small_ints": (["1", "2", "3"], lambda r: r.type.kind == "INTEGER"),  # and too few to narrow
    "sign_prefix": (["+5", "+6", "7"], lambda r: r.abstained),
    "single_bool": (["yes", "yes"], lambda r: r.abstained),
    "date_conflict": (["25/04/2026", "04/25/2026"], lambda r: r.type.kind in ("VARCHAR", "TEXT")),
    "beyond_int64": (["99999999999999999999999", "1"], lambda r: r.type.kind == "DECIMAL"),
    "unicode": (["café", "naïve"], lambda r: r.type.kind == "VARCHAR"),
}
for name, (values, predicate) in edge.items():
    r = infer_column(name, values)
    check(f"{name} -> {r.type}", predicate(r) and not is_lossy(r.type, values))

r = infer_column("padded", [" 5 ", " 6", "7"])
check("whitespace is trimmed and flagged",
      r.type.kind == "INTEGER" and any(f.code == "whitespace-padded" for f in r.findings))

ragged = infer_table([{"a": "1"}, {"a": "2", "b": "x"}])
check("ragged rows keep late columns", [x.name for x in ragged] == ["a", "b"])

print("\nDDL")
for dialect in ("postgres", "duckdb", "sqlite"):
    ddl = emit_ddl("orders", results, dialect)
    check(f"{dialect} DDL emits every column",
          all(r.name in ddl for r in results) and ddl.startswith("CREATE TABLE orders ("))

tight = infer_table(rows, Policy(min_rows_for_not_null=1000))
check("policy is respected (NOT NULL suppressed)", all(r.type.nullable for r in tight))

print()
if FAILURES:
    raise SystemExit(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
print("all checks passed")
