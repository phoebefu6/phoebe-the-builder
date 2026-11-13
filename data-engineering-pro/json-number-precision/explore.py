"""json-number-precision engine: 64-bit integer IDs through 9 JSON readers, compared to the integer that was sent.

A single big ID usually survives. What breaks it is the COLUMN: readers infer one type per column, and one row
that is null, written as "123.0", or too big for int64 turns every ID in that column into a float64 - which holds
integers exactly only up to 2**53. So every ID group is parsed in five column contexts: clean, plus one extra row of
each kind. Every value is fictional, seeded Generator(214).

ID groups (2,000 each):
  small     - sequential database keys below 2**31 (what a test fixture usually holds)
  boundary  - an auto-increment counter crossing 2**53 (2**53 - 1000 .. 2**53 + 999)
  snowflake - Twitter-style 64-bit IDs: (ms since epoch << 22) | (worker << 12) | sequence, from a 10-second burst
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np

SEED = 214
N = 2000
GROUPS = ["small", "boundary", "snowflake"]
EPOCH_MS = 1288834974657  # the published Twitter snowflake epoch; the IDs themselves are fictional
CONTEXTS: Dict[str, Optional[str]] = {
    "clean": None,
    "one null": '{"id": null}',
    "one 123.0": '{"id": 123.0}',
    "one > 2**63": '{"id": 9223372036854775813}',
    "one > 2**64": '{"id": 18446744073709551621}',
}


def make_ids(seed: int = SEED) -> Dict[str, List[int]]:
    g = np.random.default_rng(seed)
    small = list(range(100_001, 100_001 + N))
    boundary = list(range(2**53 - 1000, 2**53 + 1000))
    ms = int(g.integers(1_780_000_000_000, 1_790_000_000_000))
    snow = set()
    while len(snow) < N:
        t = ms + int(g.integers(0, 10_000)) - EPOCH_MS
        snow.add((t << 22) | (int(g.integers(0, 32)) << 12) | int(g.integers(0, 64)))
    return {"small": small, "boundary": boundary, "snowflake": sorted(snow)}


def ndjson(ids: List[int], extra: Optional[str]) -> str:
    rows = [f'{{"id": {i}}}' for i in ids] + ([extra] if extra else [])
    return "\n".join(rows) + "\n"


# --- readers: each takes NDJSON text and returns (values, dtype) for every row --------------------------------


def _json(text: str) -> tuple:
    return [json.loads(line)["id"] for line in text.splitlines()], "int"


def _json_double(text: str) -> tuple:
    # JavaScript's JSON.parse has exactly this behaviour: every number is an IEEE double
    return [json.loads(line, parse_int=float)["id"] for line in text.splitlines()], "double"


def _ujson(text: str) -> tuple:
    import ujson

    return [ujson.loads(line)["id"] for line in text.splitlines()], "int"


def _pydantic(text: str) -> tuple:
    import pydantic

    class Row(pydantic.BaseModel):
        id: Optional[int]

    # strict=False (the default) accepts 123.0 as 123; that is the documented lax mode
    return [Row.model_validate_json(line).id for line in text.splitlines()], "Optional[int]"


def _pandas(text: str, dtype: Any = True) -> tuple:
    import pandas as pd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_json(io.StringIO(text), lines=True, dtype=dtype)
    return [None if v is pd.NA or v != v else v for v in df["id"].tolist()], str(df["id"].dtype)


def _pandas_int64(text: str) -> tuple:
    return _pandas(text, {"id": "Int64"})


def _polars(text: str) -> tuple:
    import polars as pl

    s = pl.read_ndjson(io.BytesIO(text.encode()))["id"]
    return s.to_list(), str(s.dtype)


def _pyarrow(text: str) -> tuple:
    import pyarrow.json as paj

    col = paj.read_json(io.BytesIO(text.encode()))["id"]
    return col.to_pylist(), str(col.type)


def _duckdb(text: str, columns: str = "") -> tuple:
    import duckdb

    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        con = duckdb.connect()
        rows = con.execute(f"SELECT id, typeof(id) FROM read_json('{path}'{', columns=' + columns if columns else ''})").fetchall()
        con.close()
    finally:
        os.remove(path)
    return [r[0] for r in rows], rows[0][1]


READERS: Dict[str, Callable[[str], tuple]] = {
    "json": _json,
    "JSON.parse (double)": _json_double,
    "ujson": _ujson,
    "pydantic int": _pydantic,
    "pandas read_json": _pandas,
    "pandas dtype=Int64": _pandas_int64,
    "polars": _polars,
    "pyarrow": _pyarrow,
    "duckdb": _duckdb,
}
def _pandas_str(text: str) -> tuple:
    # a string column is joined as a string: '2.06e+18' is not the ID even if it rounds to it
    vals, dtype = _pandas(text, {"id": str})
    return [None if v is None or str(v) in ("nan", "None") else int(v) if str(v).lstrip("-").isdigit() else v
            for v in vals], dtype


def _pyarrow_schema(text: str) -> tuple:
    import pyarrow as pa
    import pyarrow.json as paj

    opts = paj.ParseOptions(explicit_schema=pa.schema([("id", pa.int64())]))
    col = paj.read_json(io.BytesIO(text.encode()), parse_options=opts)["id"]
    return col.to_pylist(), str(col.type)


def _duckdb_hugeint(text: str) -> tuple:
    return _duckdb(text, "{'id': 'HUGEINT'}")


def _json_frame(text: str) -> tuple:
    import pandas as pd

    s = pd.Series(_json(text)[0], dtype=object)
    return [None if v is None else v for v in s.tolist()], "object (Python int)"


# The fix each reader's documentation offers, run through the same five columns
FIXES: Dict[str, Callable[[str], tuple]] = {
    "pandas dtype=Int64": _pandas_int64,
    "pandas dtype=str": _pandas_str,
    "pyarrow explicit int64": _pyarrow_schema,
    "duckdb columns=HUGEINT": _duckdb_hugeint,
    "json, then a DataFrame": _json_frame,
}

OPT_OUT = {
    "json": "none needed: Python ints are unbounded",
    "JSON.parse (double)": "a reviver reading context.source into BigInt (Node 21+), or send IDs as strings",
    "ujson": "none needed",
    "pydantic int": "none needed; strict=True also refuses 123.0",
    "pandas read_json": "dtype={'id': 'Int64'} - which this study shows does NOT fix it",
    "pandas dtype=Int64": "send IDs as strings and convert after",
    "polars": "pass a schema; it already widens to Int128",
    "pyarrow": "explicit_schema with pa.int64() / pa.uint64()",
    "duckdb": "columns={'id': 'HUGEINT'} or 'UBIGINT'",
}


def parse(reader: str, ids: List[int], context: str) -> Dict[str, Any]:
    """Parse one column; score the first len(ids) values against what was sent."""
    fn = READERS.get(reader) or FIXES[reader]
    try:
        got, dtype = fn(ndjson(ids, CONTEXTS[context]))
    except Exception as e:  # noqa: BLE001 - a refusal is a result, not a crash
        return {"exact": 0, "wrong": 0, "raised": f"{type(e).__name__}: {str(e)[:60]}", "dtype": "-",
                "merged": 0, "max_off": 0}
    got = got[: len(ids)]
    wrong = [(a, b) for a, b in zip(ids, got) if b is None or b != a]
    distinct = len({b for b in got if b is not None})  # 2**53 and 2**53 + 0.0 are one key, as in a join
    return {"exact": len(ids) - len(wrong), "wrong": len(wrong), "raised": "", "dtype": dtype,
            "merged": len(ids) - distinct, "max_off": max((abs(int(b) - a) for a, b in wrong if isinstance(b, (int, float))),
                                                          default=0)}


def study(seed: int = SEED) -> Dict[str, Any]:
    ids = make_ids(seed)
    return {"readers": {r: {c: {g: parse(r, ids[g], c) for g in GROUPS} for c in CONTEXTS} for r in READERS},
            "fixes": {r: {c: parse(r, ids["snowflake"], c) for c in CONTEXTS} for r in FIXES},
            "first": {r: first_failure(r) for r in READERS}}


def first_failure(reader: str, context: str = "clean") -> Optional[int]:
    """Smallest positive integer this reader gets wrong in that context, by binary search over powers of two
    then a linear walk - every integer below 2**53 is a double, so the first miss is 2**53 + 1 or never."""
    for k in range(40, 70):
        lo = 2**k
        probe = [lo + d for d in range(4)]
        r = parse(reader, probe, context)
        if r["raised"]:
            return None
        if r["wrong"]:
            return next(p for p in probe if parse(reader, [p], context)["wrong"])
    return None


def seeds(n: int = 30) -> Dict[str, Any]:
    """Snowflake IDs merged per reader in the 'one null' context, over n more seeds."""
    out: Dict[str, List[int]] = {r: [] for r in READERS}
    for s in range(SEED + 1, SEED + 1 + n):
        snow = make_ids(s)["snowflake"]
        for r in READERS:
            out[r].append(parse(r, snow, "one null")["merged"])
    return {"n": n, "range": {r: [min(v), max(v)] for r, v in out.items()}}


def silent(cell: Dict[str, Any]) -> bool:
    return cell["wrong"] > 0 and not cell["raised"]
