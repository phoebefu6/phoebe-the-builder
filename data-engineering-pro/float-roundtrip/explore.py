"""float-roundtrip engine: one column of float64 values through 15 writers, read back, compared bit for bit.

Four fictional value groups, seeded Generator(213):
  entered  - amounts a person typed, two decimals (what a test fixture usually holds)
  computed - entered amounts after an FX conversion and a 3-way split (what a pipeline actually holds)
  measured - sensor-style readings spread over 20 orders of magnitude
  edge     - -0.0, NaN, +/-inf, the smallest subnormal, the largest double, 0.1 + 0.2, 2**53 + 2
"""

from __future__ import annotations

import csv
import io
import json
import math
import pickle
import sqlite3
import struct
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

SEED = 213
N = 2000
GROUPS = ["entered", "computed", "measured", "edge"]
EDGE = [0.0, -0.0, float("nan"), float("inf"), float("-inf"), 5e-324, 2.2250738585072014e-308,
        1.7976931348623157e308, 0.1 + 0.2, float(2**53 + 2), 1e16 + 2.0, 123456789.12345679]


def make_values(seed: int = SEED) -> Dict[str, List[float]]:
    g = np.random.default_rng(seed)
    entered = np.round(g.uniform(0.01, 50_000, N), 2)
    rate = np.round(g.uniform(0.5, 1.6, N), 6)
    computed = entered * rate / 3.0
    measured = g.uniform(1, 10, N) * 10.0 ** g.integers(-10, 10, N)
    return {"entered": [float(x) for x in entered], "computed": [float(x) for x in computed],
            "measured": [float(x) for x in measured], "edge": list(EDGE)}


# --- writers: each takes the whole column and returns what a reader gets back (None = lost / refused) -----------

def _each(fn: Callable[[float], float]) -> Callable[[List[float]], List[Optional[float]]]:
    def run(xs: List[float]) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for x in xs:
            try:
                out.append(fn(x))
            except (ValueError, OverflowError):
                out.append(None)
        return out
    return run


def _json(xs: List[float]) -> List[Optional[float]]:
    return json.loads(json.dumps(xs))


def _json_strict(xs: List[float]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for x in xs:  # one document per value, as an API would send it; NaN/inf are not JSON
        try:
            out.append(json.loads(json.dumps({"v": x}, allow_nan=False))["v"])
        except ValueError:
            out.append(None)
    return out


def _csv_module(xs: List[float]) -> List[Optional[float]]:
    buf = io.StringIO()
    csv.writer(buf).writerows([[x] for x in xs])
    return [float(r[0]) for r in csv.reader(io.StringIO(buf.getvalue()))]


def _pandas(xs: List[float], precision: Optional[str] = None, fmt: Optional[str] = None) -> List[Optional[float]]:
    text = pd.DataFrame({"v": xs}).to_csv(index=False, float_format=fmt)
    v = pd.read_csv(io.StringIO(text), float_precision=precision)["v"]
    return [None if (isinstance(a, float) and math.isnan(a) and not math.isnan(b)) else float(a)
            for a, b in zip(v.tolist(), xs)]


def _sqlite(xs: List[float], as_text: bool) -> List[Optional[float]]:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (i INTEGER, v REAL)")
    con.executemany("INSERT INTO t VALUES (?, ?)", list(enumerate(xs)))
    q = "SELECT CAST(v AS TEXT) FROM t ORDER BY i" if as_text else "SELECT v FROM t ORDER BY i"
    rows = [r[0] for r in con.execute(q)]
    con.close()
    return [None if r is None else float(r) for r in rows]


def _savetxt(xs: List[float]) -> List[Optional[float]]:
    buf = io.StringIO()
    np.savetxt(buf, np.array(xs))
    return [float(v) for v in np.loadtxt(io.StringIO(buf.getvalue()), ndmin=1)]


WRITERS: Dict[str, Callable[[List[float]], List[Optional[float]]]] = {
    "repr": _each(lambda x: float(repr(x))),
    "%.17g": _each(lambda x: float("%.17g" % x)),
    "pickle": lambda xs: pickle.loads(pickle.dumps(xs)),
    "np.savetxt": _savetxt,
    "sqlite REAL": lambda xs: _sqlite(xs, False),
    "json": _json,
    "json strict": _json_strict,
    "csv module": _csv_module,
    "pandas round_trip": lambda xs: _pandas(xs, precision="round_trip"),
    "pandas csv": lambda xs: _pandas(xs),
    "pandas legacy": lambda xs: _pandas(xs, precision="legacy"),
    "%.15g": _each(lambda x: float("%.15g" % x)),
    "sqlite TEXT": lambda xs: _sqlite(xs, True),
    "pandas %.6f": lambda xs: _pandas(xs, fmt="%.6f"),
    "float32": _each(lambda x: struct.unpack("f", struct.pack("f", x))[0]),
}
WHAT = {
    "repr": "Python repr / str: shortest string that reads back",
    "%.17g": "17 significant digits",
    "pickle": "binary",
    "np.savetxt": "NumPy default fmt %.18e",
    "sqlite REAL": "SQLite REAL column, read as a number",
    "json": "json.dumps / loads (writes NaN, Infinity - not JSON)",
    "json strict": "json.dumps(allow_nan=False), one document per value",
    "csv module": "csv.writer / float()",
    "pandas round_trip": "to_csv / read_csv(float_precision='round_trip')",
    "pandas csv": "to_csv / read_csv defaults (the file is exact; the reader is not)",
    "pandas legacy": "read_csv(float_precision='legacy')",
    "%.15g": "15 significant digits (DBL_DIG, what a spreadsheet shows)",
    "sqlite TEXT": "SQLite CAST(v AS TEXT), as a text export does",
    "pandas %.6f": "to_csv(float_format='%.6f')",
    "float32": "a FLOAT (4-byte) column",
}


def bits(x: float) -> int:
    """Ordered integer for a double: adjacent doubles differ by 1, so |a - b| is the ULP distance."""
    i = struct.unpack("<q", struct.pack("<d", x))[0]
    return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)


def compare(xs: List[float], ys: List[Optional[float]]) -> Dict[str, float]:
    exact = changed = lost = 0
    max_ulp = 0
    for x, y in zip(xs, ys):
        if y is None or (math.isnan(y) and not math.isnan(x)):
            lost += 1
        elif math.isnan(x) and math.isnan(y):
            exact += 1
        elif struct.pack("<d", x) == struct.pack("<d", y):
            exact += 1
        else:
            changed += 1
            if math.isfinite(x) and math.isfinite(y):
                max_ulp = max(max_ulp, abs(bits(x) - bits(y)))
    return {"n": len(xs), "exact": exact, "changed": changed, "lost": lost, "max_ulp": max_ulp}


def run(values: Dict[str, List[float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    return {w: {g: compare(values[g], fn(values[g])) for g in GROUPS} for w, fn in WRITERS.items()}


def ledger(values: Dict[str, List[float]], writer: str) -> Dict[str, float]:
    """What the change does downstream on the computed column: total drift vs keys that stop matching."""
    xs = values["computed"]
    ys = WRITERS[writer](xs)
    kept = [y for y in ys if y is not None]
    before, after = math.fsum(xs), math.fsum(kept)
    orig = {struct.pack("<d", x) for x in xs}
    miss = sum(1 for y in ys if y is None or struct.pack("<d", y) not in orig)
    return {"total_before": before, "total_after": after, "drift": after - before,
            "join_miss": miss / len(xs), "dedupe_after": len(set(kept)), "dedupe_before": len(set(xs))}


def edge_detail(writer: str) -> List[str]:
    """What each edge value came back as, as text."""
    ys = WRITERS[writer](list(EDGE))
    return ["lost" if y is None else repr(y) for y in ys]


def study(seed: int = SEED) -> dict:
    v = make_values(seed)
    return {"writers": run(v), "ledger": {w: ledger(v, w) for w in WRITERS},
            "edge": {w: edge_detail(w) for w in WRITERS}}


def seeds(n: int = 30) -> dict:
    """Changed share of the computed column per writer, across n more worlds."""
    out: Dict[str, List[float]] = {w: [] for w in WRITERS}
    for s in range(1, n + 1):
        v = make_values(SEED + s)
        for w, fn in WRITERS.items():
            c = compare(v["computed"], fn(v["computed"]))
            out[w].append((c["changed"] + c["lost"]) / c["n"])
    return {"n": n, "range": {w: (min(r), max(r)) for w, r in out.items()}}
