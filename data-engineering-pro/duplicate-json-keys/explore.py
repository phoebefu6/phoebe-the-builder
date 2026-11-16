"""Which value wins when a JSON object names the same key twice?

RFC 8259 says names "SHOULD be unique" and that receivers of a non-unique object are unpredictable: some report
the last pair, some report an error. So every reader here is spec-compliant, and they disagree. This module sends
one set of fixtures and one seeded order feed through 14 readers and records, per reader, which value came back.

Every value is fictional. The feed is seeded Generator(215).
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from typing import Any, Callable, Dict, List

import numpy as np

warnings.filterwarnings("ignore")

Lines = List[str]
Path = List[str]


def _dig(obj: Any, path: Path) -> Any:
    for k in path:
        obj = obj[k]
    return obj


def _per_line(loads: Callable[[str], Any]) -> Callable[[Lines, Path], List[Any]]:
    return lambda lines, path: [_dig(loads(s), path) for s in lines]


def _strict_pairs(pairs: List[Any]) -> Dict[str, Any]:
    seen: Dict[str, Any] = {}
    for k, v in pairs:
        if k in seen:
            raise ValueError(f"duplicate key {k!r}")
        seen[k] = v
    return seen


def _strict(s: str) -> Any:
    return json.loads(s, object_pairs_hook=_strict_pairs)


def _ujson(s: str) -> Any:
    import ujson

    return ujson.loads(s)


def _pydantic(s: str) -> Any:
    import pydantic

    return pydantic.TypeAdapter(dict).validate_json(s)


def _pyyaml(s: str) -> Any:
    import yaml

    return yaml.safe_load(s)


def _ruamel(s: str) -> Any:
    from ruamel.yaml import YAML

    return YAML(typ="safe").load(s)


def _frame(df: Any, path: Path) -> List[Any]:
    """A column reader turns path ['o','a'] into column 'o' then a dict lookup."""
    col = df[path[0]]
    vals = col.to_list() if hasattr(col, "to_list") else list(col)
    return [_dig(v, path[1:]) if len(path) > 1 else v for v in vals]


def _pandas(lines: Lines, path: Path) -> List[Any]:
    import pandas as pd

    return _frame(pd.read_json(io.StringIO("\n".join(lines) + "\n"), lines=True), path)


def _polars(lines: Lines, path: Path) -> List[Any]:
    import polars as pl

    return _frame(pl.read_ndjson(io.StringIO("\n".join(lines) + "\n")), path)


def _pyarrow(lines: Lines, path: Path) -> List[Any]:
    import pyarrow.json as pj

    rows = pj.read_json(io.BytesIO(("\n".join(lines) + "\n").encode())).to_pylist()
    return [_dig(r, path) for r in rows]


def _duck_file(lines: Lines, path: Path, ignore_errors: bool = False) -> List[Any]:
    import duckdb

    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "x.json")
        open(f, "w").write("\n".join(lines) + "\n")
        col = '"' + path[0] + '"' + "".join(f"['{k}']" for k in path[1:])
        q = f"select {col} from read_json('{f}'{', ignore_errors=true' if ignore_errors else ''})"
        return [r[0] for r in duckdb.sql(q).fetchall()]


def _duck_extract(lines: Lines, path: Path) -> List[Any]:
    import duckdb

    con = duckdb.connect()
    jp = "$." + ".".join(path)
    out = con.execute("select json_extract(unnest(?), ?)", [lines, jp]).fetchall()
    return [None if r[0] is None else json.loads(r[0]) for r in out]


def _cli(cmd: List[str]) -> Callable[[Lines, Path], List[Any]]:
    def run(lines: Lines, path: Path) -> List[Any]:
        p = subprocess.run(cmd, input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=60)
        if p.returncode:
            raise RuntimeError(p.stderr.strip()[:120])
        return [_dig(json.loads(s), path) for s in p.stdout.splitlines() if s.strip()]

    return run


NODE = "require('readline').createInterface({input:process.stdin}).on('line',l=>{if(l.trim())console.log(JSON.stringify(JSON.parse(l)))})"

# Re-serialising with JSON.stringify / jq -c gives back the value each one kept: neither emits a duplicate.
READERS: Dict[str, Callable[[Lines, Path], List[Any]]] = {
    "json": _per_line(json.loads),
    "ujson": _per_line(_ujson),
    "pydantic": _per_line(_pydantic),
    "PyYAML": _per_line(_pyyaml),
    "pandas read_json": _pandas,
    "polars read_ndjson": _polars,
    "duckdb json_extract": _duck_extract,
    "duckdb ignore_errors": lambda lines, path: _duck_file(lines, path, True),
    "json + strict hook": _per_line(_strict),
    "ruamel.yaml": _per_line(_ruamel),
    "pyarrow read_json": _pyarrow,
    "duckdb read_json": _duck_file,
}
# The two command-line readers join only where the binary exists (Colab has node, not always jq).
if shutil.which("node"):
    READERS = {**dict(list(READERS.items())[:4]), "Node JSON.parse": _cli(["node", "-e", NODE]),
               **dict(list(READERS.items())[4:])}
if shutil.which("jq"):
    _i = list(READERS).index("pandas read_json")
    READERS = {**dict(list(READERS.items())[:_i]), "jq 1.6": _cli(["jq", "-c", "."]), **dict(list(READERS.items())[_i:])}

# name -> (document, path, first value, last value). 'case' is NOT a duplicate under the spec: the answer is 2.
FIXTURES: Dict[str, Any] = {
    "top level": ('{"a":1,"a":2}', ["a"], 1, 2),
    "escaped key": ('{"a":1,"\\u0061":2}', ["a"], 1, 2),
    "nested": ('{"o":{"a":1,"a":2}}', ["o", "a"], 1, 2),
    "type change": ('{"a":1,"a":"x"}', ["a"], 1, "x"),
    "null last": ('{"a":1,"a":null}', ["a"], 1, None),
    "case only (not a dup)": ('{"A":1,"a":2}', ["a"], None, 2),
}


def classify(name: str, fixture: str) -> str:
    doc, path, first, last = FIXTURES[fixture]
    try:
        got = READERS[name]([doc], path)[0]
    except Exception:  # noqa: BLE001 - a refusal is a result
        return "refuse"
    if fixture.startswith("case"):
        return "ok" if got == 2 else f"WRONG ({got!r})"
    if got == last and type(got) is type(last):
        return "last"
    if got == first and type(got) is type(first):
        return "first"
    return f"other ({got!r})"


def matrix() -> Dict[str, Dict[str, str]]:
    return {r: {f: classify(r, f) for f in FIXTURES} for r in READERS}


# --- the order feed -------------------------------------------------------------------------------------------
def make_feed(seed: int = 215, n: int = 2000, rate: float = 0.03) -> Dict[str, Any]:
    """Orders as NDJSON. A refund job patches the status by string concatenation, so a refunded order carries
    'status' twice: the original first, the refund last. A third of the patches come from a writer that escapes
    every key character, so the second key is spelled "st\\u0061tus"."""
    rng = np.random.default_rng(seed)
    lines, refunded, amounts = [], [], []
    for i in range(n):
        amt = round(float(rng.lognormal(4.0, 0.8)), 2)
        st = str(rng.choice(["paid", "shipped", "delivered"]))
        body = f'{{"id":{100000 + i},"amount":{amt},"status":"{st}"'
        patched = bool(rng.random() < rate)
        if patched:
            key = '"st\\u0061tus"' if rng.random() < 1 / 3 else '"status"'
            body += f',{key}:"refunded"'
        lines.append(body + "}")
        refunded.append(patched)
        amounts.append(amt)
    return {"lines": lines, "refunded": refunded, "amounts": amounts}


def feed_study(feed: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    truth_n = sum(feed["refunded"])
    truth_amt = round(sum(a for a, r in zip(feed["amounts"], feed["refunded"]) if r), 2)
    out: Dict[str, Dict[str, Any]] = {}
    for name, fn in READERS.items():
        try:
            st = fn(feed["lines"], ["status"])
        except Exception as e:  # noqa: BLE001
            out[name] = {"raised": type(e).__name__, "rows": 0, "refunds": 0, "amount": 0.0}
            continue
        ref = [s == "refunded" for s in st]
        out[name] = {
            "raised": "",
            "rows": len(st),
            "refunds": sum(ref),
            "amount": round(sum(a for a, r in zip(feed["amounts"], ref) if r), 2),
        }
    out["_truth"] = {"rows": len(feed["lines"]), "refunds": truth_n, "amount": truth_amt}
    return out


# --- detection ------------------------------------------------------------------------------------------------
GREP = re.compile(r'"(\w+)"\s*:.*"\1"\s*:')


def detect(lines: Lines) -> Dict[str, int]:
    """Two pre-flight checks: a raw-text regex for a repeated quoted key, and json with a strict pairs hook."""
    grep = sum(bool(GREP.search(s)) for s in lines)
    hook = 0
    for s in lines:
        try:
            _strict(s)
        except ValueError:
            hook += 1
    return {"regex": grep, "strict hook": hook}


def check(doc: str) -> Dict[str, Any]:
    """For the app: one document through every reader at the first duplicated top-level key (or the first key)."""
    pairs = json.loads(doc, object_pairs_hook=lambda p: p)
    keys = [k for k, _ in pairs] if isinstance(pairs, list) else []
    dups = sorted({k for k in keys if keys.count(k) > 1})
    key = dups[0] if dups else (keys[0] if keys else None)
    one_line = " ".join(doc.split("\n")).strip()  # newlines are JSON whitespace; NDJSON readers need one line
    rows = []
    for name, fn in READERS.items():
        try:
            got: Any = fn([one_line], [key])[0]
            res = json.dumps(got)
        except Exception as e:  # noqa: BLE001
            res = f"refused ({type(e).__name__})"
        rows.append({"reader": name, "value of " + repr(key): res})
    return {"key": key, "dups": dups, "rows": rows}
