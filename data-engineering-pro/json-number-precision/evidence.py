"""Run the study, write results.json (read by the chart, the app and the tests) and evidence.txt.

Section 6 runs the real Node and jq binaries when they are on PATH; the notebook does not reproduce it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Dict, List

import explore as E

INT_TYPES = ("int64", "Int64", "BIGINT", "uint64", "HUGEINT", "Optional[int]", "int")
NODE = r"""
const lines = require('fs').readFileSync(0, 'utf8').trim().split('\n');
const big = process.argv[1] === 'source';  // with -e, argv[1] is the first extra argument
for (const l of lines) {
  const v = big ? JSON.parse(l, (k, v, c) => (k === 'id' && c && /^-?\d+$/.test(c.source)) ? BigInt(c.source) : v)
                : JSON.parse(l);
  console.log(String(v.id));
}
"""


def external(ids: Dict[str, List[int]]) -> Dict[str, Any]:
    """Real JSON.parse and jq over the clean columns. JSON.parse is compared to the Python double simulation."""
    out: Dict[str, Any] = {}
    runs = []
    if shutil.which("node"):
        ver = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
        runs += [(f"node {ver} JSON.parse", ["node", "-e", NODE, "plain"]),
                 (f"node {ver} reviver + BigInt", ["node", "-e", NODE, "source"])]
    if shutil.which("jq"):
        ver = subprocess.run(["jq", "--version"], capture_output=True, text=True).stdout.strip()
        runs.append((f"{ver} .id", ["jq", "-c", ".id"]))
    for name, cmd in runs:
        row = {}
        for g in E.GROUPS:
            got = subprocess.run(cmd, input=E.ndjson(ids[g], None), capture_output=True, text=True).stdout.split()
            exact_digits = "BigInt" in name  # BigInt prints every digit; a double prints its shortest form
            wrong = sum((int(s) if exact_digits else float(s)) != i for s, i in zip(got, ids[g]))
            same_as_double = all(float(s) == float(i) for s, i in zip(got, ids[g])) and not exact_digits
            row[g] = {"wrong": wrong, "same_as_double": same_as_double}
        out[name] = row
    return out


def cell(d: Dict[str, Any]) -> str:
    if d["raised"]:
        return "raised"
    return "ok" if d["wrong"] == 0 else f"{d['wrong']:,} wrong"


def lines(r: Dict[str, Any], s: Dict[str, Any], ext: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    w = out.append
    rd = r["readers"]
    big = ["boundary", "snowflake"]
    w(f"json-number-precision evidence  ({E.N:,} fictional IDs per group, seeded Generator({E.SEED}); "
      f"a value is wrong when it is null or != the integer sent; 'raised' means the reader refused the file)")
    w("")
    w(f"1. Wrong IDs out of {2 * E.N:,} past 2**53 (boundary + snowflake), per reader and column context")
    w(f"{'reader':<21}" + "".join(f"{c:>15}" for c in E.CONTEXTS))
    for k in E.READERS:
        cells = []
        for c in E.CONTEXTS:
            d = rd[k][c]
            if any(d[g]["raised"] for g in big):
                cells.append("raised")
            else:
                n = sum(d[g]["wrong"] for g in big)
                cells.append("ok" if n == 0 else f"{n:,} wrong")
        w(f"{k:<21}" + "".join(f"{x:>15}" for x in cells))
    small_bad = [k for k in E.READERS for c in E.CONTEXTS if rd[k][c]["small"]["wrong"]]
    exact_all = [k for k in E.READERS if all(rd[k][c][g]["wrong"] == 0 and not rd[k][c][g]["raised"]
                                             for c in E.CONTEXTS for g in E.GROUPS)]
    silent = [(k, c) for k in E.READERS for c in E.CONTEXTS if any(E.silent(rd[k][c][g]) for g in big)]
    w(f"small IDs (< 2**31) wrong in {len(small_bad)} of {len(E.READERS) * len(E.CONTEXTS)} reader-contexts")
    w(f"exact in every context: {len(exact_all)} of {len(E.READERS)} ({', '.join(exact_all)})")
    w(f"silently wrong (no error, no warning): {len(silent)} of {len(E.READERS) * len(E.CONTEXTS)} reader-contexts")
    w("")
    w("2. The column type still says integer while the values are wrong")
    for k in E.READERS:
        for c in E.CONTEXTS:
            d = rd[k][c]["snowflake"]
            if d["wrong"] and d["dtype"] in INT_TYPES:
                w(f"{k:<21}{c:<13} dtype {d['dtype']:<8} {d['wrong']:,} of {E.N:,} snowflake IDs wrong")
    w("")
    w("3. What a wrong ID looks like (JSON.parse (double), clean column)")
    for g in E.GROUPS:
        d = rd["JSON.parse (double)"]["clean"][g]
        w(f"{g:<10} {d['wrong']:>5,} of {E.N:,} wrong; {d['merged']:>4} distinct IDs merged into another; "
          f"worst off by {d['max_off']:,}")
    w("")
    w("4. The first positive integer a clean column gets wrong")
    for k in E.READERS:
        f = r["first"][k]
        w(f"{k:<21}" + ("never below 2**70 (or raises instead)" if f is None else f"{f:,}  (2**{f.bit_length() - 1} + "
                                                                                  f"{f - 2 ** (f.bit_length() - 1)})"))
    w("")
    w(f"5. The documented fix, through the same five columns ({E.N:,} snowflake IDs)")
    w(f"{'fix':<24}" + "".join(f"{c:>15}" for c in E.CONTEXTS))
    for k in E.FIXES:
        w(f"{k:<24}" + "".join(f"{cell(r['fixes'][k][c]):>15}" for c in E.CONTEXTS))
    w("")
    w(f"6. Real binaries, clean column (wrong of {E.N:,}; 'double' = identical to the Python double simulation)")
    for k, row in ext.items():
        w(f"{k:<34}" + "   ".join(f"{g} {d['wrong']:,}{' double' if d['same_as_double'] else ''}"
                                  for g, d in row.items()))
    if not ext:
        w("node and jq not on PATH; skipped")
    w("")
    w(f"7. Snowflake IDs merged with a null in the column, over {s['n']} more seeds")
    for k, (lo, hi) in s["range"].items():
        if hi:
            w(f"{k:<21}{lo}-{hi} of {E.N:,} merged")
    return out


def main() -> None:
    r = E.study()
    s = E.seeds()
    ext = external(E.make_ids())
    json.dump({"study": r, "seeds": s, "external": ext}, open("results.json", "w"), indent=1)
    text = "\n".join(lines(r, s, ext)) + "\n"
    open("evidence.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
