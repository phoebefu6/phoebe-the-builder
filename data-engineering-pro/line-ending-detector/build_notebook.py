"""Build demo.ipynb.

Self-contained: the notebook does not import lineends.py. The corpus and the
splitters are *extracted verbatim* from it, so the notebook cannot drift from
the engine, and the analyses are written fresh in notebook style. A final cell
asserts every headline count against the README.
"""

from __future__ import annotations

import io
import json
import re
from typing import List


def extract(names: List[str]) -> str:
    src = io.open("lineends.py", encoding="utf-8").read()
    lines = src.splitlines()
    starts = {}
    for i, line in enumerate(lines):
        for n in names:
            if re.match(rf"^(def {n}\(|class {n}\b|{n}(: [^=]+)? = )", line):
                starts.setdefault(n, i)
    out = []
    for n in names:
        if n not in starts:
            raise KeyError(n)
        i = starts[n]
        while i > 0 and lines[i - 1].startswith("@"):
            i -= 1
        j = starts[n] + 1
        while j < len(lines) and (
            not lines[j] or lines[j][0] in " )}]#" or lines[j].startswith(("    ", "\t"))
        ):
            j += 1
        out.append("\n".join(lines[i:j]).rstrip())
    return "\n\n\n".join(out)


CORPUS_CELL = (
    "from __future__ import annotations\n\n"
    "from dataclasses import dataclass\n"
    "from typing import Dict, Tuple\n\n"
    'CR, LF, CRLF = b"\\r", b"\\n", b"\\r\\n"\n\n\n' + extract(["Blob", "CORPUS"])
)

SPLITTER_CELL = (
    "import csv\nimport io\n"
    "from typing import Callable, List, Optional, Sequence\n\n\n"
    + extract(
        [
            "_strip_bom",
            "split_lf",
            "split_wc",
            "split_universal",
            "split_newline_empty",
            "split_splitlines",
            "split_bytes_splitlines",
            "split_java",
            "split_js",
            "split_csv",
            "split_git_text_auto",
            "Splitter",
            "SPLITTERS",
        ]
    )
)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


CELLS = [
    md(
        """# Line-Ending Detector: a file has no lines in it

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-engineering-pro/line-ending-detector/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-engineering-pro/line-ending-detector/demo.ipynb)

**Day 151 of the FDE portfolio.** A file has bytes. *Lines* are produced by a
splitter, and every runtime ships a different one:

- `wc -l` counts LF bytes, so an unterminated last line is not a line.
- Python text mode rewrites CRLF and lone CR to LF **before your code sees the string**.
- `str.splitlines()` breaks on eleven boundaries, including VT, FF, NEL and U+2028.
- `csv.reader` keeps a CRLF that sits inside quotes, because there it is data.

So three questions have different answers per runtime: **how many lines is
this file**, **what is on each line**, and **is reading and writing it back
the identity**.

What this notebook builds:

1. fifteen byte blobs, each one an export somebody actually receives
2. ten splitters, lifted verbatim from the engine
3. the line-count matrix, and the four verdicts
4. the carriage return that survives into a parsed value
5. read-then-write, and the rewrites that land inside a value
6. the one-field edit that shows up as a whole-file diff
7. streaming: a CRLF straddling a chunk boundary
8. "detect the line ending", which for a third of these files has no answer
"""
    ),
    md(
        """## 1. The files

Every blob below is ordinary. The interesting ones are not corrupt - they are
what you get from a Windows export, an old Mac, a form field somebody pasted
into, or an FTP transfer in ASCII mode.
"""
    ),
    code(
        CORPUS_CELL
        + """


print(f"{len(CORPUS)} blobs\\n")
for b in CORPUS:
    print(f"{b.id:2d}  {b.label:22s} {len(b.data):3d} bytes  {b.note}")
    print(f"    {b.one_line}")"""
    ),
    md(
        """## 2. Ten splitters

Lifted verbatim from `lineends.py`. Four are the standard library doing
exactly what it documents; the rest are faithful models of another runtime's
reader.
"""
    ),
    code(
        SPLITTER_CELL
        + """


SPLITTER_BY_KEY = {s.key: s for s in SPLITTERS}


def lines(blob, sp):
    return sp.fn(blob.data)


def line_count(blob, sp):
    return len(lines(blob, sp))


for s in SPLITTERS:
    print(f"  {s.key:18s} {s.models}")"""
    ),
    md("""## 3. How many lines is this file?"""),
    code(
        '''import pandas as pd

counts = pd.DataFrame(
    [[line_count(b, s) for s in SPLITTERS] for b in CORPUS],
    index=[b.label for b in CORPUS],
    columns=[s.key for s in SPLITTERS],
)
counts["min"] = counts.min(axis=1)
counts["max"] = counts.max(axis=1)
drifting = counts[counts["min"] != counts["max"]]
print(f"{len(drifting)} of {len(CORPUS)} files get a different line COUNT depending on "
      f"the reader\\n")
counts'''
    ),
    md(
        """### Four verdicts

- `agreed` - same count and same bytes from all ten
- `content-drift` - same count, different bytes: every count-based check passes
- `count-drift` - they do not agree how many lines there are
- `data-split` - a terminator inside a value becomes a row
"""
    ),
    code(
        '''from enum import Enum


class Verdict(str, Enum):
    AGREED = "agreed"
    CONTENT_DRIFT = "content-drift"
    COUNT_DRIFT = "count-drift"
    DATA_SPLIT = "data-split"


def verdict(blob):
    counts_ = {line_count(blob, s) for s in SPLITTERS}
    csv_n = line_count(blob, SPLITTER_BY_KEY["csv_reader"])
    others = {line_count(blob, s) for s in SPLITTERS if s.key != "csv_reader"}
    if len(counts_) > 1 and csv_n < min(others):
        return Verdict.DATA_SPLIT
    if len(counts_) > 1:
        return Verdict.COUNT_DRIFT
    if len({tuple(lines(blob, s)) for s in SPLITTERS}) > 1:
        return Verdict.CONTENT_DRIFT
    return Verdict.AGREED


verdicts = {v: [b.label for b in CORPUS if verdict(b) is v] for v in Verdict}
for v, labels in verdicts.items():
    print(f"{v.value:<14s} {len(labels):2d}  {', '.join(labels)}")'''
    ),
    md(
        """One file in fifteen is read identically by every runtime here, and it
is the boring one: LF, terminated, nothing exotic in a value.

## 4. The carriage return that is still in the value
"""
    ),
    code(
        '''def trailing_cr(blob, sp):
    return [ln for ln in lines(blob, sp) if ln.endswith(CR)]


cr = {s.key: sum(len(trailing_cr(b, s)) for b in CORPUS) for s in SPLITTERS}
for k, v in cr.items():
    print(f"  {k:18s} {v:3d}{'   <-- CR-blind' if v else ''}")

b = next(x for x in CORPUS if x.label == "crlf-only")
raw = trailing_cr(b, SPLITTER_BY_KEY["split_lf"])[1]
clean = lines(b, SPLITTER_BY_KEY["py_universal"])[1]
print(f"\\ncrlf-only, line 2:")
print(f"  split_lf      {raw!r}")
print(f"  py_universal  {clean!r}")
print(f"  printed       {raw.decode()}| vs {clean.decode()}|")
print(f"  equal?        {raw == clean}")
try:
    int(raw.split(b",")[-1])
except ValueError as exc:
    print(f"  int(last field) -> ValueError: {exc}")'''
    ),
    md(
        """## 5. When the terminator is the data - and when it is invented

`csv.reader` is the only splitter here that knows a quoted CRLF is a value.
`str.splitlines()` goes the other way and finds boundaries no terminator set
contains.
"""
    ),
    code(
        '''b = next(x for x in CORPUS if x.label == "crlf-inside-quotes")
print(b.one_line, "\\n")
for key in ("csv_reader", "py_universal"):
    got = lines(b, SPLITTER_BY_KEY[key])
    print(f"  {key:14s} {len(got)} rows: {[x.decode() for x in got]}")

print("\\nsplitlines() inventing rows:")
for x in CORPUS:
    a = line_count(x, SPLITTER_BY_KEY["str_splitlines"])
    c = line_count(x, SPLITTER_BY_KEY["py_universal"])
    if a != c:
        print(f"  {x.label:<18s} splitlines {a}, reader {c}   ({x.note})")
print("\\nstr vs bytes splitlines on the NEL file:",
      line_count(next(x for x in CORPUS if x.label == "nel-u0085"),
                 SPLITTER_BY_KEY["str_splitlines"]),
      "vs",
      line_count(next(x for x in CORPUS if x.label == "nel-u0085"),
                 SPLITTER_BY_KEY["bytes_splitlines"]))'''
    ),
    md("""## 6. Read it, write it back, compare the bytes"""),
    code(
        '''def roundtrip(blob, sp, write_with=LF):
    out = write_with.join(lines(blob, sp))
    if blob.data.endswith((LF, CR)):
        out += write_with
    changed = out != _strip_bom(blob.data)
    before = line_count(blob, SPLITTER_BY_KEY["csv_reader"])
    after = len(split_csv(out))
    return changed, before != after, before, after


rt = [(b, s) + roundtrip(b, s) for b in CORPUS for s in SPLITTERS]
changed = sum(1 for r in rt if r[2])
inside = [r for r in rt if r[3]]
print(f"{len(rt)} runs: bytes changed in {changed}, CSV row count moved in {len(inside)}\\n")
for b, s, _c, _i, before, after in inside:
    print(f"  {b.label:<22s} {s.key:<18s} {before} rows -> {after} rows")'''
    ),
    md(
        """A change at the end of a line is formatting. A change that moves the
row count landed inside a value. Text mode is a transformation, not a read.

## 7. The diff that says every line changed
"""
    ),
    code(
        '''def diff_lines(before, after, sp):
    a, b_ = sp.fn(before), sp.fn(after)
    return sum(1 for x, y in zip(a, b_) if x != y) + abs(len(a) - len(b_))


sp_lf = SPLITTER_BY_KEY["split_lf"]
blast = []
for b in CORPUS:
    edited = b.data.replace(b"Alice", b"Alicia").replace(b"line two", b"line 2")
    alone = diff_lines(b.data, edited, sp_lf)
    normalised = diff_lines(b.data, edited.replace(CRLF, LF).replace(CR, LF), sp_lf)
    blast.append({"file": b.label, "edit alone": alone, "edit + normalise": normalised,
                  "": "  <-- whole file" if normalised > alone else ""})
pd.DataFrame(blast)'''
    ),
    md("""## 8. Streaming, concatenation, and detection"""),
    code(
        '''def naive_chunk_reader(data, chunk):
    out, carry = [], b""
    for i in range(0, len(data), chunk):
        block = carry + data[i:i + chunk]
        parts = block.split(LF)
        carry = parts.pop()
        out.extend(p[:-1] if p.endswith(CR) else p for p in parts)
    if carry:
        out.append(carry[:-1] if carry.endswith(CR) else carry)
    return out


uni = SPLITTER_BY_KEY["py_universal"]
drift = [(b.label, n, len(lines(b, uni)), len(naive_chunk_reader(b.data, n)))
         for b in CORPUS for n in (4, 8, 16)
         if naive_chunk_reader(b.data, n) != lines(b, uni)]
print(f"{len(drift)} (file, chunk size) combinations where a chunked reader is wrong")
b = next(x for x in CORPUS if x.label == "crlf-only")
print(f"\\ncrlf-only at chunk 8: {naive_chunk_reader(b.data, 8)}")
print(f"           correct:   {lines(b, uni)}")

parts = sum(line_count(b, uni) for b in CORPUS)
joined = len(uni.fn(b"".join(b.data for b in CORPUS)))
print(f"\\ncat: {parts} lines in the parts, {joined} after concatenation -> {parts - joined} lost")


def histogram(data):
    crlf = data.count(CRLF)
    return {"CRLF": crlf, "LF": data.count(LF) - crlf, "CR": data.count(CR) - crlf}


def first_seen(data):
    for i in range(len(data)):
        if data[i:i + 2] == CRLF:
            return "CRLF"
        if data[i:i + 1] == LF:
            return "LF"
        if data[i:i + 1] == CR:
            return "CR"
    return None


rows = []
for b in CORPUS:
    h = histogram(b.data)
    present = [k for k, v in h.items() if v]
    rows.append({"file": b.label, **h, "first seen": first_seen(b.data),
                 "majority": max(h, key=lambda k: h[k]),
                 "strict": present[0] if len(present) == 1 else "refuses"})
detect = pd.DataFrame(rows)
no_answer = detect[detect.strict == "refuses"]
print(f"\\n{len(no_answer)} of {len(CORPUS)} files have no single honest answer")
detect'''
    ),
    md("""## 9. The picture"""),
    code(
        '''import matplotlib
import matplotlib.pyplot as plt
import numpy as np

INK, WARM, COOL, GREEN, SAND = "#1d2733", "#c2571a", "#2d5a68", "#2f6b39", "#e8d9c0"
plt.rcParams.update({"font.size": 8, "figure.facecolor": "white"})
KEYS = [s.key for s in SPLITTERS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

vals = [cr[k] for k in KEYS]
ax1.bar(range(len(KEYS)), vals, color=[WARM if v else GREEN for v in vals],
        edgecolor=INK, lw=0.4)
for i, v in enumerate(vals):
    ax1.text(i, v + 0.3, str(v), ha="center", fontsize=6)
ax1.set_xticks(range(len(KEYS)))
ax1.set_xticklabels(KEYS, rotation=55, ha="right", fontsize=6)
ax1.set_ylabel("lines handed back with a trailing CR")
ax1.set_title("The carriage return that is still in the value", fontsize=9, loc="left")

x = np.arange(len(blast))
ax2.bar(x - 0.2, [r["edit alone"] for r in blast], 0.4, color=COOL, label="the edit alone")
ax2.bar(x + 0.2, [r["edit + normalise"] for r in blast], 0.4, color=WARM,
        label="the same edit, endings normalised")
ax2.set_xticks(x)
ax2.set_xticklabels([r["file"] for r in blast], rotation=55, ha="right", fontsize=6)
ax2.set_ylabel("lines a line-diff calls changed")
ax2.legend(fontsize=6, frameon=False)
ax2.set_title("One field edited, and what the diff says", fontsize=9, loc="left")

fig.tight_layout()
fig.savefig("eol_notebook.png", dpi=150, bbox_inches="tight")
plt.show()'''
    ),
    md(
        """## 10. Summary, and the fix

| mechanism | number |
|---|---|
| files read identically by all ten splitters | 1 of 15 |
| files whose *line count* depends on the reader | 10 of 15 |
| lines handed back with a trailing CR | 40, by 4 of 10 splitters |
| read-then-write runs that change the bytes | 81 of 150 |
| ...of those, runs that move the CSV row count | 11 |
| files whose one-field edit becomes a whole-file diff | 8 of 15 |
| chunk-boundary failures | 15 (file, chunk size) pairs |
| files with no single honest answer to "which line ending?" | 5 of 15 |

The fix, cheapest first:

1. **Read with `newline=''` and a real parser** - for CSV, `csv.reader` over a file opened that way.
2. **Never `split('\\n')`** on a payload you did not write.
3. **Never `splitlines()`** on user-supplied text you intend to treat as records.
4. **Write with one chosen terminator**, and terminate the last line.
5. **Normalise once, in its own commit**, with `.gitattributes` alongside.
6. **Return the histogram, not the terminator** - detection is a diagnostic, not a strategy.
"""
    ),
    code(
        '''checks = {
    "files read identically by all ten": (len(verdicts[Verdict.AGREED]), 1),
    "content-drift files": (len(verdicts[Verdict.CONTENT_DRIFT]), 4),
    "count-drift files": (len(verdicts[Verdict.COUNT_DRIFT]), 9),
    "data-split files": (len(verdicts[Verdict.DATA_SPLIT]), 1),
    "files whose count depends on reader": (len(drifting), 10),
    "lines with a trailing CR": (sum(cr.values()), 40),
    "CR-blind splitters": (sum(1 for v in cr.values() if v), 4),
    "roundtrips that change bytes": (changed, 81),
    "roundtrips that move the row count": (len(inside), 11),
    "whole-file diffs": (sum(1 for r in blast if r["edit + normalise"] > r["edit alone"]), 8),
    "chunk-boundary failures": (len(drift), 15),
    "lines lost by cat": (parts - joined, 1),
    "files with no single EOL answer": (len(no_answer), 5),
}
for label, (got, want) in checks.items():
    print(f"  [{'ok ' if got == want else 'MISMATCH'}] {label:38s} {got} (expected {want})")
assert all(got == want for got, want in checks.values())
print("\\nnotebook agrees with the engine on every headline count")'''
    ),
    md(
        """## Try your own file

Point this at a real export - a partner CSV, a log file, anything that came
from another machine.
"""
    ),
    code(
        '''# with open("your_export.csv", "rb") as fh:
#     data = fh.read()
#
# mine = Blob(0, "yours", data, "")
# print("histogram:", histogram(data))
# for s in SPLITTERS:
#     got = lines(mine, s)
#     bad = sum(1 for ln in got if ln.endswith(CR))
#     print(f"{s.key:18s} {len(got):6d} lines   {bad} with a trailing CR")
# print("verdict:", verdict(mine).value)'''
    ),
    md(
        """---

**Day 151 of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder)** -
[project README](README.md) for all ten mechanisms and the findings table,
`python evidence.py` for every number, `python -m pytest -q` for the 42 tests,
`streamlit run app.py` to paste your own bytes.

Previous days on the same theme - an operation that looks total and is not:
[`duration-parser`](../../automation-suite/duration-parser/) (147),
[`percent-recomputer`](../../analytics-engineering-bi/percent-recomputer/) (148),
[`header-casing`](../../automation-suite/header-casing/) (149),
[`sort-order-drift`](../sort-order-drift/) (150).
"""
    ),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    with io.open("demo.ipynb", "w", encoding="utf-8") as fh:
        json.dump(NB, fh, ensure_ascii=False, indent=1)
    print(f"wrote demo.ipynb ({len(CELLS)} cells)")
