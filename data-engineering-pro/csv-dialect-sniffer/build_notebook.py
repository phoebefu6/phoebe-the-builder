"""Generates demo.ipynb with sniff.py, evidence.py and make_chart.py embedded.

The modules are embedded as JSON-encoded string literals rather than triple-quoted
blocks: each contains docstrings, so any triple-quote wrapper terminates early.
Embedding rather than importing keeps the notebook runnable on Colab with no clone
step, and keeps one copy of the logic so nothing drifts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-engineering-pro/csv-dialect-sniffer"

HERE = Path(__file__).parent


def md(text: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def embed(name: str) -> str:
    src = (HERE / name).read_text()
    return (
        f"_src = {json.dumps(src)}\n"
        f"from pathlib import Path\n"
        f'Path("{name}").write_text(_src)\n'
        f'print("wrote {name}:", len(_src.splitlines()), "lines")\n'
    )


CELLS: List[Dict[str, Any]] = [
    md(
        f"""# Every export has a different delimiter

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

**Day 142 - Data Engineering Pro.**

`csv.Sniffer().sniff()` returns a `Dialect` or raises `csv.Error`. It has no third
answer. So a file that two different dialects parse *perfectly* - every record the
same width, nothing ragged, no exception - gets one of them, silently, and the
other one is never mentioned.

This notebook builds a detector that reports **every** viable dialect and
separates three verdicts: **unambiguous** (one candidate parses cleanly),
**contested** (several do, and the bytes cannot choose), **undetermined** (none do).

Eight mechanisms, each measured on a bundled sample file:

| | mechanism | what it costs |
|---|---|---|
| 1 | two clean parses, different widths | wrong column count, silently |
| 2 | a header row settles the contest | 24 bytes decide it |
| 3 | a successful decode is not evidence | mojibake with no exception |
| 4 | the C1 test, and its exact blind spot | half of accented letters |
| 5 | a BOM inside a column name | `KeyError` on a column that prints fine |
| 6 | `has_header()` answering the undecidable | a year counted as a measurement |
| 7 | sniffing a prefix | at 128 bytes it returns the letter `i` |
| 8 | the wrong quotechar | one record deleted at full width |

Standard library only for the core: `csv`, `io`, `re`, `codecs`, `collections`.
"""
    ),
    md("## Setup\n\nWrite the three modules to disk, so this runs on Colab with no clone step."),
    code(embed("sniff.py")),
    code(embed("evidence.py")),
    code(embed("make_chart.py")),
    md(
        """## 1. Two parses, both clean, different widths

`sensor.csv` is a four-record export from a German ERP. The delimiter is `;` and
`1,50` is one and a half. Read it the comma way and every record has three fields;
read it the semicolon way and every record has four. Both are 100% consistent.

There is no header row to break the tie, so nothing in the file prefers either."""
    ),
    code("import sniff, evidence\nevidence.exp1_contested()\n"),
    md(
        """## 2. A header row is 24 bytes and it settles the question

The same data with `day;units;price;total` on top is no longer ambiguous. The
header contains no comma, so under the comma dialect it is one field where the
body has three - ragged, ruled out.

Writing a header row is a data-quality control, not a formatting habit."""
    ),
    code("evidence.exp2_header_breaks_tie()\n"),
    md(
        """## 3. A successful decode is not evidence; a failed one is

`UnicodeDecodeError` is a fact about the bytes: those bytes are not valid in that
encoding. Success is much weaker. `latin-1` maps all 256 byte values, so it
decodes every file ever written and its success carries no information at all.

The discriminator that does work: a decode that produces **C1 control characters**
(U+0080-U+009F) is not credible, because text does not contain control codes. That
is exactly the byte range where cp1252 keeps its curly quotes."""
    ),
    code("evidence.exp3_encoding()\n"),
    md(
        """## 4. Where the C1 test is blind, and why it is exactly half

A code point's second UTF-8 byte is `0x80 | (cp & 0x3f)`. So U+00C0-U+00DF land
inside the C1 range and U+00E0-U+00FF do not: upper-case accented letters are
caught, lower-case ones are not. `Ü` read as latin-1 is disqualified; `ü` is not.

One caught character anywhere in the file disqualifies latin-1 for the whole file,
so on real prose the test usually fires. On lower-case-only data it does not, and
saying so is part of the report."""
    ),
    code(
        """letters = [chr(c) for c in range(0x00C0, 0x0180)]
caught = [c for c in letters if any(0x80 <= b <= 0x9F for b in c.encode("utf-8")[1:])]
print("caught:", len(caught), "of", len(letters))
print("  ", "".join(caught[:24]))
print("missed:", len(letters) - len(caught))
print("  ", "".join([c for c in letters if c not in caught][:24]))
"""
    ),
    md(
        """## 5. The BOM that becomes part of a column name

A UTF-8 byte order mark read with `encoding="utf-8"` instead of `"utf-8-sig"`
stays in the text as U+FEFF, glued to the first column name. It has zero width, so
the name prints correctly, renders correctly in a dataframe, and compares unequal
to itself."""
    ),
    code("evidence.exp4_bom()\n"),
    md(
        """## 6. `has_header()` returns a bool for a question the file does not answer

There is exactly one decidable direction. A first row that is text where the body
is numeric or dated cannot be a data row, so `header` is provable. The converse is
not: a first row whose types match the body is consistent with a data row *and*
with a header whose labels happen to be numbers.

`2019,2020,2021` is that case, and it is common - any year-per-column pivot export
looks like this. So this module has no `no_header` verdict at all."""
    ),
    code("evidence.exp5_header_undecidable()\n"),
    md(
        """## 7. Sniffing a prefix, and the letter `i`

Sniffing the first kilobyte of a large file is the normal thing to do. On
`late.csv`, `_guess_delimiter` falls through to *any* character with a consistent
per-line frequency when none of its preferred candidates has one - and returns a
`Dialect` whose delimiter is a letter from the word `Widget`."""
    ),
    code("evidence.exp6_sample_size()\n"),
    md(
        """## 8. Counting records, and the circularity underneath it

To know whether a newline terminates a record or sits inside a quoted field you
need the quotechar, which is part of the dialect you are trying to detect. So
`scan_terminators()` takes the quotechar as an argument rather than pretending it
can be inferred first."""
    ),
    code("evidence.exp7_line_counting()\n"),
    md(
        """## 9. The wrong quotechar deletes a row and keeps the width

Dutch surnames begin with an apostrophe. Read as a quotechar, the apostrophe on
`'t Hooft` opens a field that runs to the apostrophe on `'s Gravesande`, absorbing
the record terminator between them. Four records become three.

The column count does not change, so every check that watches the column count
passes."""
    ),
    code("evidence.exp8_quotechar()\n"),
    md("## The figure\n\nSix panels, every value computed by `sniff.py`."),
    code(
        """import matplotlib.pyplot as plt
from IPython.display import Image, display
import make_chart as mc

fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
fig.patch.set_facecolor("white")
panels = [mc.panel_candidates, mc.panel_header_tie, mc.panel_encoding,
          mc.panel_c1_coverage, mc.panel_sample_size, mc.panel_line_counts]
for ax, fn in zip(axes.ravel(), panels):
    ax.set_facecolor("white")
    fn(ax)
fig.tight_layout()
fig.subplots_adjust(hspace=0.55, wspace=0.28)
fig.savefig("sniff_audit_nb.png", dpi=140, facecolor="white")
plt.close(fig)
display(Image("sniff_audit_nb.png"))
"""
    ),
    md(
        """## The ledger

Ten failure modes across eleven sample files. Nine of the ten produce a plausible
answer with no exception, and all ten are reproducible - re-running the load gives
the same wrong number, so a reconciliation against yesterday agrees."""
    ),
    code("evidence.ledger()\n"),
    md(
        """## Auditing one file

`audit()` returns everything the bytes determine and everything they do not, in
one object. `decided` is `True` only when the delimiter is unambiguous, the
encoding is settled, and the header question is answerable."""
    ),
    code(
        """for name, raw in sniff.sample_files().items():
    a = sniff.audit(raw, name)
    flag = "OK " if a.decided else "?? "
    print("{0}{1:<18} {2:<13} enc={3:<10} header={4}".format(
        flag, name, a.delimiter.status, a.encoding.verdict,
        a.header.status if a.header else "-"))
    for note in a.notes:
        print("      - " + note[:96])
"""
    ),
    md(
        """## Try your own

Paste a few lines of a real export below. The interesting cases are European
exports, anything from a spreadsheet on a Mac, and anything with a header made of
dates or years."""
    ),
    code(
        """text = '''region;units;price
north;12;1,50
south;8;2,25
'''

v = sniff.classify_delimiter(text)
print("status :", v.status)
print("reason :", v.reason)
for s in v.viable:
    print("   viable:", s.label, "->", s.records, "records x", s.modal, "fields")
print("Sniffer:", repr(sniff.sniffer_says(text)))

rows = [r for r in sniff.parse(text, v.preferred.delimiter, v.preferred.quotechar) if r]
h = sniff.classify_header(rows, text)
print("header :", h.status, "-", h.reason)

# And on bytes, to exercise the encoding probe:
# r = sniff.probe_encoding(open("your_file.csv", "rb").read())
# print(r.verdict, "|", r.reason)
"""
    ),
    md(
        f"""## What to take from this

- **Enumerate, then judge.** A detector that returns one answer cannot tell you it
  had two. Enumerating candidates costs one parse per candidate and buys a verdict
  you can act on.
- **Asymmetric evidence is normal.** A failed decode is a fact; a successful one
  often is not. A text-over-numeric first row proves a header; a matching first row
  proves nothing. Build the API around which direction is decidable.
- **Name the tie-breaks.** Column count, then record count, then delimiter order
  are preferences, not detections. They belong in the output, not in a comment.
- **Report the blind spot with the test.** The C1 heuristic catches exactly half of
  accented letters, and which half is derivable from the encoding, not measured.
- **Sample size is part of the answer.** Any prefix-based detection should report
  the prefix it used.

Reproduce everything here:

```bash
python3 test_sniff.py   # 46 tests over the core
python3 evidence.py     # every table above
python3 make_chart.py   # the six-panel figure
```

Part of [phoebe-the-builder](https://github.com/{REPO}) - Day 142, Data Engineering Pro.
The Streamlit version puts the verdict first and the parsed table last, so an
undecidable file cannot be mistaken for a decided one:

```bash
pip install -r requirements.txt
streamlit run app.py
```
"""
    ),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    out = HERE / "demo.ipynb"
    out.write_text(json.dumps(NOTEBOOK, indent=1))
    print("wrote {0}: {1} cells".format(out.name, len(CELLS)))


if __name__ == "__main__":
    main()
